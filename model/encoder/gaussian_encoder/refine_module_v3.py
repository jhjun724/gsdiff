from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
from functools import partial
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction, cartesian, reverse_cartesian
from ...utils.safe_ops import safe_sigmoid


class RGBDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_size=256, n_blocks=5):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)

        self.activation = nn.ReLU()

    def forward(self, points, point_feats):
        x = self.fc_p(points)
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        # x = torch.sigmoid(x)
        return x


@MODELS.register_module()
class SparseGaussian3DRefinementModuleV3(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        include_rgb=False,
        semantics_activation='softmax',
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        contracted=-1.0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0

        self.rgb_start = 10
        self.opa_start = 10 + int(include_rgb) * 3
        self.output_dim = self.opa_start + int(include_opa) + semantic_dim
        self.semantic_start = self.opa_start + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_rgb = include_rgb
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float), False)
        self.get_xyz = partial(
            cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"), contracted=contracted)
        self.reverse_xyz = partial(
            reverse_cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"), contracted=contracted)
        
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))

        if not include_rgb:
            self.rgb_decoder = RGBDecoder(embed_dims)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        output = self.layers(instance_feature + anchor_embed)

        #### for xyz
        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1.) * self.unit_xyz[None, None]
        original_xyz = self.get_xyz(anchor[..., :3])
        anchor_xyz = original_xyz + delta_xyz
        anchor_xyz = self.reverse_xyz(anchor_xyz)

        #### for scale
        anchor_scale = output[..., 3:6]

        #### for rotation
        anchor_rotation = output[..., 6:10]
        anchor_rotation = torch.nn.functional.normalize(anchor_rotation, 2, -1)

        ### for rgb
        if self.include_rgb:
            anchor_rgb = output[..., self.rgb_start:(self.rgb_start + int(self.include_rgb) * 3)]
        else:
            anchor_rgb = self.rgb_decoder(anchor_xyz, instance_feature + anchor_embed)

        #### for opacity
        anchor_opa = output[..., self.opa_start:(self.opa_start + int(self.include_opa))]

        #### for semantic
        anchor_sem = output[..., self.semantic_start:(self.semantic_start + self.semantic_dim)]

        output = torch.cat([
            anchor_xyz, anchor_scale, anchor_rotation, anchor_rgb, anchor_opa, anchor_sem], dim=-1)
        
        xyz = self.get_xyz(anchor_xyz)

        if self.scale_act == 'sigmoid':
            scale = safe_sigmoid(anchor_scale)
        scale = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        
        if self.semantics_activation == 'softmax':
            semantics = anchor_sem.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(anchor_sem)
        else:
            semantics = anchor_sem
        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=scale,
            rotations=anchor_rotation,
            opacities=safe_sigmoid(anchor_opa),
            semantics=semantics,
            original_means=original_xyz,
            delta_means=delta_xyz,
            colors=safe_sigmoid(anchor_rgb),
            features=instance_feature
        )
        return output, gaussian
