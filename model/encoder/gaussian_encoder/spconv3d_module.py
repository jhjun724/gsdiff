from mmengine.registry import MODELS
from mmengine.model import BaseModule
import spconv.pytorch as spconv
import torch.nn as nn, torch
from functools import partial
from .utils import spherical2cartesian, cartesian
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class SparseConv3D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        xyz_activation="sigmoid",
        use_out_proj=False,
        kernel_size=5,
        use_multi_layer=False,
        init_cfg=None,
        contracted=-1.0,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        if use_multi_layer:
            self.layer = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                nn.LayerNorm(embed_channels),
                nn.ReLU(True),
            )        
        else:
            self.layer = spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.contracted = contracted
        self.get_xyz = partial(cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"), contracted=contracted)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))
        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.register_buffer('spatial_shape', spatial_shape.to(torch.int32))
        if contracted > 0:
            self.spatial_shape = (self.spatial_shape / contracted + 1).to(torch.int32)
            self.pc_range = torch.tensor([-75.0, -75.0, -7.5, 75.0, 75.0, 4.5])

    def forward(self, instance_feature, anchor):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        bs, g, _ = instance_feature.shape

        # sparsify
        anchor_xyz = anchor[..., :3]
        if False and self.contracted > 0:
            if self.get_xyz.keywords.get('use_sigmoid', False):
                anchor_xyz = safe_sigmoid(anchor_xyz)
            else:
                min_xyz = anchor_xyz.min(dim=1, keepdim=True).values
                max_xyz = anchor_xyz.max(dim=1, keepdim=True).values
                anchor_xyz = (anchor_xyz - min_xyz) / (max_xyz - min_xyz + 1e-6)
            indices = anchor_xyz.flatten(0, 1)  * self.spatial_shape[None, :]
        else:
            # anchor_xyz = self.get_xyz(anchor_xyz).flatten(0, 1)
            # anchor_xyz[..., 0] = torch.clamp(anchor_xyz[..., 0], self.pc_range[0], self.pc_range[3])
            # anchor_xyz[..., 1] = torch.clamp(anchor_xyz[..., 1], self.pc_range[1], self.pc_range[4])
            # anchor_xyz[..., 2] = torch.clamp(anchor_xyz[..., 2], self.pc_range[2], self.pc_range[5])
            # indices = anchor_xyz - self.pc_range[None, :3]
            # indices = indices / self.grid_size[None, :] # bg, 3
            anchor_xyz = self.get_xyz(anchor_xyz).flatten(0, 1)
            indices = anchor_xyz - anchor_xyz.min(0, keepdim=True)[0]
            # anchor_xyz = self.get_xyz(anchor_xyz)
            # indices = (anchor_xyz - anchor_xyz.min(1, keepdim=True)[0]).flatten(0, 1)

        indices = indices.to(torch.int32)
        # indices[..., 0] = torch.clamp(indices[..., 0], 0, self.spatial_shape[0] - 1)
        # indices[..., 1] = torch.clamp(indices[..., 1], 0, self.spatial_shape[1] - 1)
        # indices[..., 2] = torch.clamp(indices[..., 2], 0, self.spatial_shape[2] - 1)
        batched_indices = torch.cat([
            torch.arange(
                bs, device=indices.device, dtype=torch.int32
            ).reshape(bs, 1, 1).expand(-1, g, -1).flatten(0, 1),
            indices
        ], dim=-1)

        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=self.spatial_shape,
            batch_size=bs)

        output = self.layer(input)
        output = output.features.unflatten(0, (bs, g))

        return self.output_proj(output)
