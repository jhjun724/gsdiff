from mmengine.registry import MODELS
from mmengine.model import BaseModule
from .utils import linear_relu_ln
import torch.nn as nn, torch


@MODELS.register_module()
class SparseGaussian3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256, 
        include_scale=True,
        include_rot=True,
        include_opa=True,
        include_rgb=False,
        semantics=False,
        semantic_dim=None
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.include_scale = include_scale
        self.include_rot = include_rot
        self.include_rgb = include_rgb
        self.include_opa = include_opa
        self.semantics = semantics

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))
        assert include_scale == include_rot

        self.rgb_start = 3 + int(include_scale) * 3 + int(include_rot) * 4
        self.opa_start = self.rgb_start + int(include_rgb) * 3
        self.xyz_fc = embedding_layer(3)
        if include_scale:
            self.scale_fc = embedding_layer(3)
        if include_rot:
            self.rot_fc = embedding_layer(4)
        if include_rgb:
            self.rgb_fc = embedding_layer(3)
        if include_opa:
            self.opacity_fc = embedding_layer(1)
        if semantics:
            assert semantic_dim is not None
            self.semantics_fc = embedding_layer(semantic_dim)
            self.sem_start = self.opa_start + int(include_opa)
        else:
            semantic_dim = 0
        self.semantic_dim = semantic_dim
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        xyz_feat = self.xyz_fc(box_3d[..., :3])
        scale_feat = self.scale_fc(box_3d[..., 3:6]) if self.include_scale else 0.
        rot_feat = self.rot_fc(box_3d[..., 6:10]) if self.include_rot else 0.
        rgb_feat = self.rgb_fc(
            box_3d[..., self.rgb_start:self.rgb_start+3]
        ) if self.include_rgb else 0.
        opa_feat = self.opacity_fc(
            box_3d[..., self.opa_start:self.opa_start+1]
        ) if self.include_opa else 0.
        sem_feat = self.semantics_fc(
            box_3d[..., self.sem_start:(self.sem_start + self.semantic_dim)]
        ) if self.semantics else 0.

        output = xyz_feat + scale_feat + rot_feat + rgb_feat + opa_feat + sem_feat
        output = self.output_fc(output)
        return output
