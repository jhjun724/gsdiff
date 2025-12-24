import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix
from ..encoder.gaussian_encoder.utils import linear_relu_ln


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        embed_dims=-1,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        combine_geosem=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        if use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.embed_dims = embed_dims
        if embed_dims > 0:
            self.sem_head = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2, input_dims=embed_dims),
                nn.Linear(embed_dims, num_classes - 1),
            )
        self.combine_geosem = combine_geosem
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        semantics = gaussians.semantics # b, g, c
        opacities = gaussians.opacities # b, g, 1
        if opacities.numel() == 0:
            opacities = torch.ones_like(semantics[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert semantics.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                semantics = torch.cat([torch.zeros_like(semantics[..., :1]), semantics], dim=-1)
            else:
                semantics = torch.cat([semantics, torch.zeros_like(semantics[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            semantics = torch.cat([semantics, empty_sem], dim=1)
            opacities = torch.cat([opacities, self.empty_opa], dim=1)
        elif self.use_localaggprob:
            if self.embed_dims > 0:
                semantics = self.sem_head(gaussians.features)
            assert semantics.shape[-1] == self.num_classes - 1
            semantics = semantics.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                semantics = torch.cat([torch.zeros_like(semantics[..., :1]), semantics], dim=-1)
            else:
                semantics = torch.cat([semantics, torch.zeros_like(semantics[..., :1])], dim=-1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, opacities, semantics, scales, CovInv

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        apply_loss_layers = self.apply_loss_layers

        prediction = []
        bin_logits = []
        density = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']

            means, opacities, semantics, scales, CovInv = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]

            semantics = self.aggregator(
                sampled_xyz.clone().float(), 
                means, 
                opacities.reshape(bs, g),
                semantics,
                scales,
                CovInv) # 1, c, n
            if self.use_localaggprob:
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                else:
                    geosem = semantics[0]
                    
                prediction.append(geosem[None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))
        
        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)
        
        return {
            'pred_occ': prediction,
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation]
        }

