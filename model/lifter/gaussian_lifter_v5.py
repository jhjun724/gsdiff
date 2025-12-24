import torch, torch.nn as nn, math, os
import numpy as np
from einops import rearrange
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from ..utils.sampler import DistributionSampler, DifferentiableSampler
from ..utils.contract import contract_x2s

try:
    from ..ops.pointops import farthest_point_sampling
except:
    print("farthest_point_sampling import error.")


@MODELS.register_module()
class GaussianLifterV5(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        include_rgb=False,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        sampler=dict(type="DistributionSampler", deterministic=True),

        num_samples=64,
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        empty_label=17,
        anchors_per_pixel=1,
        random_sampling=True,
        projection_in=None,
        initializer=None,
        initializer_img_downsample=None,
        pretrained_path=None,
        random_samples=0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.include_rgb = include_rgb
        self.semantics = semantics
        self.semantic_dim = semantic_dim
        self.pretraining = kwargs.get("pretraining", False)
        self.subsampling = kwargs.get("subsampling", False)
        self.contracted = kwargs.get("contracted", -1.0)
        spacing_fn = kwargs.get("spacing_fn", "uniform")

        self.random_samples = random_samples
        if random_samples > 0:
            self.random_anchors = self.init_random_anchors()
                    
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if scale_activation == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(
                0.5 * torch.ones((num_anchor, 1), dtype=torch.float)
            )
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if include_rgb:
            # rgb = torch.randn(num_anchor, 3, dtype=torch.float)
            rgb = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 3), dtype=torch.float))
        else:
            rgb = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([scale, rots, rgb, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.instance_feature = nn.Parameter(
            torch.zeros([num_anchor + random_samples, self.embed_dims]),
            requires_grad=feat_grad,
        )
        projection_in = embed_dims * 4 if projection_in is None else projection_in
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_in, num_samples + 1),
        )
        if sampler['type'] == "DistributionSampler":
            self.sampler_type = 'dist'
            self.sampler = DistributionSampler()
            self.deterministic = sampler.get("deterministic", True)
        elif sampler['type'] == "DifferentiableSampler":
            self.sampler_type = 'diff'
            self.sampler = DifferentiableSampler(
                method=sampler.get("method", "soft"),
                temperature=sampler.get("temperature", 1.0),
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler['type']}")
        self.num_samples = num_samples
        depth_range = kwargs.get("depth_range", [1.0, 80.0])
        bins = torch.linspace(0.0, 1.0, num_samples)
        if spacing_fn == "uniform":
            spacing_fn = lambda x: x
            spacing_fn_inv = lambda x: x
        elif spacing_fn == "linear":
            spacing_fn = lambda x: 1 / x
            spacing_fn_inv = lambda x: 1 / x
        elif spacing_fn == "sqrt":
            spacing_fn = lambda x: torch.sqrt(x)
            spacing_fn_inv = lambda x: x ** 2.
        elif spacing_fn == "log":
            spacing_fn = lambda x: torch.log(x)
            spacing_fn_inv = lambda x: torch.exp(x)
        else:
            raise ValueError(f"Unknown spacing function: {spacing_fn}")
        near = spacing_fn(torch.tensor(depth_range[0]))
        far = spacing_fn(torch.tensor(depth_range[1]))
        spacing_to_euclidean_fn = lambda x : spacing_fn_inv(x * far + (1 - x) * near)
        depth_bins = spacing_to_euclidean_fn(bins)
        self.register_buffer("depth_bins", depth_bins, persistent=False)
        if sampler['type'] == "DifferentiableSampler":
            self.depth_bins = torch.cat([
                self.depth_bins,
                self.depth_bins.new_tensor([depth_range[1] * 1.5])
            ], dim=0)
        self.register_buffer("pc_start", torch.tensor(
            pc_range[:3], dtype=torch.float), persistent=False)
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution
        self.empty_label = empty_label
        self.anchors_per_pixel = anchors_per_pixel
        self.random_sampling = random_sampling
        if initializer is not None:
            self.initialize_backbone = MODELS.build(initializer)
        else:
            self.initialize_backbone = None
        self.initializer_img_downsample = initializer_img_downsample
        
        self.pretrained_path = pretrained_path
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            if 'instance_feature' in ckpt:
                del ckpt['instance_feature']
            if 'anchor' in ckpt:
                del ckpt['anchor']
            self.load_state_dict(ckpt, strict=False)
            print("Gaussian Initializer Weight Loaded Successfully.")

    def init_random_anchors(self):
        num_anchor = self.random_samples

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
        
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if self.include_opa:
            opacity = safe_inverse_sigmoid(
                0.5 * torch.ones((num_anchor, 1), dtype=torch.float)
            )
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if self.include_rgb:
            rgb = torch.randn(num_anchor, 3, dtype=torch.float)
            # rgb = safe_inverse_sigmoid(0.5 * torch.ones((self.num_anchor, 3), dtype=torch.float))
        else:
            rgb = torch.ones((num_anchor, 0), dtype=torch.float)

        if self.semantics:
            semantic_dim = self.semantic_dim
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, rgb, opacity, semantic], dim=-1)
        anchor = nn.Parameter(anchor, True)
        return anchor

    def init_weights(self):
        if self.pretrained_path is not None:
            return
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, metas, **kwargs):
        if self.initialize_backbone is not None:
            b, n = kwargs["imgs"].shape[:2]
            initialize_input = kwargs["imgs"].flatten(0, 1)
            if self.initializer_img_downsample is not None:
                initialize_input = nn.functional.interpolate(
                    initialize_input, scale_factor=self.initializer_img_downsample, 
                    mode='bilinear', align_corners=True)
            secondfpn_out = self.initialize_backbone(initialize_input)
            secondfpn_out = secondfpn_out.unflatten(0, (b, n))
        else:
            secondfpn_out = kwargs["secondfpn_out"]
        
        b, n, _, h, w = secondfpn_out.shape
        feature = rearrange(secondfpn_out, 'b n c h w -> b n h w c')
        logits = self.projection(feature) # b, n, h, w, d + 1

        projection_mat = metas["projection_mat"].inverse() # img2lidar
        u = (torch.arange(w, dtype=feature.dtype, device=feature.device) + 0.5) / w
        v = (torch.arange(h, dtype=feature.dtype, device=feature.device) + 0.5) / h
        uv = torch.stack([
            u[None, :].expand(h, w), v[:, None].expand(h, w)], dim=-1) # h, w, 2
        uv = uv[None, None].expand(b, n, h, w, 2) * metas['image_wh'][:, :, None, None] # b, n, h, w, 2
        if self.sampler_type == "diff":
            uvd = uv.unsqueeze(4).expand(b, n, h, w, self.num_samples+1, 2)
        else:
            uvd = uv.unsqueeze(4).expand(b, n, h, w, self.num_samples, 2)
        uvd1 = torch.cat([uvd, torch.ones_like(uvd)], dim=-1) # b, n, h, w, d, 4
        uvd1[..., :3] = uvd1[..., :3] * self.depth_bins.view(1, 1, 1, 1, -1, 1)
        anchor_pts = projection_mat[:, :, None, None, None] @ uvd1[..., None]
        anchor_pts = anchor_pts.squeeze(-1)[..., :3]
        if self.pretraining:
            anchor_gt = None
        else:
            oob_mask = (
                (anchor_pts[..., 0] <  self.pc_range[0]) |
                (anchor_pts[..., 0] >= self.pc_range[3]) |
                (anchor_pts[..., 1] <  self.pc_range[1]) |
                (anchor_pts[..., 1] >= self.pc_range[4]) |
                (anchor_pts[..., 2] <  self.pc_range[2]) |
                (anchor_pts[..., 2] >= self.pc_range[5])
            )
            anchor_idx = (anchor_pts - self.pc_start.view(1, 1, 1, 1, 1, 3)) / self.voxel_size
            anchor_idx = anchor_idx.to(torch.int)
            anchor_idx[..., 0].clamp_(0, self.occ_resolution[0] - 1)
            anchor_idx[..., 1].clamp_(0, self.occ_resolution[1] - 1)
            anchor_idx[..., 2].clamp_(0, self.occ_resolution[2] - 1)

            occupancy = metas["occ_label"]
            valid_mask = metas["occ_cam_mask"]
            anchor_occ = torch.stack([
                occ[idx[..., 0], idx[..., 1], idx[..., 2]] 
                for occ, idx in zip(occupancy, anchor_idx)
            ])
            anchor_occ[oob_mask] = self.empty_label
            anchor_valid = torch.stack([
                occ[idx[..., 0], idx[..., 1], idx[..., 2]]
                for occ, idx in zip(valid_mask, anchor_idx)
            ])
            anchor_valid[oob_mask] = False
            anchor_gt = (anchor_occ != self.empty_label) & anchor_valid
            anchor_gt = torch.cat(
                [anchor_gt, ~torch.any(anchor_gt, dim=-1, keepdim=True)], dim=-1
            )
        
        pdfs = torch.softmax(logits, dim=-1)
        if self.sampler_type == "dist":
            deterministic = getattr(self, 'deterministic', True)
            index, pdf_i = self.sampler.sample(pdfs, deterministic, self.anchors_per_pixel) # b, n, h, w, a
            disable_mask = (pdfs.argmax(dim=-1, keepdim=True) == self.num_samples).expand(
                -1, -1, -1, -1, self.anchors_per_pixel)
            # disable_mask = index == self.num_samples
            sampled_anchor = self.sampler.gather(
                index.clamp(max=(self.num_samples-1)), anchor_pts
            ) # b, n, h, w, a
            depth_pdfs_norm = pdfs[..., :-1] / (pdfs[..., :-1].sum(dim=-1, keepdim=True) + 1e-8)
        elif self.sampler_type == "diff":
            sampled_anchor, confidence, depth_pdfs_norm = self.sampler.sample(
                pdfs, anchor_pts
            )
            disable_mask = torch.zeros_like(confidence, dtype=torch.bool)
        
        anchor_xyz = []
        for i in range(b):
            if self.pretraining and self.contracted > 0.0:
                anchor_xyz.append(sampled_anchor[i].flatten(0, 3))
                continue
            cur_sampled_anchor = sampled_anchor[i][~disable_mask[i]]
            cur_oob_mask = (
                (cur_sampled_anchor[..., 0] <  self.pc_range[0]) | 
                (cur_sampled_anchor[..., 0] >= self.pc_range[3]) |
                (cur_sampled_anchor[..., 1] <  self.pc_range[1]) | 
                (cur_sampled_anchor[..., 1] >= self.pc_range[4]) |
                (cur_sampled_anchor[..., 2] <  self.pc_range[2]) | 
                (cur_sampled_anchor[..., 2] >= self.pc_range[5])
            )
            scan = cur_sampled_anchor[~cur_oob_mask]
            
            if self.random_sampling:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_ = scan_[
                        np.random.choice(
                            scan_.shape[0],
                            self.num_anchor - scan.shape[0],
                            False
                        )
                    ]
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                else:
                    scan = scan[np.random.choice(scan.shape[0], self.num_anchor, False)]
            else:
                if scan.shape[0] < self.num_anchor:
                    multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                    scan_ = scan.repeat(multi, 1)
                    scan_ = scan_ + torch.randn_like(scan_) * 0.1
                    scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                    scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                    scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                    scan = torch.cat([scan, scan_], 0)
                # breakpoint()
                if self.subsampling:
                    scan = scan[np.random.permutation(scan.shape[0])]
                    num_subsets = 3
                    sublens = torch.linspace(
                        0,
                        scan.shape[0],
                        num_subsets + 1,
                        dtype=torch.int,
                        device=scan.device
                    )[1:]
                    new_sublens = torch.linspace(
                        0,
                        self.num_anchor,
                        num_subsets + 1,
                        dtype=torch.int,
                        device=scan.device
                    )[1:]
                    scanidx = farthest_point_sampling(scan, sublens, new_sublens)
                else:
                    # breakpoint()
                    device = scan.device
                    scanidx = farthest_point_sampling(
                        scan.to(torch.float32), 
                        torch.tensor([scan.shape[0]], device=device, dtype=torch.int),
                        torch.tensor([self.num_anchor], device=device, dtype=torch.int)
                    )
                scan = scan[scanidx, :]
            
            anchor_xyz.append(scan)

            if os.environ.get("DEBUG", 'false') == 'true':
                prefix = 'kitti-'
                #### save pred scan
                np.save(f'{prefix}pred_scan.npy', scan.detach().cpu().numpy())
                #### save gt scan
                np.save('gt_scan_occ.npy', anchor_occ.detach().cpu().numpy())
                np.save('gt_scan_pts.npy', anchor_pts.detach().cpu().numpy())
                #### save gt occupancy
                np.save('gt_occ.npy', metas['occ_label'].detach().cpu().numpy())
                np.save('gt_pts.npy', metas['occ_xyz'].detach().cpu().numpy())

                breakpoint()
        
        anchor_xyz = torch.stack(anchor_xyz)
        if self.contracted > 0.0:
            anchor_xyz = contract_x2s(anchor_xyz, self.pc_range, self.contracted) / 2 + 0.5
            # Clamp to valid range for safe_inverse_sigmoid to prevent gradient explosion
            anchor_xyz = anchor_xyz.clamp(0.0001, 0.9999)
        else:
            anchor_xyz[..., 0] = (anchor_xyz[..., 0] - self.pc_range[0]) \
                / (self.pc_range[3] - self.pc_range[0])
            anchor_xyz[..., 1] = (anchor_xyz[..., 1] - self.pc_range[1]) \
                / (self.pc_range[4] - self.pc_range[1])
            anchor_xyz[..., 2] = (anchor_xyz[..., 2] - self.pc_range[2]) \
                / (self.pc_range[5] - self.pc_range[2])

        if self.xyz_act == "sigmoid":
            anchor_xyz = safe_inverse_sigmoid(anchor_xyz)
        anchor = torch.cat([
            anchor_xyz, torch.tile(self.anchor[None], (b, 1, 1))], dim=-1)
        
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (b, 1, 1))
            anchor = torch.cat([anchor, random_anchors], dim=1)

        instance_feature = torch.tile(
            self.instance_feature[None], (b, 1, 1)
        )
        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor[0].clone(),
            'pixel_logits': logits,
            'pixel_gt': anchor_gt,
            'depth_preds': depth_pdfs_norm,
            'confidence': confidence.permute(0, 1, 4, 2, 3),
        }