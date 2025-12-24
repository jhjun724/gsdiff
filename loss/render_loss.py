import torch
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


def assign_depth_target(pts, img_metas):
    lidar2img = img_metas['lidar2img']
    num_cams = lidar2img.shape[1]

    depth_gt = []
    depth_mask = []
    for bs_idx in range(len(pts)):
        i_pts = pts[bs_idx].tensor.to(lidar2img.device)
        i_pts = i_pts[i_pts[:, :2].norm(dim=-1) > 3.0]
        i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)

        i_lidar2img = lidar2img[bs_idx].type(i_pts.dtype)
        i_pts_cam = torch.matmul(
            i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
        ).squeeze(-1)

        eps = 1e-5
        i_pts_depth = i_pts_cam[..., 2].clone()
        i_pts_mask = i_pts_depth > eps
        i_pts_cam = i_pts_cam[..., :2] / torch.maximum(
            i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
        )

        # (N*C, 3) [(H, W, 3), ...]
        Hs = img_metas['cam_params'][bs_idx][0]['height']
        Ws = img_metas['cam_params'][bs_idx][0]['width']

        # (N*C, M)
        i_pts_mask = (
            i_pts_mask
            & (i_pts_cam[..., 0] > 0)
            & (i_pts_cam[..., 0] < Ws - 1)
            & (i_pts_cam[..., 1] > 0)
            & (i_pts_cam[..., 1] < Hs - 1)
            & (i_pts_depth > 0.0)
        )

        depth_map = i_pts_depth.new_ones((num_cams, Hs, Ws)) * -1
        for c_idx in range(len(i_pts_mask)):
            # (M,) -> (Q,)
            j_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
            coor, depth = (
                torch.round(i_pts_cam[c_idx][j_pts_idx]),
                i_pts_depth[c_idx][j_pts_idx],
            )
            ranks = coor[:, 0] + coor[:, 1] * Ws
            sort = (ranks + depth / 100.0).argsort()
            coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
            kept = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
            kept[1:] = ranks[1:] != ranks[:-1]
            coor, depth = coor[kept], depth[kept]
            coor = coor.to(torch.long)
            depth_map[c_idx, coor[:, 1], coor[:, 0]] = depth
        depth_gt.append(depth_map)
        depth_mask.append(depth_map < 0)
    depth_gt = torch.stack(depth_gt, dim=0)
    depth_mask = torch.stack(depth_mask, dim=0)

    return depth_gt, depth_mask

def get_downsampled_gt_depth(depth_gt, downsample, depth_tg=None):
    B, NC, H, W = depth_gt.shape
    depth_gt = depth_gt.view(
        B * NC, H // downsample, downsample, W // downsample, downsample
    )
    depth_gt = depth_gt.permute(0, 1, 3, 2, 4).flatten(-2, -1)

    if depth_tg is None:
        # Minimum selection (original behavior)
        depth_gt_tmp = torch.where(
            depth_gt < 0.0, 1e5 * torch.ones_like(depth_gt), depth_gt
        )
        depth_gt_ds = torch.min(depth_gt_tmp, dim=-1).values
        depth_gt_ds = depth_gt_ds.view(B, NC, H // downsample, W // downsample)
        return depth_gt_ds
    else:
        # Random selection from valid values
        valid_mask = depth_gt > 0.0
        depth_gt_tmp = torch.rand_like(depth_gt)
        depth_gt_tmp[~valid_mask] = 1e5
        # Get the index of minimum random value (random selection from valid values)
        sampled_idx = torch.min(depth_gt_tmp, dim=-1).indices
        # Gather actual depth values using the random indices
        depth_gt_ds = torch.gather(depth_gt, dim=-1, index=sampled_idx.unsqueeze(-1))
        depth_gt_ds = depth_gt_ds.view(B, NC, H // downsample, W // downsample)

        # Downsample depth_tg using the same indices
        depth_tg = depth_tg.view(
            B * NC, H // downsample, downsample, W // downsample, downsample
        )
        depth_tg = depth_tg.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        depth_tg_ds = torch.gather(depth_tg, dim=-1, index=sampled_idx.unsqueeze(-1))
        depth_tg_ds = depth_tg_ds.view(B, NC, H // downsample, W // downsample)
        return depth_gt_ds, depth_tg_ds[:, None]


@OPENOCC_LOSS.register_module()
class RGBLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        input_dict=None,
        normalized=True
    ):
        
        super().__init__(weight)

        self.mean = torch.tensor(mean).view(1, 1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, 1, -1, 1, 1)
        if input_dict is None:
            self.input_dict = {
                'rendered_rgbs': 'rendered_rgbs',
                'imgs': 'imgs',
            }
        else:
            self.input_dict = input_dict
        self.normalized = normalized
        self.loss_func = self.loss_rgb

    def loss_rgb(self, rendered_rgbs, imgs):
        if self.normalized:
            mean = self.mean.to(imgs.device)
            std = self.std.to(imgs.device)
            imgs = (imgs * std + mean) / 255.
        downsample = imgs.shape[-1] // rendered_rgbs.shape[-1]
        if downsample > 1:
            bs, num_cams, _, H, W = imgs.shape
            imgs = F.interpolate(
                imgs.flatten(0, 1),
                scale_factor=1. / downsample,
                mode='bilinear',
                align_corners=False
            ).view(bs, num_cams, 3, H // downsample, W // downsample)
        loss = (rendered_rgbs - imgs[:, None]).abs().mean()
        return loss


@OPENOCC_LOSS.register_module()
class DepthLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        depth_range=[1.0, 72.0],
        num_samples=128,
        input_dict=None
    ):
        
        super().__init__(weight)

        self.depth_range = depth_range
        self.max_depth = depth_range[1]
        self.depth_res = (depth_range[1] - depth_range[0]) / num_samples
        self.depth_channels = num_samples
        if input_dict is None:
            self.input_dict = {
                'depth_preds': 'depth_preds',
                'points': 'points',
                'metas': 'metas',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_depth

    def get_binary_gt_depth(self, depth_gt):
        depth_gt_bin = ((depth_gt - self.depth_range[0]) / self.depth_res).long()
        depth_gt_bin = torch.where(
            (depth_gt_bin < self.depth_channels) & (depth_gt_bin >= 0),
            depth_gt_bin + 1,
            torch.zeros_like(depth_gt_bin),
        )
        depth_gt_bin = (
            F.one_hot(depth_gt_bin, num_classes=self.depth_channels + 1)
            .view(-1, self.depth_channels + 1)[..., 1:]
            .float()
        )
        return depth_gt_bin

    def loss_depth(self, depth_preds, points, metas):
        target_gt, _ = assign_depth_target(points, metas)
        target_gt = target_gt.clamp(max=self.max_depth)
        downsample = target_gt.shape[-1] // depth_preds.shape[-2]
        ds_gt = get_downsampled_gt_depth(target_gt, downsample)
        bin_gt = self.get_binary_gt_depth(ds_gt)
        depth_preds = depth_preds.clamp(max=self.max_depth).view(-1, bin_gt.size(-1))
        fg_mask = torch.max(bin_gt, dim=1).values > 0.0
        depth_loss = F.binary_cross_entropy(
            depth_preds[fg_mask].clamp(0.0, 1.0),
            bin_gt[fg_mask],
            reduction="none",
        ).sum() / max(1.0, fg_mask.sum())
        return depth_loss
    

@OPENOCC_LOSS.register_module()
class RenderedDepthLoss(DepthLoss):

    def __init__(
        self,
        weight=1.0,
        depth_range=[1.0, 72.0],
        patch_size=1,
        input_dict=None
    ):
        
        super().__init__(weight)

        self.depth_range = depth_range
        self.max_depth = depth_range[1]
        self.patch_size = int(patch_size)
        if input_dict is None:
            self.input_dict = {
                'rendered_depths': 'rendered_depths',
                'points': 'points',
                'metas': 'metas',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_depth

    def loss_depth(self, rendered_depths, points, metas):
        # LiDAR Supervision
        target_gt, target_mask = assign_depth_target(points, metas)

        # # Depth Anything 3 Supervision
        # target_gt = metas['da3_depths'].to(torch.float32)
        # target_gt = F.interpolate(
        #     target_gt,
        #     size=rendered_depths.shape[-2:],
        #     mode='bilinear',
        #     align_corners=False
        # )
        # target_mask = (target_gt < 0.0) # & (target_gt > 120.0)

        downsample = target_gt.shape[-1] // rendered_depths.shape[-1]
        if downsample > 1:
            target_gt = get_downsampled_gt_depth(target_gt, downsample)
            target_mask = target_gt > 1e4
        if (self.patch_size // downsample) > 1:
            patch_size = self.patch_size // downsample
            target_gt, rendered_depths = get_downsampled_gt_depth(
                target_gt, patch_size, depth_tg=rendered_depths
            )
            target_mask = target_gt < 0.0

        target_gt = target_gt.clamp(max=self.max_depth)
        rendered_depths = rendered_depths.clamp(max=self.max_depth)
        loss = (rendered_depths[:, -1] - target_gt)[~target_mask]
        loss = loss.abs().mean() / self.max_depth
        return loss


@OPENOCC_LOSS.register_module()
class DistillLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):

        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'rendered_feats': 'rendered_feats',
                'dino_feats': 'dino_feats',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_distill

    def loss_distill(self, rendered_feats, dino_feats):
        downsample = dino_feats.shape[-1] // rendered_feats.shape[-1]
        if downsample > 1:
            bs, num_cams, _, H, W = dino_feats.shape
            dino_feats = F.interpolate(
                dino_feats.flatten(0, 1),
                scale_factor=1. / downsample,
                mode='bilinear',
                align_corners=False
            ).view(bs, num_cams, -1, H // downsample, W // downsample)
        loss = (rendered_feats - dino_feats[:, None]).abs().mean()
        return loss
    

@OPENOCC_LOSS.register_module()
class MaskLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):

        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'is_empty': 'is_empty',
                'mask': 'mask',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.mask_loss

    def mask_loss(self, is_empty, mask):
        mask_ds = F.interpolate(
            mask.to(torch.float),
            size=is_empty.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        mask = torch.zeros_like(mask_ds, dtype=torch.bool)
        mask[mask_ds > 0.5] = True
        loss = (-torch.log(is_empty + 1e-8)[mask]).mean()
        return loss
