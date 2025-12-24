import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer
# from mmengine.runner import force_fp32, auto_fp16
from mmdet.models.backbones.resnet import BasicBlock
from mmengine.registry import MODELS


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

@MODELS.register_module()
class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        depth_channels,
        depth_range=[1.0, 45.0],
        depth_res=0.5,
        loss_weight=1.0,
        use_dcn=True,
        use_aspp=True,
        aspp_mid_channels=-1,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        depth_conv_input_channels = mid_channels
        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type="DCN",
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )
                )
            )
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        )
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.depth_channels = depth_channels

        self.depth_range = depth_range
        self.depth_res = depth_res
        self.loss_weight = loss_weight

    def assign_depth_target(self, pts, imgs, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])

        lidar2img = pts[0].new_tensor(np.asarray(lidar2img))
        lidar2img = lidar2img.flatten(1, 2)

        depth_gt = []
        for bs_idx in range(len(pts)):
            i_pts = pts[bs_idx]
            i_pts = i_pts[i_pts[:, :2].norm(dim=-1) > 3.0]
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_imgs = imgs[bs_idx]

            i_lidar2img = lidar2img[bs_idx]
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
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
                & (i_pts_depth > self.depth_range[0])
                & (i_pts_depth < self.depth_range[1])
            )

            depth_map = i_imgs.new_zeros((i_imgs.shape[0], *i_imgs.shape[2:]))
            for c_idx in range(len(i_pts_mask)):
                # (M,) -> (Q,)
                j_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                coor, depth = (
                    torch.round(i_pts_cam[c_idx][j_pts_idx]),
                    i_pts_depth[c_idx][j_pts_idx],
                )
                ranks = coor[:, 0] + coor[:, 1] * i_imgs.shape[-1]
                sort = (ranks + depth / 100.0).argsort()
                coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
                kept = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
                kept[1:] = ranks[1:] != ranks[:-1]
                coor, depth = coor[kept], depth[kept]
                coor = coor.to(torch.long)
                depth_map[c_idx, coor[:, 1], coor[:, 0]] = depth
            depth_gt.append(depth_map)
        depth_gt = torch.stack(depth_gt, dim=0)

        return depth_gt

    def get_downsampled_gt_depth(self, depth_gt, downsample):
        B, NC, H, W = depth_gt.shape
        depth_gt = depth_gt.view(
            B * NC, H // downsample, downsample, W // downsample, downsample
        )
        depth_gt = depth_gt.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        depth_gt_tmp = torch.where(
            depth_gt == 0.0, 1e5 * torch.ones_like(depth_gt), depth_gt
        )
        depth_gt = torch.min(depth_gt_tmp, dim=-1).values
        depth_gt = ((depth_gt - self.depth_range[0]) / self.depth_res).long()
        depth_gt = torch.where(
            (depth_gt < self.depth_channels) & (depth_gt >= 0),
            depth_gt + 1,
            torch.zeros_like(depth_gt),
        )
        depth_gt = (
            F.one_hot(depth_gt, num_classes=self.depth_channels + 1)
            .view(-1, self.depth_channels + 1)[..., 1:]
            .float()
        )
        return depth_gt

    # @force_fp32(apply_to=("depth_preds", "pts", "imgs"))
    def loss(self, depth_preds, pts, imgs, img_metas):
        depth_gt = self.assign_depth_target(pts, imgs, img_metas)
        loss_dict = {}
        for i, d_pred in enumerate(depth_preds):
            # if not self.loss_cfg.depth_loss_weights[i] > 0.0:
            #     continue
            downsample = depth_gt.shape[-1] // d_pred.shape[-1]
            d_gt = self.get_downsampled_gt_depth(depth_gt, downsample)
            d_pred = d_pred.permute(0, 2, 3, 1).contiguous().view(-1, d_gt.shape[-1])
            assert d_gt.shape[0] == d_pred.shape[0]
            fg_mask = torch.max(d_gt, dim=1).values > 0.0
            depth_loss = F.binary_cross_entropy(
                d_pred[fg_mask].clamp(0.0, 1.0),
                d_gt[fg_mask],
                reduction="none",
            ).sum() / max(1.0, fg_mask.sum())
            loss_dict[f"loss_depth_{i}"] = depth_loss * self.loss_weight
        return loss_dict

    def forward(self, x):
        x = self.reduce_conv(x)
        depth = self.depth_conv(x)
        return depth