import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone

from .base_segmentor import CustomBaseSegmentor
from ..utils import renderer
from ..encoder.gaussian_encoder.utils import linear_relu_ln

import os, math

@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        render=False,
        mvs=False,
        updown=True,
        downsample=1,
        # use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.out_render = render
        self.out_mvs = mvs
        self.downsample = downsample
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

        self.embed_dims = self.img_neck.out_channels
        if not hasattr(self.lifter, 'pretraining'):
            self.lifter.pretraining = False
        if self.lifter.pretraining and updown:
            self.linear_down = nn.Sequential(*linear_relu_ln(32, 2, 2, input_dims=self.embed_dims))
            self.linear_up = nn.Sequential(*linear_relu_ln(384, 2, 2, input_dims=32))

        num_decoder = self.encoder.operation_order.count("refine")
        if (not self.training) or (not hasattr(self, "head")):
            self.apply_loss_layers = [num_decoder - 1]
        elif self.head.apply_loss_type == "all":
            self.apply_loss_layers = list(range(num_decoder))
        elif self.head.apply_loss_type == "random":
            if self.head.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(
                    num_decoder - 1, self.head.random_apply_loss_layers - 1, False
                )
                self.apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                self.apply_loss_layers = [num_decoder - 1]
        elif self.head.apply_loss_type == 'fixed':
            self.apply_loss_layers = self.head.fixed_apply_loss_layers
        else:
            raise NotImplementedError
        if hasattr(self, "head"):
            self.head.apply_loss_layers = self.apply_loss_layers

    def load_dinov3(self, model_name='vits16'):
        weights = {
            'vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            'vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
            'vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
            'vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
            'vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
            'vit7b': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
            'convnext_tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
            'convnext_small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
            'convnext_base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
            'convnext_large': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
        }
        assert model_name in weights, f"model_name {model_name} not found"
        if 'convnext' in model_name:
            raise NotImplementedError("Error for init_weights()")
        else:
            self.dino = torch.hub.load(
                'dinov3',
                'dinov3_' + model_name,
                source='local',
                weights='dinov3/ckpts/' + weights[model_name]
            )
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad_(False)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result

    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def render(self, imgs, cam_params, representation, f: int=1):
        gaussians = []
        for idx in self.apply_loss_layers:
            gaussians.append(representation[idx]['gaussian'])
        linear = self.linear_down if hasattr(self, 'linear_down') else None
        return renderer.render(imgs, cam_params, gaussians, linear=linear, f=f)

    def get_dino_features(self, imgs, rendered_feats, **kwargs):
        batch_size, num_cams, _, H, W = imgs.shape
        dH, dW = H // 16, W // 16
        dino_feats = self.dino.forward_features(imgs.flatten(0, 1))
        dino_feats = dino_feats['x_norm_patchtokens'].permute(0, 2, 1)
        dino_feats = dino_feats.view(batch_size, num_cams, -1, dH, dW)

        rendered_feats = F.interpolate(rendered_feats.flatten(0, 2), size=(dH, dW), mode='bilinear')
        if hasattr(self, 'linear_up'):
            rendered_feats = self.linear_up(rendered_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rendered_feats = rendered_feats.view(batch_size, -1, num_cams, dino_feats.shape[2], dH, dW)

        return dino_feats, rendered_feats

    def get_depth_anything(self, metas):
        return metas['da3_depths']

    def forward(self,
                imgs=None,
                points=None,
                metas=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        ts = time.time()
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'mask': metas.get('semantics', None),
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)
        image_end = time.time()

        outs = self.lifter(**results)
        results.update(outs)
        lifter_end = time.time()

        outs = self.encoder(**results)
        encoder_end = time.time()
        if rep_only:
            return outs['representation']
        results.update(outs)
        if self.out_render:
            rendered_rgbs, _, rendered_depths, rendered_feats = self.render(
                results['imgs'],
                results['metas']['cam_params'],
                results['representation'],
                f=self.downsample,
            )
            results.update({
                'rendered_rgbs': rendered_rgbs,
                'rendered_depths': rendered_depths,
                'rendered_feats': rendered_feats
            })
        render_end = time.time()
        if self.out_mvs:
            rendered_temp_rgbs, _, rendered_temp_depths, _ = self.render(
                results['imgs'],
                results['metas']['temCam_params'],
                results['representation'],
                f=self.downsample,
            )
            results.update({
                'curr_imgs': metas['curr_imgs'],
                'prev_imgs': metas['prev_imgs'],
                'next_imgs': metas['next_imgs'],
                'rendered_temp_rgbs': rendered_temp_rgbs,
                'rendered_temp_depths': rendered_temp_depths[:, 0],
                # 'rendered_temp_depths': self.get_depth_anything(metas),
            })
            # if ('prevCam_params' in metas) and ('nextCam_params' in metas):
            #     rendered_prev_rgbs, _, rendered_prev_depths, _ = self.render(
            #         results['imgs'],
            #         results['metas']['prevCam_params'],
            #         results['representation'],
            #         f=8,
            #     )
            #     rendered_next_rgbs, _, rendered_next_depths, _ = self.render(
            #         results['imgs'],
            #         results['metas']['nextCam_params'],
            #         results['representation'],
            #         f=8,
            #     )
            #     results.update({
            #         'rendered_prev_rgbs': rendered_prev_rgbs,
            #         'rendered_prev_depths': rendered_prev_depths[:, 0],
            #         'rendered_next_rgbs': rendered_next_rgbs,
            #         'rendered_next_depths': rendered_next_depths[:, 0],
            #     })
        mvs_end = time.time()
        if hasattr(self, 'dino') and self.out_render:
            ms_dino_feats, rendered_feats = self.get_dino_features(imgs, rendered_feats)
            results['dino_feats'] = ms_dino_feats
            results['rendered_feats'] = rendered_feats
        dino_end = time.time()
        if hasattr(self, 'head'):
            if occ_only and hasattr(self.head, 'forward_occ'):
                outs = self.head.forward_occ(**results)
            else:
                outs = self.head(**results)
            results.update(outs)
        head_end = time.time()
        if 'gaussian' not in results and not self.training: # for visualization of pre-training
            results['gaussians'] = [r['gaussian'] for r in results['representation']]
            gaussian = results['representation'][-1]['gaussian']
            semantics = self.get_linear_label(gaussian.features)
            results['gaussian'] = self.update_gaussian_sem(gaussian, semantics)
            results['final_occ'] = torch.zeros((imgs.shape[0], 200*200*16), device=imgs.device)

        results.update({
            'image_time': image_end - ts,
            'lifter_time': lifter_end - image_end,
            'encoder_time': encoder_end - lifter_end,
            'render_time': render_end - encoder_end,
            'mvs_time': mvs_end - render_end,
            'dino_time': dino_end - mvs_end,
            'head_time': head_end - dino_end,
        })


        return results

    def get_linear_label(self, features):
        from model.utils.renderer import get_robust_pca
        batch_size = features.shape[0]
        out_label = []
        for bi in range(batch_size):
            out_label.append(get_robust_pca(features[bi], out_dim=17))
        out_label = torch.stack(out_label, dim=0)
        return out_label#.softmax(dim=-1)

    def update_gaussian_sem(self, gaussian, semantics):
        from model.encoder.gaussian_encoder.utils import GaussianPrediction
        updated_gaussian = GaussianPrediction(
            means=gaussian.means,
            scales=gaussian.scales,
            rotations=gaussian.rotations,
            opacities=gaussian.opacities,
            semantics=semantics,
            original_means=gaussian.original_means,
            delta_means=gaussian.delta_means,
            colors=gaussian.colors,
            features=gaussian.features
        )
        return updated_gaussian
