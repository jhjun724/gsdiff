import os

import imageio
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(imgs, cam_params, gaussians, num_groups=1, linear=None, f: int=1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    batch_size, num_cams = imgs.shape[:2]
    
    b_rendered_rgbs = []
    b_rendered_radiis = []
    b_rendered_depths = []
    b_rendered_features = []
    for bi in range(batch_size):
        g_rendered_rgbs = []
        g_rendered_radiis = []
        g_rendered_depths = []
        g_rendered_features = []
        for gaussian in gaussians:
            pts_xyz = gaussian.means[bi]
            if hasattr(gaussian, 'colors') and (gaussian.colors is not None):
                assert gaussian.colors.shape[-1] == 3
                pts_rgb = gaussian.colors[bi]
            else:
                pts_rgb = get_robust_pca(gaussian.features[bi])
            opacity = gaussian.opacities[bi]
            scales = gaussian.scales[bi]
            rotations = gaussian.rotations[bi]
            if linear is not None:
                features = linear(gaussian.features[bi])
            else:
                features = gaussian.features[bi]

            device = pts_xyz.device
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(
                pts_xyz, dtype=torch.float32, requires_grad=True, device=device
            ) + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            bg_color=[0, 0, 0]
            bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

            c_rendered_rgbs = []
            c_rendered_radiis = []
            c_rendered_depths = []
            c_rendered_features = []
            for cam_idx in range(num_cams):
                height = cam_params[bi][cam_idx]['height'] // f
                width = cam_params[bi][cam_idx]['width'] // f
                fovx = cam_params[bi][cam_idx]['fovx']
                fovy = cam_params[bi][cam_idx]['fovy']
                viewmatrix = cam_params[bi][cam_idx]['viewmatrix'].to(device)
                projmatrix = cam_params[bi][cam_idx]['projmatrix'].to(device)
                cam_pos = cam_params[bi][cam_idx]['cam_pos'].to(device)
                depth = (
                    torch.cat([pts_xyz, torch.ones_like(pts_xyz[:, :1])], dim=1)
                    @ viewmatrix[:, 2:3]
                )

                # Set up rasterization configuration
                tanfovx = math.tan(fovx * 0.5)
                tanfovy = math.tan(fovy * 0.5)
                raster_settings = GaussianRasterizationSettings(
                    image_height=height,
                    image_width=width,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg_color,
                    scale_modifier=1.0,
                    viewmatrix=viewmatrix,
                    projmatrix=projmatrix,
                    sh_degree=3,
                    campos=cam_pos,
                    prefiltered=False,
                    render_features=True,
                    render_gaussian_idx=False,
                    debug=False
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                if num_groups == 1:
                    ras_rgb, ras_feature, ras_depth, ras_gauss_idx, ras_radii = rasterizer(
                        means3D=pts_xyz,
                        means2D=screenspace_points,
                        shs=None,
                        colors_precomp=pts_rgb,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                        distill_feats=features
                    )
                    # ras_depth = ras_depth.unsqueeze(0)
                
                if gaussian.colors is None:
                    pca_feature = get_robust_pca(
                        ras_feature.permute(1, 2, 0).flatten(0, 1)
                    ).view(height, width, 3)
                    ras_rgb = pca_feature.permute(2, 0, 1)

                c_rendered_rgbs.append(ras_rgb)
                c_rendered_radiis.append(ras_radii)
                c_rendered_depths.append(ras_depth)
                c_rendered_features.append(ras_feature)
            g_rendered_rgbs.append(torch.stack(c_rendered_rgbs, dim=0))
            g_rendered_radiis.append(torch.stack(c_rendered_radiis, dim=0))
            g_rendered_depths.append(torch.stack(c_rendered_depths, dim=0))
            g_rendered_features.append(torch.stack(c_rendered_features, dim=0))
        b_rendered_rgbs.append(torch.stack(g_rendered_rgbs, dim=0))
        b_rendered_radiis.append(torch.stack(g_rendered_radiis, dim=0))
        b_rendered_depths.append(torch.stack(g_rendered_depths, dim=0))
        b_rendered_features.append(torch.stack(g_rendered_features, dim=0))

    rendered_rgbs = torch.stack(b_rendered_rgbs, dim=0)
    rendered_radiis = torch.stack(b_rendered_radiis, dim=0)
    rendered_depths = torch.stack(b_rendered_depths, dim=0)
    rendered_features = torch.stack(b_rendered_features, dim=0)

    return rendered_rgbs, rendered_radiis, rendered_depths, rendered_features


def get_robust_pca(features: torch.Tensor, out_dim: int = 3, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=out_dim, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=out_dim, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    if out_dim == 3:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]

        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    else:
        rgb_min = torch.tensor([colors[fg_mask][s[:, i] < m, i].min() for i in range(out_dim)])
        rgb_max = torch.tensor([colors[fg_mask][s[:, i] < m, i].max() for i in range(out_dim)])

    rgb_min = rgb_min.to(reduction_mat)
    rgb_max = rgb_max.to(reduction_mat)

    reduced_rgb = features @ reduction_mat
    reduced_rgb = (reduced_rgb - rgb_min) / (rgb_max - rgb_min)

    return reduced_rgb
