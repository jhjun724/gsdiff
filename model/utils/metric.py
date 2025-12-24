import os
import numpy as np
import torch
import torch.nn.functional as F

import imageio
from matplotlib import pyplot as plt
from mayavi import mlab
from skimage.metrics import structural_similarity as ssim

mlab.options.offscreen = True


def cal_depth_metric(pred_depth, gt_depth, mask=None, is_prop=False):
    if mask is None:
        mask = torch.ones_like(pred_depth, dtype=torch.bool)
    if is_prop:
        pred_depth, pred_prop_depth = pred_depth
    
    mask = mask & (gt_depth < 80.0) & (gt_depth > 1.0)
    abs_rel = abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]
    sq_rel = ((pred_depth[mask] - gt_depth[mask]) ** 2) / gt_depth[mask]
    rmse = ((pred_depth[mask] - gt_depth[mask]) ** 2).mean() ** 0.5
    rmse_log = ((torch.log(pred_depth[mask] / gt_depth[mask])) ** 2).mean() ** 0.5
    
    delta = torch.maximum(
        pred_depth[mask] / gt_depth[mask],
        gt_depth[mask] / pred_depth[mask],
    )
    delta1 = (delta < 1.25     ).float().mean()
    delta2 = (delta < 1.25 ** 2).float().mean()
    delta3 = (delta < 1.25 ** 3).float().mean()
    
    depth_loss = dict(
        abs_rel=float(abs_rel.mean()),
        sq_rel=float(sq_rel.mean()),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        delta1=float(delta1),
        delta2=float(delta2),
        delta3=float(delta3),
    )
        
    if is_prop:
        prop_abs_rel = abs(pred_prop_depth[mask] - gt_depth[mask]) / gt_depth[mask]
        prop_sq_rel = ((pred_prop_depth[mask] - gt_depth[mask]) ** 2) / gt_depth[mask]
        prop_rmse = ((pred_prop_depth[mask] - gt_depth[mask]) ** 2).mean() ** 0.5
        prop_rmse_log = ((torch.log(pred_prop_depth[mask] / gt_depth[mask])) ** 2).mean() ** 0.5
        
        prop_delta = torch.maximum(
            pred_prop_depth[mask] / gt_depth[mask],
            gt_depth[mask] / pred_prop_depth[mask],
        )
        prop_delta1 = (prop_delta < 1.25     ).float().mean()
        prop_delta2 = (prop_delta < 1.25 ** 2).float().mean()
        prop_delta3 = (prop_delta < 1.25 ** 3).float().mean()
        prop_depth_loss = dict(
            abs_rel=(prop_abs_rel.mean()),
            sq_rel=(prop_sq_rel.mean()),
            rmse=(prop_rmse),
            rmse_log=(prop_rmse_log),
            delta1=(prop_delta1),
            delta2=(prop_delta2),
            delta3=(prop_delta3),
        )
        return depth_loss, prop_depth_loss
    
    return depth_loss

def cal_photo_metric(pred_rgb, gt_rgb, lpips_fn):
    num_cam = len(pred_rgb)
    
    psnr_val = [
        -10 * torch.log10(F.mse_loss(pred_rgb[i], gt_rgb[i])).item()
        for i in range(num_cam)
    ]
    ssim_val = [
        ssim(
            pred_rgb[i].cpu().numpy(),
            gt_rgb[i].cpu().numpy(),
            data_range=1.0,
            channel_axis=-1
        )
        for i in range(num_cam)
    ]
    lpips_val = [
        lpips_fn(
            pred_rgb[i:i+1].permute(0, 3, 1, 2) * 2 - 1,
            gt_rgb[i:i+1].permute(0, 3, 1, 2) * 2 - 1,
        ).mean().item()
        for i in range(num_cam)
    ]
    
    rgb_loss = dict(
        psnr=sum(psnr_val) / len(psnr_val),
        ssim=sum(ssim_val) / len(ssim_val),
        lpips=sum(lpips_val) / len(lpips_val),
    )
    
    return rgb_loss
    
def vis_nerf(img_metas, pred_rgb, pred_depth, idx=0, vis_dir="vis/tmp", num_cam=6):
    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())

    pred_rgb = pred_rgb.cpu().numpy()
    pred_depth = pred_depth.cpu().numpy()

    if num_cam == 6:
        merged_rgb = np.concatenate([
            np.concatenate([pred_rgb[2], pred_rgb[0], pred_rgb[1]], axis=1),
            np.concatenate([pred_rgb[4, :, ::-1], pred_rgb[3, :, ::-1], pred_rgb[5, :, ::-1]], axis=1)
        ], axis=0)
        merged_depth = np.concatenate([
            np.concatenate([pred_depth[2], pred_depth[0], pred_depth[1]], axis=1),
            np.concatenate([pred_depth[4, :, ::-1], pred_depth[3, :, ::-1], pred_depth[5, :, ::-1]], axis=1)
        ], axis=0)
    elif num_cam == 7:
        empty = np.ones_like(pred_rgb[0, :, ::2])
        merged_rgb = np.concatenate([
            np.concatenate([empty, pred_rgb[0], empty], axis=1),
            np.concatenate([pred_rgb[1], pred_rgb[2]], axis=1),
            np.concatenate([pred_rgb[3], pred_rgb[4]], axis=1),
            np.concatenate([pred_rgb[5], pred_rgb[6]], axis=1),
        ], axis=0)
        merged_depth = np.concatenate([
            np.concatenate([empty[..., 0], pred_depth[0], empty[..., 0]], axis=1),
            np.concatenate([pred_depth[1], pred_depth[2]], axis=1),
            np.concatenate([pred_depth[3], pred_depth[4]], axis=1),
            np.concatenate([pred_depth[5], pred_depth[6]], axis=1),
        ], axis=0)
    
    assert (merged_depth < 0.0).sum() == 0
    merged_depth[merged_depth > 1.0] = 1.0
    merged_depth = plt.cm.magma(1 - merged_depth)[..., :3]
    
    os.makedirs(f"{vis_dir}/rgb", exist_ok=True)
    os.makedirs(f"{vis_dir}/depth", exist_ok=True)

    imageio.imwrite(
        "{}/rgb/{:04}_{}_rgb.png".format(
        vis_dir, idx, img_metas[0]["sample_idx"]
        ), (merged_rgb * 255).astype('uint8')
    )
    imageio.imwrite(
        "{}/depth/{:04}_{}_depth.png".format(
        vis_dir, idx, img_metas[0]["sample_idx"]
        ), (merged_depth * 255).astype('uint8')
    )
    
def vis_occ(batch_rays, results, img_metas, img_num=0, vis_dir="vis/tmp"):
    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    # point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
    voxel_size = point_cloud_range[3] / 100.0
    occupancy_size = [voxel_size] * 3
    
    occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
    occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
    occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
    
    device = batch_rays[0]["ray_o"].device
    
    occupancy = np.zeros((len(batch_rays), 200, 200, 16), dtype=np.uint8)
    colors_map = np.array(
        [[255, 255, 255, 255], # free
            [222, 184, 135, 255]]  # manmade
    ).astype(np.uint8)
    
    offset = -torch.tensor(point_cloud_range[:3])
    # lidar2ego_trans = torch.tensor(
    #     [[ 0.0020344,  0.9997041,  0.0242383, 0.943713],
    #      [-0.9999805,  0.0021768, -0.0058515, 0.0     ],
    #      [-0.0059025, -0.0242259,  0.9996891, 1.840230],
    #      [ 0.0      ,  0.0      ,  0.0      , 1.0     ]]
    # )
    lidar2ego_trans = torch.eye(4)
    
    gpus = torch.cuda.device_count()
    idx = img_num * gpus + batch_rays[0]["ray_o"].device.index
    
    origins = batch_rays[0]["ray_o"].detach().cpu()
    directions = batch_rays[0]["ray_d"].detach().cpu()
    # origins = results[0]["origins"]
    # directions = results[0]["directions"]
    depth = results[0]["depth"]#.to(origins.device)
    pos = origins + directions * depth
    """ego coordinates"""
    pos = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    pos_ego = torch.matmul(lidar2ego_trans.unsqueeze(0), pos.unsqueeze(-1))
    pos_ego = (pos_ego.squeeze(-1)[:, :3] + offset) / voxel_size
    voxel_coords = pos_ego.long().detach().cpu().numpy()
    selector = (
        (voxel_coords[..., 0] > 0)
        & (voxel_coords[..., 0] < occ_xdim)
        & (voxel_coords[..., 1] > 0)
        & (voxel_coords[..., 1] < occ_ydim)
        & (voxel_coords[..., 2] > 0)
        & (voxel_coords[..., 2] < occ_zdim)
    )
    voxel_coords_x = voxel_coords[..., 0][selector]
    voxel_coords_y = voxel_coords[..., 1][selector]
    voxel_coords_z = voxel_coords[..., 2][selector]
    occupancy[0, voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1
    
    points_x = (voxel_coords_x.astype(np.float) - occ_xdim / 2) * occupancy_size[0]
    points_y = (voxel_coords_y.astype(np.float) - occ_ydim / 2) * occupancy_size[1]
    points_z = (voxel_coords_z.astype(np.float) - occ_zdim / 2) * occupancy_size[2]
    """lidar coordinates"""
    # point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    # selector = (
    #     (pos[:, 0] > point_cloud_range[0])
    #     & (pos[:, 0] < point_cloud_range[3])
    #     & (pos[:, 1] > point_cloud_range[1])
    #     & (pos[:, 1] < point_cloud_range[4])
    #     & (pos[:, 2] > point_cloud_range[2])
    #     & (pos[:, 2] < point_cloud_range[5])
    # )
    
    # points_x = pos[:, 0][selector].detach().cpu().numpy()
    # points_y = pos[:, 1][selector].detach().cpu().numpy()
    # points_z = pos[:, 2][selector].detach().cpu().numpy()
    """"""
    # point_colors = np.ones(points_x.shape)
    norm_z = (points_z - point_cloud_range[2]) / (point_cloud_range[5] - point_cloud_range[2])
    
    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    lidar_plot = mlab.points3d(points_x, points_y, points_z, norm_z,
                                scale_factor=voxel_size, scale_mode="vector",
                                mode="cube", opacity=1.0, colormap="jet")
    #                          mode="cube", opacity=1.0, vmin=1, vmax=2,)
    # lidar_plot.module_manager.scalar_lut_manager.lut.table = colors_map
    mlab.view(azimuth=45, elevation=45, distance=180, focalpoint=[0,0,0])
    mlab.savefig("{}/occ/{:04}_{}_occ.png".format(vis_dir, idx, img_metas[0]["sample_idx"]))
    mlab.close(figure)
    
    np.savez_compressed("{}/occ_numpy/{:04}_{}_occ.npz".format(vis_dir, idx, img_metas[0]["sample_idx"]), occupancy)

def vis_sdf(sdf, img_metas, pc_range, voxel_size, img_num=0, vis_dir="vis/tmp"):
    colors_map = np.array(
        [[255, 255, 255, 255], # free
            [222, 184, 135, 255]]  # manmade
    ).astype(np.uint8)
    
    # pc_range = self.pts_bbox_head.pc_range
    # voxel_size = self.pts_bbox_head.unified_voxel_size
    num_voxel = [int((pc_range[i+3] - pc_range[i]) / voxel_size[i]) for i in range(len(voxel_size))]
    tx = torch.linspace(pc_range[0], pc_range[3]-voxel_size[0], num_voxel[0]) + voxel_size[0] / 2
    ty = torch.linspace(pc_range[1], pc_range[4]-voxel_size[1], num_voxel[1]) + voxel_size[1] / 2
    tz = torch.linspace(pc_range[2], pc_range[5]-voxel_size[2], num_voxel[2]) + voxel_size[2] / 2
    points_x, points_y, points_z = torch.meshgrid(tx, ty, tz)
    
    selector = (sdf.detach().clone().cpu().numpy() < 0.0).flatten()
    points_x = points_x.flatten()[selector].numpy()
    points_y = points_y.flatten()[selector].numpy()
    points_z = points_z.flatten()[selector].numpy()
    point_colors = np.ones(points_x.shape)
    
    gpus = torch.cuda.device_count()
    idx = img_num * gpus + sdf.device.index
    
    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    lidar_plot = mlab.points3d(points_x, points_y, points_z, point_colors,
                                scale_factor=voxel_size[0], scale_mode="vector",
                                mode="cube", opacity=1.0, colormap="jet")
    lidar_plot.module_manager.scalar_lut_manager.lut.table = colors_map
    mlab.view(azimuth=45, elevation=45, distance=180, focalpoint=[0,0,0])
    mlab.savefig("{}/sdf/{:04}_{}_sdf.png".format(vis_dir, idx, img_metas[0]["sample_idx"]))
    mlab.close(figure)
