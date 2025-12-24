import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
from tools.utils.visualize import get_denormalized_img


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


@OPENOCC_LOSS.register_module()
class TemporalViewReprojLoss(BaseLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_dict = {
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ts': 'ts',
                'metas': 'metas',
            }
        else:
            self.input_dict = input_dict
        self.no_ssim = kwargs.get('no_ssim', False)
        self.no_automask = kwargs.get('no_automask', False)
        self.ratio = kwargs.get('ratio', 0.7)
        self.dims = kwargs.get('dims', 3)
        self.img_size = kwargs.get('img_size', [896, 1600])
        if not self.no_ssim:
            self.ssim = SSIM()
        
        self.loss_func = self.reproj_loss

    @torch.cuda.amp.autocast(enabled=False)
    def cal_pixel(self, trans, coords):
        trans = trans.float()
        coords = coords.float()
        eps = 1e-5
        pixel = torch.matmul(trans, coords).squeeze(-1) # bs, N, R, 4
        mask = pixel[..., 2] > eps
        pixel_uv = pixel[..., :2] / torch.maximum(
            torch.ones_like(pixel[..., :1]) * eps, pixel[..., 2:3]
        )
        mask = (
            mask &
            (pixel_uv[..., 0] > 0) &
            (pixel_uv[..., 0] < self.img_size[1] - 1) &
            (pixel_uv[..., 1] > 0) &
            (pixel_uv[..., 1] < self.img_size[0] - 1)
        )
        return pixel_uv, mask
            
    def sample_pixel(self, pixel, imgs):
        # imgs: B, N, 3, H, W
        # pixel: B, N, 1, R, 2
        bs, num_cams = imgs.shape[:2]
        pixel = pixel.clone()
        pixel[..., 0] /= (self.img_size[1] - 1) # Map [0, W-1] to [0, 1]
        pixel[..., 1] /= (self.img_size[0] - 1) # Map [0, H-1] to [0, 1]
        pixel = 2 * pixel - 1
        pixel_rgb = F.grid_sample(
            imgs.flatten(0, 1), 
            pixel.flatten(0, 1), 
            mode='bilinear',
            padding_mode='border',
            align_corners=True) # BN, 3, 1, R
        tmp_dim = pixel_rgb.shape[1]
        pixel_rgb = pixel_rgb.reshape(bs, num_cams, tmp_dim, pixel_rgb.shape[-1])
        pixel_rgb = pixel_rgb.permute(0, 1, 3, 2) # B, N, R, 3
        return pixel_rgb
            
    def compute_reprojection_loss(self, pred, target):
        # pred, target: B, N, HW, 3
        bs, num_cams = target.shape[:2]
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(-1) # Average over all dimensions
        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            pred_reshape = pred.reshape(bs * num_cams, *self.ray_resize, self.dims)
            target_reshape = target.reshape(bs * num_cams, *self.ray_resize, self.dims)
            ssim_loss = self.ssim(
                pred_reshape.permute(0, 3, 1, 2),
                target_reshape.permute(0, 3, 1, 2)
            ).mean(1, True).flatten(2)
            reprojection_loss = 0.15 * l1_loss + 0.85 * ssim_loss
        return reprojection_loss
    
    def get_diff(self,x1, x2):
        return torch.mean(torch.abs(x1 - x2), dim=-1)

    def calculate(self, imgs, rays, depths, img2prevImg, img2nextImg, num_cams=6):
        # Process per camera to avoid memory issues
        curr_imgs, prev_imgs, next_imgs = imgs
        bs = curr_imgs.shape[0] // num_cams
        num_rays = rays.shape[0]
        device = curr_imgs.device
        pixel_coords = torch.ones((bs * num_cams, num_rays, 4), device=device)
        pixel_coords[:, :, :2] = rays.unsqueeze(0).repeat(bs * num_cams, 1, 1)
        pixel_coords[:, :, :3] *= depths.unsqueeze(-1)
        pixel_coords = pixel_coords.unsqueeze(1).unsqueeze(-1)  # BN, HW, 4, 1

        pixel_prev, prev_mask = self.cal_pixel(img2prevImg, pixel_coords)  # B, 1, HW, 2
        pixel_next, next_mask = self.cal_pixel(img2nextImg, pixel_coords)
        pixel_prev = pixel_prev.unsqueeze(2)  # B, 1, 1, HW, 2
        pixel_next = pixel_next.unsqueeze(2)

        # Sample RGB values (B, 1, HW, 3)
        rgb_prev = self.sample_pixel(pixel_prev, prev_imgs[:, None])
        rgb_next = self.sample_pixel(pixel_next, next_imgs[:, None])
        rgb_curr = curr_imgs.permute(0, 2, 3, 1).reshape(bs * num_cams, 1, -1, 3)

        # Compute differences
        diff_prev = self.get_diff(rgb_curr, rgb_prev)  # B, 1, HW
        diff_next = self.get_diff(rgb_curr, rgb_next)
        diff_prev[~prev_mask] = 1e5
        diff_next[~next_mask] = 1e5
        loss_map_prev = diff_prev
        loss_map_next = diff_next

        if not self.no_ssim:
            ssim_shape = (bs * num_cams, *self.ray_resize, self.dims)
            
            rgb_prev_reshaped = rgb_prev.reshape(ssim_shape).permute(0, 3, 1, 2)
            rgb_next_reshaped = rgb_next.reshape(ssim_shape).permute(0, 3, 1, 2)
            rgb_curr_reshaped = rgb_curr.reshape(ssim_shape).permute(0, 3, 1, 2)
                
            # Compute per-pixel loss maps
            ssim_loss_prev_map = self.ssim(rgb_prev_reshaped, rgb_curr_reshaped) # BN, 1, H, W
            ssim_loss_next_map = self.ssim(rgb_next_reshaped, rgb_curr_reshaped) # BN, 1, H, W
            
            # Reshape back to B, 1, HW
            ssim_loss_prev_map = ssim_loss_prev_map.mean(dim=1, keepdim=True).flatten(2)
            ssim_loss_next_map = ssim_loss_next_map.mean(dim=1, keepdim=True).flatten(2)
            
            # Mask out invalid pixels
            ssim_loss_prev_map[~prev_mask] = 1e5
            ssim_loss_next_map[~next_mask] = 1e5

            loss_map_prev = 0.15 * diff_prev + 0.85 * ssim_loss_prev_map
            loss_map_next = 0.15 * diff_next + 0.85 * ssim_loss_next_map

        loss_map = [loss_map_prev, loss_map_next]

        if not self.no_automask:
            pix_curr = rays.reshape(1, 1, 1, num_rays, 2).repeat(bs * num_cams, 1, 1, 1, 1)
            target_prev = self.sample_pixel(pix_curr, prev_imgs[:, None])
            target_next = self.sample_pixel(pix_curr, next_imgs[:, None])

            # Compute static scene losses
            identity_prev = self.compute_reprojection_loss(target_prev, rgb_curr)
            identity_next = self.compute_reprojection_loss(target_next, rgb_curr)

            # Use minimum of temporal and static losses
            loss_map.append(identity_prev)
            loss_map.append(identity_next)

        return loss_map, [prev_mask, next_mask]
    
    def reproj_loss(
            self,
            curr_imgs,
            prev_imgs,
            next_imgs,
            ts, # depth map of curr_imgs
            metas,
        ):
        # curr_imgs: B, N, C, H, W
        # ts: B, num_groups, N, C, H, W (rendered depths from Gaussian rasterizer)
        # Need to extract the depth channel and handle multi-group rendering
        device = curr_imgs.device
        bs, num_cams, _, H, W = curr_imgs.shape
        ds = curr_imgs.shape[-1] // ts.shape[-1]
        curr_imgs = get_denormalized_img(curr_imgs)
        prev_imgs = get_denormalized_img(prev_imgs)
        next_imgs = get_denormalized_img(next_imgs)
        if ds > 1:
            curr_imgs = F.interpolate(
                curr_imgs.flatten(0, 1),
                scale_factor=1/ds,
                mode='bilinear',
                align_corners=False
            )
            self.ray_resize = tuple(ts.shape[-2:])
        else:
            curr_imgs = curr_imgs.flatten(0, 1)
            self.ray_resize = (H, W)
        prev_imgs = prev_imgs.flatten(0, 1)
        next_imgs = next_imgs.flatten(0, 1)

        # Handle multi-group rendering output: take the last group (most refined)
        if len(ts.shape) == 6:   # B, num_groups, N, C, H, W (C=1)
            ts = ts[:, -1, :, 0] # take last group, first channel
        elif len(ts.shape) == 5: # B, num_groups, N, H, W
            ts = ts[:, -1]       # take last group
        elif len(ts.shape) == 4: # B, N, H, W
            pass                 # Already in correct format
        else:
            raise ValueError(f"Unexpected depth map shape: {ts.shape}")

        # Create pixel grid for all pixels
        ty, tx = torch.meshgrid(
            torch.arange(H // ds, device=device, dtype=torch.float32) * ds + ds / 2,
            torch.arange(W // ds, device=device, dtype=torch.float32) * ds + ds / 2,
            indexing='ij'
        )
        rays = torch.stack([tx, ty], dim=-1).reshape(-1, 2)  # HW, 2

        # Get depths for all pixels: B, N, H, W -> BN, HW
        depths = ts.reshape(bs * num_cams, -1)
        inf_depths = depths * 1e5

        # Prepare transformation matrices - handle both batched and list formats
        if isinstance(metas, list):
            # List of dicts (batch size > 1)
            img2prevImg = torch.stack([m['img2prevImg'] for m in metas], dim=0)  # B, N, 4, 4
            img2nextImg = torch.stack([m['img2nextImg'] for m in metas], dim=0)  # B, N, 4, 4
        else:
            # Single dict (batch size = 1)
            img2prevImg = metas['img2prevImg']  # N, 4, 4 or B, N, 4, 4
            img2nextImg = metas['img2nextImg']
            if len(img2prevImg.shape) == 3:
                img2prevImg = img2prevImg.unsqueeze(0)  # 1, N, 4, 4
                img2nextImg = img2nextImg.unsqueeze(0)

        img2prevImg = img2prevImg.flatten(0, 1).to(torch.float32)[:, None, None]  # BN, 1, 1, 4, 4
        img2nextImg = img2nextImg.flatten(0, 1).to(torch.float32)[:, None, None]  # BN, 1, 1, 4, 4
        sky2prevImg = img2prevImg.clone()
        sky2nextImg = img2nextImg.clone()
        sky2prevImg[..., :3, 3] = 0.0
        sky2nextImg[..., :3, 3] = 0.0

        imgs = (curr_imgs, prev_imgs, next_imgs)
        loss_map, mask_map = self.calculate(imgs, rays, depths, img2prevImg, img2nextImg, num_cams)
        loss_sky, mask_sky = self.calculate(imgs, rays, inf_depths, sky2prevImg, sky2nextImg, num_cams)

        # loss_map.append(loss_sky[0])
        # loss_map.append(loss_sky[1])
        # ratio_check = (
        #     (loss_sky[0] < (loss_map[0] * self.ratio)) &
        #     (loss_sky[1] < (loss_map[1] * self.ratio))
        # )
        # significant_error = (loss_map[0] > 0.01) & (loss_map[1] > 0.01)
        # depth_threshold = torch.quantile(depths, 0.6, dim=1, keepdim=True)
        # is_far = (depths > depth_threshold).unsqueeze(1)
        # is_sky = ratio_check & significant_error & is_far
        # is_sky = is_sky.view(bs * num_cams, 1, *ts.shape[-2:])[:, 0]

        combined_loss = torch.stack(loss_map, dim=1)
        min_loss = torch.min(combined_loss, dim=1)[0]

        if not self.no_automask:
            final_mask = torch.ones_like(min_loss)
        else:
            final_mask = (mask_map[0] | mask_map[1]).float()
        final_loss = (min_loss * final_mask).sum() / final_mask.sum().clamp(min=1.0)

        # Average over cameras
        return final_loss


@OPENOCC_LOSS.register_module()
class MultiViewReprojLoss(TemporalViewReprojLoss):

    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight, **kwargs)

        if input_dict is None:
            self.input_dict = {
                'curr_imgs': 'curr_imgs',
                'prev_imgs': 'prev_imgs',
                'next_imgs': 'next_imgs',
                'ts': 'ts',
                'metas': 'metas',
            }
        else:
            self.input_dict = input_dict
        self.no_ssim = kwargs.get('no_ssim', False)
        self.no_automask = kwargs.get('no_automask', False)
        self.ratio = kwargs.get('ratio', 0.7)
        self.dims = kwargs.get('dims', 3)
        self.img_size = kwargs.get('img_size', [896, 1600])
        if not self.no_ssim:
            self.ssim = SSIM()
        
        self.loss_func = self.reproj_loss

    def get_overlap_mask(self, rays, img2leftImg, img2rightImg):
        bs, num_cams = img2leftImg.shape[:2]
        device = img2leftImg.device
        # Sample multiple points along the ray.
        num_samples = 2  # Number of samples along the ray
        near, far = 1.0, 81.0 # Depth range

        # Create depth steps: (num_samples, ) -> (1, 1, 1, num_samples)
        step_depths = torch.linspace(near, far, num_samples, device=device)
        step_depths = step_depths.view(1, 1, num_samples, 1)

        # Expand dimensions for ray extension
        # pixel_coords: (BN, HW, num_samples, 4)
        pixel_coords = torch.ones((bs * num_cams, rays.shape[0], num_samples, 4), device=device)
        pixel_coords[..., :2] = rays.view(1, -1, 1, 2) # (BN, HW, 1, 2) Broadcasting
        pixel_coords[..., :3] *= step_depths # Scale rays by depth (u*d, v*d, d)
        # Flatten for matrix multiplication: (BN, HW * S, 4, 1)
        pixel_coords = pixel_coords.reshape(bs * num_cams, 1, -1, 4).unsqueeze(-1)

        # cal_pixel returns:
        #   - Projected coordinates: (BN, Total_Points, 2)
        #   - Valid Mask: (BN, Total_Points) -> here Total_Points = HW * S
        _, mask_left = self.cal_pixel(img2leftImg, pixel_coords)
        _, mask_right = self.cal_pixel(img2rightImg, pixel_coords)

        # Check if *any* point along the ray falls into the neighbor's view frustum.
        # (BN, HW * S) -> (BN, HW, S) -> (BN, HW)
        mask_left = mask_left.view(bs * num_cams, -1, num_samples).any(dim=-1)
        mask_right = mask_right.view(bs * num_cams, -1, num_samples).any(dim=-1)

        return mask_left, mask_right
    
    def reproj_loss(
            self,
            imgs,
            ts, # depth map of curr_imgs
            metas,
        ):
        # imgs: B, N, C, H, W
        # ts: B, num_groups, N, C, H, W (rendered depths from Gaussian rasterizer)
        # Need to extract the depth channel and handle multi-group rendering

        # Assumed Loader Order: ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        #                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        # Geometric Neighbors (Left/Right) Definition:
        # 0(F) <-> 2(FL), 1(FR)
        # 1(FR)<-> 0(F), 5(BR)
        # 2(FL)<-> 0(F), 4(BL)
        # 3(B) <-> 4(BL), 5(BR)
        # 4(BL)<-> 2(FL), 3(B)
        # 5(BR)<-> 1(FR), 3(B)
        left_idx  = [2, 0, 4, 4, 2, 1] # 'Left' neighbor index for each camera
        right_idx = [1, 5, 0, 5, 3, 3] # 'Right' neighbor index for each camera

        left_imgs = imgs[:, left_idx]
        right_imgs = imgs[:, right_idx]

        device = imgs.device
        bs, num_cams, _, H, W = imgs.shape
        ds = imgs.shape[-1] // ts.shape[-1]
        imgs = get_denormalized_img(imgs)
        left_imgs = get_denormalized_img(left_imgs)
        right_imgs = get_denormalized_img(right_imgs)
        if ds > 1:
            imgs = F.interpolate(
                imgs.flatten(0, 1),
                scale_factor=1/ds,
                mode='bilinear',
                align_corners=False
            )
            self.ray_resize = tuple(ts.shape[-2:])
        else:
            imgs = imgs.flatten(0, 1)
            self.ray_resize = (H, W)
        left_imgs = left_imgs.flatten(0, 1)
        right_idx = right_idx.flatten(0, 1)

        # Handle multi-group rendering output: take the last group (most refined)
        if len(ts.shape) == 6:   # B, num_groups, N, C, H, W (C=1)
            ts = ts[:, -1, :, 0] # take last group, first channel
        elif len(ts.shape) == 5: # B, num_groups, N, H, W
            ts = ts[:, -1]       # take last group
        elif len(ts.shape) == 4: # B, N, H, W
            pass                 # Already in correct format
        else:
            raise ValueError(f"Unexpected depth map shape: {ts.shape}")

        # Create pixel grid for all pixels
        ty, tx = torch.meshgrid(
            torch.arange(H // ds, device=device, dtype=torch.float32) * ds + ds / 2,
            torch.arange(W // ds, device=device, dtype=torch.float32) * ds + ds / 2,
            indexing='ij'
        )
        rays = torch.stack([tx, ty], dim=-1).reshape(-1, 2)  # HW, 2

        # Get depths for all pixels: B, N, H, W -> BN, HW
        depths = ts.reshape(bs * num_cams, -1)

        # Prepare transformation matrices - handle both batched and list formats
        if isinstance(metas, list):
            # List of dicts (batch size > 1)
            lidar2img = torch.stack([m['lidar2img'] for m in metas], dim=0)  # B, N, 4, 4
        else:
            # Single dict (batch size = 1)
            lidar2img = metas['lidar2img']  # N, 4, 4 or B, N, 4, 4

        # Calculate Transformation Matrices
        # (B, 6, 4, 4)
        img2leftImg = torch.zeros_like(lidar2img)
        img2rightImg = torch.zeros_like(lidar2img)

        # Compute matrices for each sample in the batch
        for b in range(bs):
            l2i = lidar2img[b] # (6, 4, 4)
            # Compute Inverse (Assuming Projection Matrix is 4x4)
            i2l = torch.linalg.inv(l2i) 
            
            # Compute transformation from current view to neighbor views
            img2leftImg[b] = l2i[left_idx] @ i2l   # Neighbor_Proj @ Current_Inv
            img2rightImg[b] = l2i[right_idx] @ i2l

        # Flatten for batch processing: (BN, 1, 1, 4, 4)
        img2leftImg = img2leftImg.flatten(0, 1).to(torch.float32)[:, None, None] 
        img2rightImg = img2rightImg.flatten(0, 1).to(torch.float32)[:, None, None]

        mask_left, mask_right = self.get_overlap_mask(rays, img2leftImg, img2rightImg)
        # If a pixel overlaps with either the left or right neighbor, it is an overlap region.
        overlap_mask = mask_left | mask_right
        overlap_mask = overlap_mask.view(bs, num_cams, H // ds, W // ds)

        imgs = (imgs, left_imgs, right_imgs)
        loss_map, _ = self.calculate(imgs, rays, depths, img2leftImg, img2rightImg, num_cams)

        combined_loss = torch.stack(loss_map, dim=1)
        min_loss = torch.min(combined_loss, dim=1)[0]

        if not self.no_automask:
            final_mask = torch.ones_like(min_loss)
        else:
            final_mask = (mask_left | mask_right).float()
        final_loss = (min_loss * final_mask).sum() / final_mask.sum().clamp(min=1.0)

        # Average over cameras
        return final_loss
