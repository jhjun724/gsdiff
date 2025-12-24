import os
from copy import deepcopy
import numpy as np
from skimage import transform
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        vis_scene_index=-1,
        phase='train',
        load_interval=1,
        return_keys=[
            'img',
            'projection_mat',
            'image_wh',
            'occ_label',
            'occ_xyz',
            'occ_cam_mask',
            'ori_img',
            'cam_positions',
            'focal_positions',
            'cam_params',
        ],
    ):
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        # self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
        self.keyframes = sorted(self.keyframes, key=lambda x: self.scene_infos[x[0]][x[1]]['timestamp'])
        self.keyframes = self.keyframes[::load_interval]

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')
        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.return_keys = return_keys
        if vis_scene_index >= 0:
            frame = self.keyframes[vis_scene_index]
            num_frames = len(self.scene_infos[frame[0]])
            self.keyframes = [(frame[0], i) for i in range(num_frames)]
            print(f'Scene length: {len(self.keyframes)}')
        elif vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize_ls, resize_dims_ls, crop_ls, flip_ls, rotate_ls = [], [], [], [], []
            for _ in self.sensor_types:
                resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = (
                    int(
                        (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                        * newH
                    )
                    - fH
                )
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
                flip = False
                if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                    flip = True
                rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
                resize_ls.append(resize)
                resize_dims_ls.append(resize_dims)
                crop_ls.append(crop)
                flip_ls.append(flip)
                rotate_ls.append(rotate)
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            resize_ls = [resize] * len(self.sensor_types)
            resize_dims_ls = [resize_dims] * len(self.sensor_types)
            crop_ls = [crop] * len(self.sensor_types)
            flip_ls = [flip] * len(self.sensor_types)
            rotate_ls = [rotate] * len(self.sensor_types)
        return resize_ls, resize_dims_ls, crop_ls, flip_ls, rotate_ls

    def _aug_subsamples(self, imgs, aug_configs):
        resize = aug_configs[0]
        resize_dims = aug_configs[1]
        crop = aug_configs[2]
        flip = aug_configs[3]
        rotate = aug_configs[4]
        origin_type = imgs.dtype
        new_imgs = []
        for i, img in enumerate(imgs):
            img = transform.resize(img, resize_dims[i][::-1])
            img = img[crop[i][1]:crop[i][3], crop[i][0]:crop[i][2]]
            img = img[:, ::-1] if flip[i] else img
            img = transform.rotate(img, rotate[i])
            new_imgs.append(img.astype(origin_type))
        new_imgs = np.stack(new_imgs, axis=0)
        return new_imgs

    def __getitem__(self, index):
        scene_token, scene_index = self.keyframes[index]
        info = deepcopy(self.scene_infos[scene_token][scene_index])
        input_dict = self.get_data_info(info)

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)

        return_dict = {k: input_dict[k] for k in self.return_keys}
        return_dict['index'] = index
        return_dict['token'] = info['token']
        return_dict['lidar2img'] = input_dict['lidar2img']
        if 'da3_depths' in input_dict:
            return_dict['da3_depths'] = self._aug_subsamples(
                input_dict['da3_depths'], input_dict["aug_configs"]
            )
        if 'semantics' in input_dict:
            return_dict['semantics'] = self._aug_subsamples(
                input_dict['semantics'], input_dict["aug_configs"]
            )
            if 'da3_depths' in input_dict:
                return_dict['semantics'][input_dict['da3_depths'] < 80.0] = False
        return return_dict

    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        ego2image_rts = []
        cam_positions = []
        focal_positions = []
        dinov3_paths = []
        da3_paths = []
        sem_paths = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T

        pts_filename = getattr(
            info,
            'pts_filename',
            os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename'])
        )    

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))
            dinov3_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_dinov3/'))
            da3_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_depth_da3/').replace('.jpg', '.npy'))
            sem_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_mask/').replace('.jpg', '.png'))

            img2global, cam2global, _ = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            lidar2cam = np.linalg.inv(cam2global) @ lidar2global

            lidar2img_rts.append(lidar2img)
            lidar2cam_rts.append(lidar2cam)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)

            img2lidar = np.linalg.inv(lidar2global) @ img2global
            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic
            cam_intrinsics.append(viewpad)

            cam_position = img2lidar @ viewpad @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = img2lidar @ viewpad @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])
            
        input_dict =dict(
            # sample_idx=info["token"],
            sample_idx=info.get("token", ""),
            # occ_path=info["occ_path"],
            occ_path=info.get("occ_path", ""),
            timestamp=info["timestamp"] / 1e6,
            img_filename=image_paths,
            pts_filename=pts_filename,
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts),
            lidar2cam=np.asarray(lidar2cam_rts),
            ego2img=np.asarray(ego2image_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions),
            cam_intrinsic=np.asarray(cam_intrinsics),
            sweeps=getattr(info, 'sweeps', []),
            dinov3_paths=dinov3_paths,
            da3_paths=da3_paths,
            sem_paths=sem_paths,
        )

        return input_dict

    def __len__(self):
        return len(self.keyframes)