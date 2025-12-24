import os
import random
import math
import numpy as np
import torch
from skimage import transform
from copy import deepcopy
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmcv
import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global


@OPENOCC_DATASET.register_module()
class NuScenesTemporalDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        curr_prob=0.333,
        prev_prob=0.5,
        composite_prev_next=True,
        choose_nearest=True,
        sensor_mus=[0.5, 0.5],
        sensor_sigma=0.5,
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
        self.training = True if phase == 'train' else False
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        # self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
        self.keyframes = sorted(self.keyframes, key=lambda x: self.scene_infos[x[0]][x[1]]['timestamp'])
        self.keyframes = self.keyframes[::load_interval]
        # self.except_non_temporal()
        # self.get_only_keyframes()

        self.curr_prob = curr_prob if self.training else 1.0
        self.prev_prob = prev_prob
        self.composite_prev_next = composite_prev_next
        self.choose_nearest = choose_nearest
        self.sensor_mus = {
            'CAM_FRONT': sensor_mus[0],
            'CAM_FRONT_RIGHT': sensor_mus[1],
            'CAM_FRONT_LEFT': sensor_mus[1],
            'CAM_BACK': sensor_mus[0],
            'CAM_BACK_LEFT': sensor_mus[1],
            'CAM_BACK_RIGHT': sensor_mus[1],
        }
        self.sensor_sigma = sensor_sigma

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

    def except_non_temporal(self):
        while True:
            non_temporals = []
            for i, (scene_token, scene_index) in enumerate(self.keyframes):
                info = self.scene_infos[scene_token][scene_index]
                if len(info['prev_samples']) == 0 or len(info['next_samples']) == 0:
                    non_temporals.append((scene_token, scene_index))
            self.keyframes = [keyframe for keyframe in self.keyframes if keyframe not in non_temporals]
            for i, (scene_token, scene_index) in enumerate(self.keyframes):
                info = self.scene_infos[scene_token][scene_index]
                prev_samples_idx = []
                next_samples_idx = []
                for j, prev_sample in enumerate(info['prev_samples']):
                    if prev_sample not in non_temporals:
                        prev_samples_idx.append(j)
                info['prev_samples'] = [info['prev_samples'][idx] for idx in prev_samples_idx]
                info['prev_dists'] = [info['prev_dists'][idx] for idx in prev_samples_idx]
                for j, next_sample in enumerate(info['next_samples']):
                    if next_sample not in non_temporals:
                        next_samples_idx.append(j)
                info['next_samples'] = [info['next_samples'][idx] for idx in next_samples_idx]
                info['next_dists'] = [info['next_dists'][idx] for idx in next_samples_idx]
                self.scene_infos[scene_token][scene_index] = info
            if len(non_temporals) == 0:
                break

    def get_only_keyframes(self):
        for i, keyframe in enumerate(self.keyframes):
            scene_token, scene_index = keyframe
            info = self.scene_infos[scene_token][scene_index]
            prev_idx = []
            next_idx = []
            for j, prev_sample in enumerate(info['prev_samples']):
                prev_info = self.scene_infos[prev_sample[0]][prev_sample[1]]
                if prev_info['is_key_frame']:
                    prev_idx.append(j)
            for j, next_sample in enumerate(info['next_samples']):
                next_info = self.scene_infos[next_sample[0]][next_sample[1]]
                if next_info['is_key_frame']:
                    next_idx.append(j)
            info['prev_samples'] = [info['prev_samples'][idx] for idx in prev_idx]
            info['prev_dists'] = [info['prev_dists'][idx] for idx in prev_idx]
            info['next_samples'] = [info['next_samples'][idx] for idx in next_idx]
            info['next_dists'] = [info['next_dists'][idx] for idx in next_idx]
            self.scene_infos[scene_token][scene_index] = info

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
        return [resize_ls, resize_dims_ls, crop_ls, flip_ls, rotate_ls]

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

    def composite_dict(self, anchor_info):
        datas = []
        for prefix in ['prev_', 'next_']:
            data = dict()
            dists = np.asarray(anchor_info[prefix + 'dists'])
            for sensor_type in self.sensor_types:
                mu = self.sensor_mus[sensor_type]
                sigma = self.sensor_sigma
                probs = 1 / math.sqrt(2 * math.pi) / sigma \
                    * np.exp(-1 / (2 * sigma * sigma) * ((dists - mu) ** 2))
                probs = probs / np.sum(probs)
                idx = np.random.choice(len(dists), p=probs)
                scene_token, sample_idx = anchor_info[prefix + 'samples'][idx]
                data.update({sensor_type: self.scene_infos[scene_token][sample_idx]['data'][sensor_type]})
            datas.append(data)
        return {'data': datas[0]}, {'data': datas[1]}

    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def get_temp_cam_params(self, pad_before_shape, cam_intrinsic, lidar2temCam):
        results = {
            'pad_before_shape': pad_before_shape,
            'cam_intrinsic': cam_intrinsic,
            'lidar2cam': lidar2temCam,
        }
        pipeline = None
        for p in self.pipeline:
            if p.__class__.__name__ == 'LoadCameraParam':
                pipeline = p
                break
        if pipeline is not None:
            results = pipeline(results)
        return results['cam_params']

    def __getitem__(self, index):
        # scene_token, scene_index = self.keyframes[index]
        # info = deepcopy(self.scene_infos[scene_token][scene_index])
        idx, info, anchor_info, anchor_prev, anchor_next = self.get_temporal_sweeps(index)
        input_dict = self.get_data_info(info)
        cam_intrinsic = input_dict['cam_intrinsic'].copy()
        if self.training:
            anchor_dict = self.get_data_info_anchor(info, anchor_info)
            prev_dict = self.get_data_info_temporal(anchor_info, anchor_prev, info)
            next_dict = self.get_data_info_temporal(anchor_info, anchor_next, info)
            prev_dict['temImg2lidar'] = anchor_dict['temImg2lidar'] @ np.linalg.inv(prev_dict['img2temImg'])
            next_dict['temImg2lidar'] = anchor_dict['temImg2lidar'] @ np.linalg.inv(next_dict['img2temImg'])

        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)

        result_dict = {k: input_dict[k] for k in self.return_keys}
        result_dict['index'] = idx
        result_dict['token'] = info['token']

        result_dict.update({
            'input_imgs_path': input_dict['img_filename'],
            'lidar2img': input_dict['lidar2img'],
            'img2lidar': input_dict['img2lidar'],
            'intrinsic': input_dict['cam_intrinsic'],
            'cam2ego': input_dict['cam2ego'],
            'ego2lidar': input_dict['ego2lidar'],
        })

        if self.training:
            anchor_dict = self.read_surround_imgs(anchor_dict, cam_intrinsic)
            prev_dict = self.read_surround_imgs(prev_dict, cam_intrinsic)
            next_dict = self.read_surround_imgs(next_dict, cam_intrinsic)

            img2prevImg = np.linalg.inv(prev_dict['temImg2lidar']) @ anchor_dict['temImg2lidar']
            img2nextImg = np.linalg.inv(next_dict['temImg2lidar']) @ anchor_dict['temImg2lidar']

            result_dict.update({
                # 'curr_imgs_path': anchor_dict['image_paths'],
                # 'prev_imgs_path': prev_dict['image_paths'],
                # 'next_imgs_path': next_dict['image_paths'],
                'temImg2lidar': anchor_dict['temImg2lidar'],
                'lidar2temCam': anchor_dict['lidar2temCam'],
                'img2prevImg': img2prevImg,
                'img2nextImg': img2nextImg,
            })

            result_dict.update({
                'curr_imgs': anchor_dict['img'],
                'prev_imgs': prev_dict['img'],
                'next_imgs': next_dict['img'],
            })

            # temCam_params = self.get_temp_cam_params(
            #     input_dict['pad_before_shape'], 
            #     input_dict['cam_intrinsic'], 
            #     result_dict['lidar2temCam']
            # )
            result_dict.update({
                'temCam_params': anchor_dict['cam_params'],
                'prevCam_params': prev_dict['cam_params'],
                'nextCam_params': next_dict['cam_params'],
            })
        else:
            result_dict.update({
                'temImg2lidar': input_dict['img2lidar'],
                'lidar2temCam': input_dict['lidar2cam'],
                'img2prevImg': np.eye(4)[None].repeat(6, 0),
                'img2nextImg': np.eye(4)[None].repeat(6, 0),
                'temCam_params': input_dict['cam_params'],
                'prevCam_params': input_dict['cam_params'],
                'nextCam_params': input_dict['cam_params'],
                'curr_imgs': input_dict['img'],
                'prev_imgs': input_dict['img'],
                'next_imgs': input_dict['img'],
            })

        if 'da3_depths' in input_dict:
            result_dict['da3_depths'] = self._aug_subsamples(
                input_dict['da3_depths'], input_dict["aug_configs"]
            )
        if 'semantics' in input_dict:
            result_dict['semantics'] = self._aug_subsamples(
                input_dict['semantics'], input_dict["aug_configs"]
            )
            if 'da3_depths' in input_dict:
                result_dict['semantics'][input_dict['da3_depths'] < 80.0] = False

        return result_dict
    
    def read_surround_imgs(self, input_dict, cam_intrinsic, aug_configs=None):
        if aug_configs is None:
            aug_configs = self._sample_augmentation()
        aug_configs[3] = [False] * cam_intrinsic.shape[0]
        results = {
            'img_filename': input_dict['image_paths'],
            'aug_configs': aug_configs,
            'lidar2img': np.linalg.inv(input_dict['temImg2lidar']),
            'lidar2cam': input_dict.get('lidar2temCam', None),
            'cam_intrinsic': cam_intrinsic,
        }
        for t in self.pipeline:
            if 'Points' in t.__class__.__name__:
                continue
            if results.get('lidar2cam', None) is None:
                if t.__class__.__name__ == 'LoadCameraParam':
                    continue
            if results.get('da3_paths', None) is None:
                if t.__class__.__name__ == 'LoadDA3Depth':
                    continue
            if results.get('sem_paths', None) is None:
                if t.__class__.__name__ == 'LoadSemantic':
                    continue
            results = t(results)
        input_dict.update({
            'img': results['img'],
            'temImg2lidar': np.linalg.inv(results['lidar2img']),
            'cam_params': results['cam_params'] if 'cam_params' in results else None,
        })
        return input_dict

    def get_temporal_sweeps(self, index):
        #### 1. get color, temporal_depth choice if necessary
        if random.random() < self.curr_prob:
            temporal_supervision = 'curr'
        elif random.random() < self.prev_prob:
            temporal_supervision = 'prev'
        else:
            temporal_supervision = 'next'
        pass

        #### 2. get self, prev, next infos for the stem, and also temp_depth info
        while True:
            scene_token, scene_index = self.keyframes[index]
            info = deepcopy(self.scene_infos[scene_token][scene_index])

            if not self.training:
                return index, info, deepcopy(info), None, None

            if temporal_supervision == 'prev' and len(info['prev_samples']) == 0:
                temporal_supervision = 'curr'
            if temporal_supervision == 'next' and len(info['next_samples']) == 0:
                temporal_supervision = 'curr'

            if temporal_supervision == 'curr':
                anchor_info = deepcopy(info)
            elif temporal_supervision == 'prev':
                if len(info['prev_samples']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scene_token, anchor_info_id = info['prev_samples'][np.random.randint(len(info['prev_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['prev_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id <= scene_index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])
            else:
                if len(info['next_samples']) == 0:
                    index = np.random.randint(len(self))
                    continue
                anchor_scene_token, anchor_info_id = info['next_samples'][np.random.randint(len(info['next_samples']))]
                # anchor_scene_token, anchor_info_id = np.random.choice(info['next_samples'])
                assert anchor_scene_token == scene_token and anchor_info_id >= scene_index
                anchor_info = deepcopy(self.scene_infos[scene_token][anchor_info_id])

            # if len(info['prev_samples']) == 0:
            #     info['prev_samples'] = [(scene_token, scene_index)]
            # if len(info['next_samples']) == 0:
            #     info['next_samples'] = [(scene_token, scene_index)]

            if len(anchor_info['prev_samples']) == 0 or \
                len(anchor_info['next_samples']) == 0:
                index = np.random.randint(len(self))
                continue

            if self.composite_prev_next:
                anchor_prev, anchor_next = self.composite_dict(anchor_info)
            else:
                if self.choose_nearest:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][0]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][0]
                else:
                    anchor_prev_scene_token, anchor_prev_idx = anchor_info['prev_samples'][np.random.randint(len(anchor_info['prev_samples']))]
                    anchor_next_scene_token, anchor_next_idx = anchor_info['next_samples'][np.random.randint(len(anchor_info['next_samples']))]
                    # anchor_prev_scene_token, anchor_prev_idx = np.random.choice(anchor_info['prev_samples'])
                    # anchor_next_scene_token, anchor_next_idx = np.random.choice(anchor_info['next_samples'])
                assert anchor_prev_scene_token == scene_token and \
                    anchor_next_scene_token == scene_token
                anchor_prev = deepcopy(self.scene_infos[scene_token][anchor_prev_idx])
                anchor_next = deepcopy(self.scene_infos[scene_token][anchor_next_idx])
            break

        return index, info, anchor_info, anchor_prev, anchor_next

    def get_data_info_temporal(self, info, info_tem, ref_info=None):
        image_paths = []
        img2temImg_rts = []

        if ref_info is not None:
            lidar2temCam_rts = []
            lidar2global = get_lidar2global(
                ref_info['data']['LIDAR_TOP']['calib'],
                ref_info['data']['LIDAR_TOP']['pose']
            )

        for cam_type in self.sensor_types:
            cam_info_tem = info_tem['data'][cam_type]
            cam_info = info['data'][cam_type]
            image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

            temImg2global, temCam2global, _ = get_img2global(cam_info_tem['calib'], cam_info_tem['pose'])
            img2global, cam2global, cam2ego = get_img2global(cam_info['calib'], cam_info['pose'])

            img2temImg = np.linalg.inv(temImg2global) @ img2global            
            img2temImg_rts.append(img2temImg)

            if ref_info is not None:
                lidar2temCam = np.linalg.inv(temCam2global) @ lidar2global
                lidar2temCam_rts.append(lidar2temCam)

        out_dict = dict(
            image_paths=image_paths,
            img2temImg=np.asarray(img2temImg_rts),
            lidar2temCam=np.asarray(lidar2temCam_rts) if ref_info is not None else None
        )
        return out_dict
    
    def get_data_info_anchor(self, info, info_tem):
        image_paths = []
        temImg2lidar_rts = []
        lidar2temCam_rts = []

        lidar2global = get_lidar2global(
            info['data']['LIDAR_TOP']['calib'],
            info['data']['LIDAR_TOP']['pose']
        )

        for cam_type in self.sensor_types:
            cam_info_tem = info_tem['data'][cam_type]
            image_paths.append(os.path.join(self.data_path, cam_info_tem['filename']))

            temImg2global, temCam2global, temCam2ego = get_img2global(
                cam_info_tem['calib'], cam_info_tem['pose']
            )

            temImg2lidar = np.linalg.inv(lidar2global) @ temImg2global
            lidar2temCam = np.linalg.inv(temCam2global) @ lidar2global
            temImg2lidar_rts.append(temImg2lidar)
            lidar2temCam_rts.append(lidar2temCam)

        out_dict = dict(
            image_paths=image_paths,
            temImg2lidar=np.asarray(temImg2lidar_rts),
            lidar2temCam=np.asarray(lidar2temCam_rts)
        )
        return out_dict
    
    def get_data_info(self, info):
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        ego2image_rts = []
        cam2ego_rts = []
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

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))
            dinov3_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_dinov3/'))
            da3_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_depth_da3/').replace('.jpg', '.npy'))
            sem_paths.append(image_paths[-1].replace('nuscenes/', 'nuscenes_mask/').replace('.jpg', '.png'))

            img2global, cam2global, cam2ego = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            lidar2cam = np.linalg.inv(cam2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            lidar2img_rts.append(lidar2img)
            lidar2cam_rts.append(lidar2cam)
            img2lidar_rts.append(img2lidar)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)
            cam2ego_rts.append(cam2ego)

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
            pts_filename=info['pts_filename'],
            ego2lidar=ego2lidar,
            lidar2img=np.asarray(lidar2img_rts),
            lidar2cam=np.asarray(lidar2cam_rts),
            img2lidar=np.asarray(img2lidar_rts),
            ego2img=np.asarray(ego2image_rts),
            cam2ego=np.asarray(cam2ego_rts),
            cam_positions=np.asarray(cam_positions),
            focal_positions=np.asarray(focal_positions),
            cam_intrinsic=np.asarray(cam_intrinsics),
            sweeps=info['sweeps'],
            dinov3_paths=dinov3_paths,
            da3_paths=da3_paths,
            sem_paths=sem_paths,
        )

        return input_dict

    def __len__(self):
        return len(self.keyframes)