import cv2
import torch
import numpy as np
import math

import mmengine
from mmdet3d.structures.points import BasePoints, get_points_type

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class LoadCameraParam(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        cam_nums=6,
        znear=1.0,
        zfar=50,
    ):
        self.cam_nums = cam_nums
        self.znear = znear
        self.zfar = zfar
    
    def _focal2fov(self, focal, pixel):
        # return 2 * math.atan(pixel / (2 * focal)) * (180 / np.pi) 
        return 2 * math.atan(pixel / (2 * focal))

    def _getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    def _getProjectionMatrixShift(
        self, znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY
    ):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        # the origin at center of image plane
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # shift the frame window due to the non-zero principle point offsets
        offset_x = cx - (width/2)
        offset_x = (offset_x/focal_x)*znear
        offset_y = cy - (height/2)
        offset_y = (offset_y/focal_y)*znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        cam_params = []
        for cam_idx in range(self.cam_nums):
            height = results['pad_before_shape'][cam_idx][0]
            width = results['pad_before_shape'][cam_idx][1]

            #TODO
            if isinstance(results['cam_intrinsic'][cam_idx], list) \
                or results['cam_intrinsic'][cam_idx].ndim >=3:

                fx = results['cam_intrinsic'][cam_idx][0][0, 0]
                fy = results['cam_intrinsic'][cam_idx][0][1, 1]
                cx = results['cam_intrinsic'][cam_idx][0][0, 2]
                cy = results['cam_intrinsic'][cam_idx][0][1, 2]

                fovx = self._focal2fov(fx, width)
                fovy = self._focal2fov(fy, height)

                w2c = results['lidar2cam'][cam_idx][0]
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                viewmatrix = self._getWorld2View2(R, T, translate=np.zeros(3), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)

                projmatrix_ = self._getProjectionMatrixShift(
                    self.znear, self.zfar, fx, fy, cx, cy, width, height, fovx, fovy
                ).transpose(0,1)

            else:
                fx = results['cam_intrinsic'][cam_idx][0, 0]
                fy = results['cam_intrinsic'][cam_idx][1, 1]
                cx = results['cam_intrinsic'][cam_idx][0, 2]
                cy = results['cam_intrinsic'][cam_idx][1, 2]

                fovx = self._focal2fov(fx, width)
                fovy = self._focal2fov(fy, height)

                w2c = results['lidar2cam'][cam_idx]
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                viewmatrix = self._getWorld2View2(R, T, translate=np.zeros(3), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)

                projmatrix_ = self._getProjectionMatrixShift(
                    self.znear, self.zfar, fx, fy, cx, cy, width, height, fovx, fovy
                ).transpose(0,1)
                projmatrix_ = torch.tensor(projmatrix_, dtype=torch.float32)

            # w2i
            full_proj_transform = (
                viewmatrix.unsqueeze(0).bmm(projmatrix_.unsqueeze(0))
            ).squeeze(0)

            cam_pos = viewmatrix.inverse()[3, :3]
            cam_param = {
                'height': height, 
                'width':width, 
                'fovx':fovx, 
                'fovy':fovy, 
                'viewmatrix':viewmatrix, 
                'projmatrix':full_proj_transform, 
                'cam_pos':cam_pos
            }
            cam_params.append(cam_param)
        
        results['cam_params'] = cam_params

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # if self.file_client is None:
        #     self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            # pts_bytes = self.file_client.get(pts_filename)
            pts_bytes = mmengine.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # if self.file_client is None:
        #     self.file_client = mmengine.FileClient(**self.file_client_args)
        try:
            # pts_bytes = self.file_client.get(pts_filename)
            pts_bytes = mmengine.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@OPENOCC_TRANSFORMS.register_module()
class LoadDinov3Features(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        model_name='vits16',
    ):
        self.model_name = model_name

    def __call__(self, results):
        dinov3_paths = results['dinov3_paths']
        dinov3_feats = []
        for path in dinov3_paths:
            path = path.replace('.jpg', f'_{self.model_name}.pth')
            dinov3_feats.append(torch.load(path))
        results['dinov3_feats'] = dinov3_feats
        return results


@OPENOCC_TRANSFORMS.register_module()
class LoadDA3Depth(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        img_size=(900, 1600),
    ):
        self.img_size = img_size

    def __call__(self, results):
        da3_paths = results['da3_paths']
        da3_depths = []
        for path in da3_paths:
            da3_depths.append(np.load(path))
        da3_depths = np.stack(da3_depths, axis=0)
        results['da3_depths'] = da3_depths
        return results


@OPENOCC_TRANSFORMS.register_module()
class LoadSemantics(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, img_size=(900, 1600)):
        self.img_size = img_size
        self.semantic_map = np.array([
            0,   # ignore
            4,   # sedan      -> car
            11,  # highway    -> driveable_surface
            3,   # bus        -> bus
            10,  # truck      -> truck
            14,  # terrain    -> terrain
            16,  # tree       -> vegetation
            13,  # sidewalk   -> sidewalk
            2,   # bicycle    -> bycycle
            1,   # barrier    -> barrier
            7,   # person     -> pedestrian
            15,  # building   -> manmade
            6,   # motorcycle -> motorcycle
            5,   # crane      -> construction_vehicle
            9,   # trailer    -> trailer
            8,   # cone       -> traffic_cone
            17   # sky        -> ignore
        ], dtype=np.int8)

    def __call__(self, results):
        sem_paths = results['sem_paths']
        semantics = []
        for path in sem_paths:
            # semantics.append(self.semantic_map[
            #     np.fromfile(path, dtype=np.int8).reshape(self.img_size)[:, ::-1]
            # ])
            semantics.append(cv2.imread(path, 0).astype('bool'))
        semantics = np.stack(semantics, axis=0)
        results['semantics'] = semantics
        return results
