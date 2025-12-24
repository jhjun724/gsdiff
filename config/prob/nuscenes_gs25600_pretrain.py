_base_ = [
    '../_base_/misc.py',
    '../_base_/model.py',
    '../_base_/surroundocc.py'
]

# =========== data config ==============
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"

final_shape = (800, 448)
data_aug_conf = {
    "resize_lim": (0.5, 0.61),
    "final_dim": final_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900, # original image height
    "W": 1600, # original image width
    "rand_flip": True,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

lidar_sweep_num = 10

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=lidar_sweep_num - 1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=80.0),
    dict(type="LoadDA3Depth", img_size=final_shape[::-1]),
    dict(type="LoadSemantics"),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=lidar_sweep_num - 1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=80.0),
    dict(type="LoadDA3Depth", img_size=final_shape[::-1]),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_pre.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    load_interval=4,
    return_keys=[
        'img',
        'projection_mat',
        'image_wh',
        'ori_img',
        'cam_positions',
        'focal_positions',
        'cam_params',
        'points',
        'da3_depths',
        'semantics',
    ],
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_pre.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    return_keys=[
        'img',
        'projection_mat',
        'image_wh',
        'ori_img',
        'cam_positions',
        'focal_positions',
        'cam_params',
        'points',
        'da3_depths',
    ],
)

train_loader = dict(
    batch_size=2,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=1,
    num_workers=1
)
# =========== misc config ==============
amp = False
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=4e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35

max_epochs = 12
max_keep_ckpts = 2
save_interval = 6
# ========= loss config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type="RGBLoss",
            weight=10.0,
        ),
        dict(
            type="RenderedDepthLoss",
            weight=10.0,
            depth_range=[1.0, 50.0],
        ),
        dict(
            type="DistillLoss",
            weight=0.0,
        ),
    ]
)

loss_input_convertion = dict(
    rendered_rgbs="rendered_rgbs",
    imgs="imgs",
    rendered_depths="rendered_depths",
    points="points",
    metas="metas",
    rendered_feats="rendered_feats",
    dino_feats="dino_feats",
)
# ========= model config ===============
embed_dims = 128
num_decoder = 4
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_rgb = False
include_opa = True
anchor_grad = False
feat_grad = False
semantics = False
contracted = -1.0
semantic_dim = 17
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

class_names = [
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation'
]

model = dict(
    # freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    render=True,
    mvs=False,
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1,
        out_channels=embed_dims),
    lifter=dict(
        type='GaussianLifterV3',
        num_anchor=33600,
        embed_dims=embed_dims,
        anchor_grad=anchor_grad,
        feat_grad=feat_grad,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_rgb=include_rgb,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        pretraining=True,
        contracted=contracted,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path="work_dirs/prob/init/init.pth",
        deterministic=False,
        random_samples=0),
    encoder=dict(
        type='GaussianOccEncoder',
        contracted=contracted,
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            include_rgb=include_rgb,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            _delete_=True,
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            residual_mode="none",
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV3',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[0.0, 0.0, 0.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            include_rgb=include_rgb,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[1.0, 1.0, 1.0],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True,
        ),
        num_decoder=num_decoder,
        operation_order=[
            "identity",
            "deformable",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "identity",
            "spconv",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",
            
            "refine",
        ] * num_decoder,
    ),
    head=None
)
