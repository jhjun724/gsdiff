_base_ = [
    '../_base_/misc.py',
    '../_base_/model.py',
    '../_base_/surroundocc.py'
]

# =========== data config ==============
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/surroundocc/samples"

input_shape = (1600, 896)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

lidar_sweep_num = 10

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=100.0),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancySurroundOcc", occ_path=occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=100.0),
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
    load_interval=2,
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
        # 'points',
        'lidar2img',
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
        'occ_label',
        'occ_xyz',
        'occ_cam_mask',
        'ori_img',
        'cam_positions',
        'focal_positions',
        'cam_params',
        # 'points',
        'lidar2img',
    ],
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
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],
            ignore_empty=False,
            lovasz_use_softmax=False,
        ),
        dict(
            type="PixelDistributionLoss",
            weight=1.0,
            use_sigmoid=False,
        ),
        # dict(
        #     type="RGBLoss",
        #     weight=10.0,
        # ),
        # dict(
        #     type="DepthLoss",
        #     weight=10.0,
        #     depth_range=[1.0, 80.0],
        # ),
        # dict(
        #     type="DistillLoss",
        #     weight=1.0,
        # ),
        # dict(
        #     type="BinaryCrossEntropyLoss",
        #     weight=10.0,
        #     empty_label=17,
        #     class_weights=[1.0, 1.0],
        # ),
        # dict(
        #     type='DensityLoss',
        #     weight=0.01,
        #     thresh=0.0,
        # )
    ]
)

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    bin_logits="bin_logits",
    density="density",
    pixel_logits="pixel_logits",
    pixel_gt="pixel_gt",
    # rendered_rgbs="rendered_rgbs",
    # imgs="imgs",
    # rendered_depths="rendered_depths",
    # points="points",
    # metas="metas",
    # rendered_feats="rendered_feats",
    # dino_feats="dino_feats"
)
# ========= model config ===============
embed_dims = 128
num_decoder = 4
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
include_rgb = True
anchor_grad = False
feat_grad = True
# load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
load_from = 'out/nuscenes_gs25600_pretrain_0_1/latest.pth'
semantics = False
semantic_dim = 17

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
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    render=False,
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
        num_anchor=19200,
        embed_dims=embed_dims,
        anchor_grad=anchor_grad,
        feat_grad=feat_grad,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        include_rgb=include_rgb,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        pretraining=False,
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
        pretrained_path=None,
        deterministic=False,
        random_samples=6400),
    encoder=dict(
        type='GaussianOccEncoder',
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
            unit_xyz=[4.0, 4.0, 1.0],
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
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        embed_dims=embed_dims,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=4,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
    )
)
