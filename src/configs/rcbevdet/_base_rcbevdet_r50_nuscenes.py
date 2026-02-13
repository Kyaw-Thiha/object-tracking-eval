"""Base RCBEVDet config shared by train and inference."""

custom_imports = dict(
    imports=[
        "model.det.rcbevdet",
    ],
    allow_failed_imports=False,
)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.8, 0.8, 8.0]
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

radar_feature_dim = 7
num_adj_frames = 1
with_prev_frames = True
img_bev_channels = 80
temporal_bev_frames = num_adj_frames + 1 if with_prev_frames else 1

model = dict(
    type="BEVDepth4D_RC",
    num_adj=num_adj_frames,
    with_prev=with_prev_frames,
    align_after_view_transfromation=False,
    freeze_img=False,
    freeze_radar=False,
    bev_size=128,
    imc=256,
    rac=64,
    # Camera image branch
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,
        start_level=2,
        num_outs=2,
    ),
    img_view_transformer=dict(
        type="LSSViewTransformerBEVDepth",
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-5.0, 3.0, 8.0],
            depth=[1.0, 60.0, 1.0],
        ),
        input_size=(256, 704),
        downsample=16,
        in_channels=512,
        out_channels=img_bev_channels,
        accelerate=False,
        sid=False,
        collapse_z=True,
        loss_depth_weight=3.0,
        depthnet_cfg=dict(use_dcn=False),
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet",
        numC_input=img_bev_channels * temporal_bev_frames,
        num_layer=[2, 2, 2],
        num_channels=[160, 320, 640],
        stride=[2, 2, 2],
    ),
    img_bev_encoder_neck=dict(
        type="FPN_LSS",
        in_channels=800,
        out_channels=256,
        scale_factor=4,
        input_feature_index=(0, 2),
        extra_upsample=2,
    ),
    # Radar branch
    radar_voxel_layer=dict(
        max_num_points=12,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000),
    ),
    radar_voxel_encoder=dict(
        type="RadarBEVNet",
        in_channels=radar_feature_dim,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        return_rcs=True,
    ),
    radar_middle_encoder=dict(
        type="PointPillarsScatterRCS",
        in_channels=64,
        output_shape=[128, 128],
    ),
    radar_bev_backbone=dict(
        type="SECOND",
        in_channels=64,
        out_channels=[64],
        layer_nums=[3],
        layer_strides=[1],
    ),
    radar_bev_neck=dict(
        type="SECONDFPN",
        in_channels=[64],
        out_channels=[64],
        upsample_strides=[1],
    ),
    # 3D detection head
    pts_bbox_head=dict(
        type="CenterHead",
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2),
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=9,
        ),
        separate_head=dict(
            type="SeparateHead",
            init_bias=-2.19,
            final_kernel=3,
        ),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[128, 128, 1],
            voxel_size=voxel_size,
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        test_cfg=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type="rotate",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        ),
    ),
)

train_cfg = dict()
test_cfg = dict()

classes = tuple(class_names)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],
)

runner = dict(type="EpochBasedRunner", max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

dist_params = dict(backend="nccl")
log_level = "INFO"
workflow = [("train", 1)]
