"""Probabilistic Cascade R-CNN + RRPN config for NuScenes.

Single end-to-end probabilistic config used for both training and evaluation.
"""

_base_ = ['../_base_/models/cascade_rcnn_r50_fpn.py']

model = dict(
    detector=dict(
        roi_head=dict(
            type='ProbabilisticCascadeRoIHead',
            bbox_head=[
                dict(
                    type='ProbabilisticBBoxHead',
                    num_classes=10,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    with_cov=True,
                    reg_class_agnostic=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    reg_decoded_bbox=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0
                    ),
                    loss_bbox=dict(
                        type='NLL',
                        covariance_type='diagonal',
                        loss_type='L1',
                        loss_weight=1.0
                    )
                ),
                dict(
                    type='ProbabilisticBBoxHead',
                    num_classes=10,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    with_cov=True,
                    reg_class_agnostic=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]
                    ),
                    reg_decoded_bbox=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0
                    ),
                    loss_bbox=dict(
                        type='NLL',
                        covariance_type='diagonal',
                        loss_type='L1',
                        loss_weight=1.0
                    )
                ),
                dict(
                    type='ProbabilisticBBoxHead',
                    num_classes=10,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    with_cov=True,
                    reg_class_agnostic=True,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]
                    ),
                    reg_decoded_bbox=False,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0
                    ),
                    loss_bbox=dict(
                        type='NLL',
                        covariance_type='diagonal',
                        loss_type='L1',
                        loss_weight=1.0
                    )
                ),
            ]
        )
    )
)

classes = (
    'car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle',
    'construction_vehicle', 'trailer', 'movable_object', 'traffic_cone'
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

evaluation = dict(interval=1, metric='bbox')
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
