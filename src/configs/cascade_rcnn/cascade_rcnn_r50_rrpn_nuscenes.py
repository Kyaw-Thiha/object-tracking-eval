"""Two-stage detector + RRPN wrapper config for NuScenes."""
_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
]

model = dict(
    detector=dict(
        roi_head=dict(
            bbox_head=dict(num_classes=10),
        )
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

classes = ('car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle',
           'construction_vehicle', 'trailer', 'movable_object', 'traffic_cone')
