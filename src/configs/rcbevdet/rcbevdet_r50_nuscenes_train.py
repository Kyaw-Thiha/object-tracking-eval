"""RCBEVDet training config scaffold for Phase-3 training pipeline wiring."""

_base_ = ["./_base_rcbevdet_r50_nuscenes.py"]

# Training-focused overrides.
data = dict(samples_per_gpu=1, workers_per_gpu=4)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)

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
