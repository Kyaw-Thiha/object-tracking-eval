"""RCBEVDet inference config (3D detection)."""

_base_ = ["./_base_rcbevdet_r50_nuscenes.py"]

# Inference-focused runtime defaults.
data = dict(samples_per_gpu=1, workers_per_gpu=2)

# Not used directly by factory unless caller passes a checkpoint, but keeping
# this field allows direct MM-style inference scripts.
load_from = None
