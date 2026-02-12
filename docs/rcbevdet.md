# RCBEVDet Component Selection

## Recommended Configuration

| # | Component | Selected Model |
|---|-----------|----------------|
| **1** | Image Backbone | **ResNet-50** |
| **2** | Image Neck | **CustomFPN** |
| **3** | View Transformer | **LSSViewTransformerBEVDepth** |
| **4** | Image BEV Encoder Backbone | **CustomResNet** |
| **5** | Image BEV Encoder Neck | **FPN_LSS** |
| **6-7** | Radar Voxel Encoder | **RadarBEVNet** |
| **8** | Radar Middle Encoder | **PointPillarsScatterRCS** |
| **9** | Radar BEV Backbone | **SECOND** |
| **10** | Radar BEV Neck | **SECONDFPN** |
| **11** | Fusion Module | **CAMF** (MSDeformAttn + RadarConvFuser) |
| **12** | Detection Head | **CenterHead** |
| **-** | Complete Detector | **BEVDepth4D_RC** |

## Configuration Parameters

| Parameter | Value |
|-----------|-------|
| Image Size | 256 × 704 |
| BEV Size | 128 × 128 |
| Radar Sweeps | 9 (8 past + 1 current) |
| Radar Features | 64-dim |
| Camera Features | 256-dim |
| Fusion Heads | 8 |
| Fusion Points | 8 |
| Training Epochs | 12 |

## Lightweight Alternative

| Component | Alternative Model |
|-----------|------------------|
| Image Backbone | ResNet-18 or DLA-34 |
| BEV Size | 64 × 64 |
| Radar Sweeps | 4 frames |

## Compile Ops

Run from repo root:

```bash
python src/model/det/rcbevdet/ops/deformattn/setup.py build_ext --inplace && \
python src/model/det/rcbevdet/ops/bev_pool_v2/setup.py build_ext --inplace
```
