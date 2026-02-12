# RCBEVDet Framework Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT MODALITIES                            │
└─────────────────────────────────────────────────────────────────────┘
                    │                             │
         ┌──────────▼──────────┐      ┌───────────▼──────────┐
         │  Multi-View Images  │      │   Radar Point Cloud  │
         └──────────┬──────────┘      └───────────┬──────────┘
                    │                             │
┌───────────────────▼──────────────┐  ┌───────────▼──────────────────┐
│     CAMERA BRANCH                │  │      RADAR BRANCH            │
├──────────────────────────────────┤  ├──────────────────────────────┤
│ 1. Image Backbone                │  │ 6. Radar Voxelization        │
│    ↓                             │  │    ↓                         │
│ 2. Image Neck (FPN)              │  │ 7. Radar Voxel Encoder       │
│    ↓                             │  │    (RadarBEVNet)             │
│ 3. View Transformer              │  │    ↓                         │
│    (LSS/BEVDepth)                │  │ 8. Radar Middle Encoder      │
│    ↓                             │  │    (BEV Scattering)          │
│ 4. Image BEV Encoder Backbone    │  │    ↓                         │
│    ↓                             │  │ 9. Radar BEV Backbone        │
│ 5. Image BEV Encoder Neck        │  │    ↓                         │
│    ↓                             │  │ 10. Radar BEV Neck           │
│    ↓                             │  │    ↓                         │
│  Camera BEV Feature (256×H×W)    │  │  Radar BEV Feature (64×H×W)  │
└──────────────────┬───────────────┘  └───────────┬──────────────────┘
                   │                              │
                   └──────────┬───────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  FUSION MODULE      │
                   │  (CAMF)             │
                   ├─────────────────────┤
                   │ • Deformable Cross  │
                   │   Attention         │
                   │ • Channel & Spatial │
                   │   Fusion            │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Fused BEV Feature  │
                   │  (256×H×W)          │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  DETECTION HEAD     │
                   │  (CenterHead, etc.) │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  3D Bounding Boxes  │
                   │  + Velocities       │
                   └─────────────────────┘
```

## Components & Available Models

| Component # | Component Name | Purpose | Available Models |
|------------|----------------|---------|------------------|
| **1** | **Image Backbone** | Extract multi-view image features | • ResNet (18, 50, 101)<br>• DLA-34<br>• Swin Transformer<br>• ConvNeXt<br>• VoVNet (V2-99)<br>• ViT-L<br>• HRNet |
| **2** | **Image Neck** | Multi-scale feature fusion | • CustomFPN<br>• FPN_LSS<br>• SECONDFPN<br>• DLANeck<br>• CPFPN |
| **3** | **View Transformer** | Perspective → BEV transformation | • LSSViewTransformer<br>• LSSViewTransformerBEVDepth ⭐<br>• LSSViewTransformerBEVStereo |
| **4** | **Image BEV Encoder Backbone** | Process camera BEV features | • CustomResNet<br>• SECOND |
| **5** | **Image BEV Encoder Neck** | Refine camera BEV features | • FPN_LSS<br>• SECONDFPN |
| **6-7** | **Radar Voxel Encoder** | Extract radar point features | • **RadarBEVNet** ⭐ (Proposed)<br>  - Dual-stream (Point + Transformer)<br>  - DMSA attention<br>  - Injection/Extraction modules<br>• PointPillar (Baseline) |
| **8** | **Radar Middle Encoder** | Scatter to BEV space | • PointPillarsScatter<br>• **PointPillarsScatterRCS** ⭐<br>• SparseEncoder |
| **9** | **Radar BEV Backbone** | Process radar BEV features | • SECOND |
| **10** | **Radar BEV Neck** | Refine radar BEV features | • SECONDFPN<br>• CustomSECONDFPN |
| **11** | **Fusion Module (CAMF)** | Align & fuse multi-modal features | • **MSDeformAttn** (Cross-attention) ⭐<br>• **RadarConvFuser** (Conv fusion) ⭐ |
| **12** | **Detection Head** | Generate predictions | • CenterHead (3D Detection)<br>• TransFusionHead<br>• BEVSegHead (Segmentation) |

⭐ = Novel components proposed by RCBEVDet

## RadarBEVNet Architecture

```
Radar Points (N × 7)  [x, y, z, vx, vy, RCS, ...]
         │
         ├────────────────┬────────────────┐
         │                │                │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Voxelize│      │ Voxelize│      │ Voxelize│
    └────┬────┘      └────┬────┘      └─────┬───┘
         │                │                 │
  ┌──────▼──────┐   ┌─────▼──────┐          │
  │ Point-based │   │ Transformer│          │
  │  Backbone   │◄─►│  Backbone  │          │
  │             │   │            │          │
  │ • PointNet  │   │ • DMSA     │          │
  │ • MLP+Pool  │   │ • FFN      │          │
  └──────┬──────┘   └─────┬──────┘          │
         │                │                 │
         │   ┌────────────┴─────────┐       │
         │   │ Injection/Extraction │       │
         │   │   Cross-Attention    │       │
         │   └──────────────────────┘       │
         │                 │                │
         └────────┬────────┘                │
                  │                         │
           ┌──────▼────────┐                │
           │ Fused Features│                │
           └──────┬────────┘                │
                  │                         │
           ┌──────▼──────────────┬──────────▼────────┐
           │  RCS-aware Scattering               RCS │
           │  (Spread features based on RCS)         │
           └──────┬──────────────────────────────────┘
                  │
           ┌──────▼──────┐
           │  BEV Feature│
           │  (64×H×W)   │
           └─────────────┘
```

## CAMF Fusion Module

```
Camera BEV (256×128×128)          Radar BEV (256×128×128)
         │                                  │
         │                                  │
    ┌────▼────┐                        ┌────▼───┐
    │ + Pos   │                        │ + Pos  │
    │ Encoding│                        │Encoding│
    └────┬────┘                        └────┬───┘
         │                                  │
         │    ┌─────────────────────┐       │
         └───►│  Deformable Cross   │◄──────┘
              │     Attention       │
              │                     │
              │ • Query: Radar      │
              │ • Key/Value: Camera │
              │ • 8 heads, 8 points │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Deformable Cross   │
              │     Attention       │
              │                     │
              │ • Query: Camera     │
              │ • Key/Value: Radar  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Concatenate       │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  RadarConvFuser     │
              │  • Conv3×3 + BN     │
              │  • 3× Deconv blocks │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ Fused BEV Feature   │
              │    (256×128×128)    │
              └─────────────────────┘
```

## Detector Variants

| Detector Model | Base Architecture | Key Features |
|----------------|------------------|--------------|
| **BEVDet_RC** | BEVDet + Radar | Basic radar-camera fusion |
| **BEVDet4D_RC** | BEVDet4D + Radar | + Temporal modeling (multi-frame) |
| **BEVDepth4D_RC** ⭐ | BEVDepth + Radar | + Depth supervision + Temporal |
| **BEVStereo4D_RC** | BEVStereo + Radar | + Stereo depth + Temporal |

⭐ = Recommended (best performance)

## Example Configuration

| Component | Selected Model | Details |
|-----------|---------------|---------|
| Image Backbone | ResNet-50 | Pretrained on ImageNet |
| Image Size | 256×704 | Input resolution |
| View Transformer | LSSViewTransformerBEVDepth | With explicit depth |
| BEV Size | 128×128 | Spatial resolution |
| Radar Encoder | RadarBEVNet | 64-dim features, 3 stages |
| Radar Sweeps | 9 frames | 8 past + 1 current |
| Fusion | CAMF | 8 heads, 8 sampling points |
| Detection Head | CenterHead | CenterPoint-style |
| Total Epochs | 12 | With CBGS sampling |
