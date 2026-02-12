"""Local gaussian heatmap utilities used by RCBEVDet."""

import numpy as np
import torch


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    h = np.exp(-1.0 * (x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device, torch.float32)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap_gaussian_feat(heatmap, center, radius, feat, k=1):
    _ = k
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[-2:]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    target = heatmap[:, y - top:y + bottom, x - left:x + right]
    heatmap[:, y - top:y + bottom, x - left:x + right] = feat.view(-1, 1, 1).expand_as(target)
    return heatmap
