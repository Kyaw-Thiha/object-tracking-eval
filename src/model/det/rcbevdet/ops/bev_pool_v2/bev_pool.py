# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch
from typing import Optional

try:
    from . import bev_pool_v2_ext
    USE_CUDA = True
except ImportError:  # pragma: no cover - depends on local CUDA extension build.
    bev_pool_v2_ext = None
    USE_CUDA = False
    import warnings
    warnings.warn(
        "bev_pool_v2_ext CUDA extension not available. "
        "Using PyTorch fallback (slower but no compilation needed). "
        "To use CUDA: compile with 'python setup.py build_ext --inplace'"
    )

__all__ = ['bev_pool_v2', 'TRTBEVPoolv2']

DEFAULT_BEV_POOL_BACKEND = "auto"  # "auto" | "cuda_ext" | "torch"


def set_default_bev_pool_backend(backend: str) -> None:
    backend = backend.lower()
    if backend not in {"auto", "cuda_ext", "torch"}:
        raise ValueError(f"Invalid bev_pool backend: {backend}")
    global DEFAULT_BEV_POOL_BACKEND
    DEFAULT_BEV_POOL_BACKEND = backend


def _resolve_backend(backend: Optional[str]) -> str:
    selected = (backend or DEFAULT_BEV_POOL_BACKEND).lower()
    if selected not in {"auto", "cuda_ext", "torch"}:
        raise ValueError(f"Invalid bev_pool backend: {selected}")
    if selected == "auto":
        return "cuda_ext" if bev_pool_v2_ext is not None else "torch"
    if selected == "cuda_ext" and bev_pool_v2_ext is None:
        raise RuntimeError("bev_pool backend='cuda_ext' requested but CUDA extension is unavailable.")
    return selected


def _bev_pool_v2_torch_forward(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch fallback matching BEVPoolV2 math in CUDA kernel.
    """
    if depth.ndim != 5:
        raise ValueError(f"Expected depth shape [B,N,D,H,W], got {tuple(depth.shape)}")
    if feat.ndim != 5:
        raise ValueError(f"Expected feat shape [B,N,H,W,C], got {tuple(feat.shape)}")

    depth = depth.contiguous().float()
    feat = feat.contiguous().float()
    ranks_depth = ranks_depth.contiguous().long()
    ranks_feat = ranks_feat.contiguous().long()
    ranks_bev = ranks_bev.contiguous().long()
    interval_starts = interval_starts.contiguous().long()
    interval_lengths = interval_lengths.contiguous().long()

    out = feat.new_zeros(bev_feat_shape)
    out_flat = out.view(-1, out.shape[-1])    # [B*Z*Y*X, C]
    depth_flat = depth.view(-1)               # [B*N*D*H*W]
    feat_flat = feat.view(-1, feat.shape[-1]) # [B*N*H*W, C]

    pooled_vals = []
    pooled_bev_idx = []
    n_intervals = int(interval_starts.numel())
    for i in range(n_intervals):
        start = int(interval_starts[i].item())
        length = int(interval_lengths[i].item())
        if length <= 0:
            continue
        end = start + length
        rd = ranks_depth[start:end]
        rf = ranks_feat[start:end]
        bev_idx = int(ranks_bev[start].item())

        w = depth_flat.index_select(0, rd).unsqueeze(1)   # [L,1]
        f = feat_flat.index_select(0, rf)                 # [L,C]
        pooled = (w * f).sum(dim=0)                       # [C]
        pooled_vals.append(pooled)
        pooled_bev_idx.append(bev_idx)

    if pooled_vals:
        src = torch.stack(pooled_vals, dim=0)  # [K,C]
        idx = torch.tensor(pooled_bev_idx, dtype=torch.long, device=out.device)
        out_flat.index_add_(0, idx, src)

    return out


class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, \
            None, None, None


def bev_pool_v2(
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    bev_feat_shape,
    interval_starts,
    interval_lengths,
    backend: Optional[str] = None,
):
    selected = _resolve_backend(backend)
    if selected == "cuda_ext":
        x = QuickCumsumCuda.apply(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )
    else:
        x = _bev_pool_v2_torch_forward(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
        )
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


class TRTBEVPoolv2(torch.autograd.Function):

    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 interval_starts,
                 interval_lengths,
                 out_height=128,
                 out_width=128):
        """symbolic function for creating onnx op."""
        return g.op(
            'mmdeploy::bev_pool_v2',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out_height_i=out_height,
            out_width_i=out_width)

    @staticmethod
    def forward(g,
                depth,  # N,D,H,W
                feat,  # N,H,W,C
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                out_height=128,
                out_width=128):
        """run forward."""
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        bev_feat_shape = (depth.shape[0], 1, out_height, out_width,
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        bev_feat = bev_feat.squeeze(2)
        bev_feat = bev_feat.permute(0, 2, 3, 1)
        return bev_feat


def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float().cuda()
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
    feat = torch.ones(
        size=[1, 1, 2, 2, 2], dtype=torch.float,
        device='cuda').requires_grad_()
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().cuda()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().cuda()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().cuda()

    kept = torch.ones(
        ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                           (1, 1, 2, 2, 2), interval_starts, interval_lengths)
    loss = torch.sum(bev_feat)
    loss.backward()
    assert loss == 4.4
    grad_depth = np.array([2., 2., 0., 0., 2., 0., 2., 0.])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.cuda().view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0., 0.])
    grad_feat = torch.from_numpy(grad_feat).float().cuda().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
