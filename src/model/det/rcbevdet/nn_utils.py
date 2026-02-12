"""Local NN utilities replacing minimal mmcv/mmdet runtime deps."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, cast

import torch
import torch.nn as nn


def build_norm_layer(norm_cfg: Optional[Dict], num_features: int, postfix=0) -> Tuple[str, nn.Module]:
    cfg = dict(type="BN") if norm_cfg is None else dict(norm_cfg)
    norm_type = str(cfg.get("type", "BN"))
    eps = float(cfg.get("eps", 1e-5))
    momentum = float(cfg.get("momentum", 0.1))

    if norm_type in ("BN", "BN2d", "SyncBN"):
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        name = f"bn{postfix}"
    elif norm_type == "BN1d":
        layer = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        name = f"bn{postfix}"
    elif norm_type == "BN3d":
        layer = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)
        name = f"bn{postfix}"
    elif norm_type in ("LN", "LayerNorm"):
        layer = nn.LayerNorm(num_features, eps=eps)
        name = f"ln{postfix}"
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    return name, layer


def _build_act(act_cfg: Optional[Dict]) -> nn.Module:
    if act_cfg is None:
        return nn.Identity()
    act_type = act_cfg.get("type", "ReLU")
    inplace = act_cfg.get("inplace", True)
    if act_type == "ReLU":
        return nn.ReLU(inplace=inplace)
    if act_type == "GELU":
        return nn.GELU()
    raise ValueError(f"Unsupported activation type: {act_type}")


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU", inplace=True),
        inplace: bool = True,
    ) -> None:
        super().__init__()
        _ = inplace
        conv_type = (conv_cfg or {}).get("type", "Conv2d")
        if conv_type == "Conv2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        elif conv_type == "Conv3d":
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            raise ValueError(f"Unsupported conv type: {conv_type}")

        self.norm = None
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = _build_act(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_cfg=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        return out


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0) -> None:
    """Rough replacement for mmcv.cnn.normal_init."""
    if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
        nn.init.normal_(cast(torch.Tensor, module.weight), mean=mean, std=std)
    else:
        for child in module.children():
            normal_init(child, mean=mean, std=std, bias=bias)
    if hasattr(module, "bias") and isinstance(getattr(module, "bias"), torch.Tensor):
        nn.init.constant_(cast(torch.Tensor, module.bias), bias)


F = TypeVar("F", bound=Callable[..., Any])


def force_fp32(*dargs, **dkwargs):
    """No-op replacement for mmcv.runner.force_fp32 decorator."""
    _ = dkwargs

    if len(dargs) == 1 and callable(dargs[0]):
        return dargs[0]

    def deco(func: F) -> F:
        return func

    return deco
