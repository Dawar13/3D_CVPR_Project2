"""Minimal stand-in for detectron2.layers.{ModulatedDeformConv, DeformConv} (JAMNet only uses these two).

Parameter names/shapes match detectron2 (weight (O, I/groups, k, k), bias (O) or None), so the released
state_dicts load unchanged. The op is torchvision.ops.deform_conv2d, which uses the same offset layout as the
original DCN CUDA kernels used by detectron2 (per deformable group: (dy, dx) pairs over the k*k taps, row-major)
and the same mask semantics (DCNv2). Verified end-to-end by reproducing the JAMNet demo PSNR (see ENV.md).
"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class _DCNBase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 deformable_groups=1, bias=True, norm=None, activation=None):
        super().__init__()
        assert norm is None and activation is None
        self.stride, self.padding, self.dilation = _pair(stride), _pair(padding), _pair(dilation)
        k = _pair(kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *k))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class ModulatedDeformConv(_DCNBase):
    def forward(self, x, offset, mask):
        return deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, mask=mask)


class DeformConv(_DCNBase):
    def __init__(self, *a, bias=False, **k):
        super().__init__(*a, bias=bias, **k)

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation)
