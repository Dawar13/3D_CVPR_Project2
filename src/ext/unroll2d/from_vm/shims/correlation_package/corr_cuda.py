"""Pure-PyTorch replacement for the FlowNet/PWC-Net `correlation_package` CUDA op (no nvcc on the VM).

Semantics copied from correlation_package_lib/corr_cuda_kernel.cu (DeepUnrollNet / BSRSC / RSCD, identical files):
  - inputs zero-padded by pad_size; output channel c <-> displacement
      dx = (c % D - r) * stride2,  dy = (c // D - r) * stride2,  D = 2r + 1, r = max_displacement // stride2
  - value = sum_{kernel window, channels} f1 * f2(shifted) / (kernel_size^2 * C)
Only the configuration used by all our methods is implemented: kernel_size=1, stride1=1, pad_size=max_displacement.
Computed in float32 (the CUDA op is float32-only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def correlation(f1, f2, max_displacement, stride2=1):
    dt = f1.dtype
    f1 = f1.float()
    f2 = f2.float()
    B, C, H, W = f1.shape
    md = max_displacement
    r = md // stride2
    f2p = F.pad(f2, (md, md, md, md))
    out = []
    for iy in range(-r, r + 1):
        for ix in range(-r, r + 1):
            dy, dx = iy * stride2, ix * stride2
            out.append((f1 * f2p[:, :, md + dy:md + dy + H, md + dx:md + dx + W]).sum(1) / C)
    return torch.stack(out, 1).to(dt)


class Correlation(nn.Module):
    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        super().__init__()
        assert kernel_size == 1 and stride1 == 1 and pad_size == max_displacement, \
            (pad_size, kernel_size, max_displacement, stride1, stride2)
        self.md, self.stride2 = max_displacement, stride2

    def forward(self, input1, input2):
        return correlation(input1, input2, self.md, self.stride2)
