"""Pure-PyTorch replacement for DeepUnrollNet's `forward_warp_package` (CUDA forward splatting; no nvcc on the VM).

Semantics copied from package_forward_warp/forward_warp_package_lib/{flow_forward_shift.cpp, cuda_renderer.cu}:
  target position p = pixel grid (integer x, y) + flow_src_to_tar
  each source pixel splats onto integer pixels q in [int(p - r + 0.5), int(p + r + 0.5)] (clamped to the image),
  weight w = exp(-0.5 * |q - p|^2 / sigma2)
  output = sum(w * I) / (sum(w) + 1e-8),  mask = 1 where any splat landed.
Difference to the CUDA kernel: the window start uses floor() instead of C int truncation. The two differ only for
target positions left of / above the image (p < r - 0.5), where the extra taps lie >= r + 0.5 px away
(weight <= exp(-6.25) for r = 2, sigma2 = 0.5) - negligible. Forward pass only (inference).
"""
import torch
import torch.nn as nn


class ForwardWarp(nn.Module):
    def __init__(self, B, C, H, W, kernel_radius=4, kernel_sigma2=0.5):
        super().__init__()
        self.B, self.C, self.H, self.W = B, C, H, W
        self.r, self.s2 = int(kernel_radius), float(kernel_sigma2)

    @classmethod
    def create_with_implicit_mesh(cls, B, C, H, W, kernel_radius=4, kernel_sigma2=0.5):
        return cls(B, C, H, W, kernel_radius, kernel_sigma2)

    def forward(self, src_image, flow_src_to_tar):
        B, C, H, W = src_image.shape
        assert (B, C, H, W) == (self.B, self.C, self.H, self.W)
        dev = src_image.device
        src = src_image.float().reshape(B, C, H * W)
        flow = flow_src_to_tar.float()
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        px = (xx[None] + flow[:, 0]).reshape(B, -1)  # (B, HW) target x
        py = (yy[None] + flow[:, 1]).reshape(B, -1)
        bx = torch.floor(px - self.r + 0.5)
        by = torch.floor(py - self.r + 0.5)
        acc = torch.zeros(B, C, H * W, device=dev)
        wsum = torch.zeros(B, 1, H * W, device=dev)
        n = 2 * self.r + 1
        for oy in range(n):
            qy = by + oy
            for ox in range(n):
                qx = bx + ox
                ok = (qx >= 0) & (qx <= W - 1) & (qy >= 0) & (qy <= H - 1) & (qx <= px + self.r + 0.5) & \
                     (qy <= py + self.r + 0.5)
                w = torch.exp(-0.5 * ((qx - px) ** 2 + (qy - py) ** 2) / self.s2) * ok
                idx = (qy.clamp(0, H - 1) * W + qx.clamp(0, W - 1)).long()
                wsum.scatter_add_(2, idx[:, None], w[:, None])
                acc.scatter_add_(2, idx[:, None].expand(B, C, -1), (w[:, None] * src))
        hit = torch.zeros(B, 1, H * W, device=dev)
        # mask = 1 wherever any in-window tap landed (even with tiny weight), as in the CUDA kernel
        for oy in range(n):
            qy = by + oy
            for ox in range(n):
                qx = bx + ox
                ok = (qx >= 0) & (qx <= W - 1) & (qy >= 0) & (qy <= H - 1) & (qx <= px + self.r + 0.5) & \
                     (qy <= py + self.r + 0.5)
                idx = (qy.clamp(0, H - 1) * W + qx.clamp(0, W - 1)).long()
                hit.scatter_add_(2, idx[:, None], ok[:, None].float())
        out = (acc / (wsum + 1e-8)).reshape(B, C, H, W).to(src_image.dtype)
        mask = (hit > 0).float().reshape(B, 1, H, W)
        return out, mask
