import torch
import torch.nn.functional as F
from typing import Optional

@torch.jit.script
def local_correlation_ts(
    feature0: torch.Tensor,            # [B, C, H, W]
    feature1: torch.Tensor,            # [B, C, H, W]
    local_radius: int,
    flow: Optional[torch.Tensor] = None  # [B, 2, H, W] in [-1,1] or None
) -> torch.Tensor:
    B, C, H, W = feature0.shape
    r = local_radius
    K = (2 * r + 1) * (2 * r + 1)

    # Base coords in [-1, 1]
    xs = torch.linspace(-1.0 + 1.0 / float(W), 1.0 - 1.0 / float(W), W, device=feature0.device)
    ys = torch.linspace(-1.0 + 1.0 / float(H), 1.0 - 1.0 / float(H), H, device=feature0.device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]
    if flow is not None:
        base = flow.permute(0, 2, 3, 1).contiguous()                      # [B,H,W,2]

    # Local window offsets (normalized)
    off_xs = torch.linspace(-2.0 * float(r) / float(W), 2.0 * float(r) / float(W), 2 * r + 1, device=feature0.device)
    off_ys = torch.linspace(-2.0 * float(r) / float(H), 2.0 * float(r) / float(H), 2 * r + 1, device=feature0.device)
    o_gy, o_gx = torch.meshgrid(off_ys, off_xs, indexing='ij')
    offsets = torch.stack((o_gx, o_gy), dim=-1).reshape(1, 1, 1, K, 2)     # [1,1,1,K,2]

    coords = (base.unsqueeze(3) + offsets).reshape(B, H, W * K, 2)         # [B,H,W*K,2]
    # Optional: coords = coords.clamp(-1.0, 1.0)

    sampled = F.grid_sample(feature1, coords, mode='bilinear', align_corners=False, padding_mode='zeros')
    sampled = sampled.reshape(B, C, H, W, K)                                # [B,C,H,W,K]

    inv_sqrt_C = 1.0 / float(C) ** 0.5
    corr = (feature0.unsqueeze(-1) * sampled).sum(dim=1) * inv_sqrt_C      # [B,H,W,K]
    corr = corr.permute(0, 3, 1, 2).contiguous()                            # [B,K,H,W]
    return corr

# Eager wrapper (keeps your old API). Marked ignore so JIT won’t try to compile it.
from torch.jit import ignore
@ignore
def local_correlation(feature0, feature1, local_radius, padding_mode="zeros", flow=None, sample_mode="bilinear"):
    return local_correlation_ts(feature0, feature1, int(local_radius), flow)
