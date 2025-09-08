import math
import torch
import torch.nn.functional as F
from typing import Optional

@torch.jit.script
def local_correlation(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    local_radius: int,                    # <- force int
    padding_mode: str = "zeros",
    flow: Optional[torch.Tensor] = None,
    sample_mode: str = "bilinear",
) -> torch.Tensor:
    r: int = int(local_radius)
    k_side: int = 2 * r + 1
    K: int = k_side * k_side

    B = int(feature0.shape[0])
    c = int(feature0.shape[1])
    h = int(feature0.shape[2])
    w = int(feature0.shape[3])

    # allocate with int sizes
    corr = torch.empty((B, K, h, w), device=feature0.device, dtype=feature0.dtype)

    # base coords (H,W,2), expanded to (B,H,W,2) if not using flow
    if flow is None:
        ys = torch.linspace(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h), h, device=feature0.device)
        xs = torch.linspace(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w), w, device=feature0.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, h, w, 2)
    else:
        # flow: (B,2,H,W) -> (B,H,W,2)
        coords = flow.permute(0, 2, 3, 1)

    # local window offsets: (1, K, 2)
    win_ys = torch.linspace(-2.0 * float(r) / float(h), 2.0 * float(r) / float(h), steps=k_side, device=feature0.device)
    win_xs = torch.linspace(-2.0 * float(r) / float(w), 2.0 * float(r) / float(w), steps=k_side, device=feature0.device)
    wgy, wgx = torch.meshgrid(win_ys, win_xs, indexing='ij')
    local_window = torch.stack((wgx, wgy), dim=-1).reshape(1, K, 2)  # (1, K, 2)

    inv_sqrt_c = 1.0 / math.sqrt(float(c))

    for b in range(B):
        # (H,W,2) + (1,K,2) -> (H,W,K,2) -> (1,H,W*K,2)
        lwc = (coords[b].unsqueeze(2) + local_window.unsqueeze(0)).reshape(1, h, w * K, 2)

        # sample (1,C,H,W*K)
        sampled = F.grid_sample(
            feature1[b:b+1], lwc, padding_mode=padding_mode, align_corners=False, mode=sample_mode
        ).reshape(c, h, w, K)  # (C,H,W,K)

        # corr: (H,W,K)
        corr_b = (feature0[b].unsqueeze(-1) * inv_sqrt_c * sampled).sum(dim=0).permute(2, 0, 1)  # (K,H,W)
        corr[b] = corr_b

    return corr
