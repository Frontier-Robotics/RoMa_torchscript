#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# --- paths to weights / compiled TS file
DINO_W = "/shared/dinov2_vitl14_pretrain.pth"
ROMA_W = "/shared/roma_outdoor.pth"
TS_PT  = "/shared/roma_outdoor_ts.pt"   # produced by your script

# --- model config (coarse MUST be multiples of 14 for ViT-L/14)
COARSE_HW   = (560, 560)        # (H,W) — e.g., 14 * 40
UPSAMPLE_HW = (768, 1024)       # any size (upsample pass doesn’t touch ViT)

# --- image input (use your own file paths if you want)
IMG_A = "left_1748531004_177842904.jpg"  # e.g. "/path/to/a.jpg"
IMG_B = "right_1748531004_177837848.jpg"

# --- tolerances (be a bit loose due to small FP diffs)
ABS_TOL = 5e-5
REL_TOL = 5e-5


def load_sd(p):
    try:    sd = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
            sd = torch.load(p, map_location="cpu")
    # unwrap “state_dict” containers
    return sd["state_dict"] if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict) else sd


def to_coarse_normalized_tensors(imgA, imgB, device, coarse_hw) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reuse the same transform as romatch to avoid preprocessing drift.
    """
    from romatch_ts.utils import get_tuple_transform_ops
    Hc, Wc = coarse_hw
    tfm = get_tuple_transform_ops(resize=(Hc, Wc), normalize=True, clahe=False)
    A, B = tfm((imgA, imgB))
    return A[None].to(device), B[None].to(device)


def max_err(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a - b).abs()
    abs_err = float(diff.max().item())
    rel_err = float((diff / (b.abs() + 1e-12)).max().item())
    return abs_err, rel_err


def main():
    # ------------------------------------------------------------------
    # 0) Device + determinism
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # ------------------------------------------------------------------
    # 1) Build eager model (romatch_ts)
    # ------------------------------------------------------------------
    from romatch_ts.models.model_zoo import roma_outdoor

    dinov2_sd = load_sd(DINO_W)
    roma_sd   = load_sd(ROMA_W)

    eager = roma_outdoor(
        device=device,
        weights=roma_sd,
        dinov2_weights=dinov2_sd,
        coarse_res=COARSE_HW,
        upsample_res=UPSAMPLE_HW
    ).eval().to(device)

    # ------------------------------------------------------------------
    # 2) Load TorchScript module
    # ------------------------------------------------------------------
    assert Path(TS_PT).exists(), f"Missing TorchScript file: {TS_PT}"
    scripted = torch.jit.load(TS_PT, map_location=device).eval().to(device)

    # ------------------------------------------------------------------
    # 3) Prepare inputs (either from files or random)
    # ------------------------------------------------------------------
    if IMG_A is None or IMG_B is None:
        # synthetic RGB, uint8-like range, but we’ll normalize anyway
        import PIL.Image as Image
        import numpy as np
        W_in, H_in = 1024, 768
        A_np = (np.random.rand(H_in, W_in, 3) * 255).astype("uint8")
        B_np = (np.random.rand(H_in, W_in, 3) * 255).astype("uint8")
        imgA = Image.fromarray(A_np, mode="RGB")
        imgB = Image.fromarray(B_np, mode="RGB")
    else:
        from PIL import Image
        imgA = Image.open(IMG_A).convert("RGB")
        imgB = Image.open(IMG_B).convert("RGB")

    imA_c, imB_c = to_coarse_normalized_tensors(imgA, imgB, device, COARSE_HW)

    # ------------------------------------------------------------------
    # 4) Run eager (romatch_ts.forward_ts) and scripted
    # ------------------------------------------------------------------
    with torch.inference_mode():
        warp_eager, cert_eager = eager.forward_ts(imA_c, imB_c, upsample=True)
        warp_ts,    cert_ts    = scripted.forward_ts(imA_c, imB_c, upsample=True)

    # ------------------------------------------------------------------
    # 5) Compare numerics
    # ------------------------------------------------------------------
    # Shapes should match exactly
    assert warp_eager.shape == warp_ts.shape, f"warp shape mismatch: {warp_eager.shape} vs {warp_ts.shape}"
    assert cert_eager.shape == cert_ts.shape, f"cert shape mismatch: {cert_eager.shape} vs {cert_ts.shape}"

    w_abs, w_rel = max_err(warp_eager, warp_ts)
    c_abs, c_rel = max_err(cert_eager, cert_ts)

    print(f"[warp] max abs: {w_abs:.3e}   max rel: {w_rel:.3e}")
    print(f"[cert] max abs: {c_abs:.3e}   max rel: {c_rel:.3e}")

    ok = (w_abs <= ABS_TOL or w_rel <= REL_TOL) and (c_abs <= ABS_TOL or c_rel <= REL_TOL)
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")

    # ------------------------------------------------------------------
    # 6) (Optional) Compare against the *official* romatch eager .match
    # ------------------------------------------------------------------
    try:
        from romatch.models.model_zoo import roma_outdoor as roma_outdoor_official
        official = roma_outdoor_official(
            device=device,
            weights=roma_sd,
            dinov2_weights=dinov2_sd,
            coarse_res=COARSE_HW,
            upsample_res=UPSAMPLE_HW,
        ).eval().to(device)

        # keep behavior aligned with forward_ts
        if hasattr(official, "attenuate_cert"): official.attenuate_cert = False
        if hasattr(official, "symmetric"):      official.symmetric = False

        with torch.inference_mode():
            # .match handles its own transforms/resize internally
            warp_off, cert_off = official.match(imgA, imgB, batched=False, device=device)

        # align shapes for comparison: add batch/channel dims
        warp_off = warp_off[None].to(warp_ts.device)              # [1,H,W,4]
        cert_off = cert_off[None,None].to(cert_ts.device)         # [1,1,H,W]

        w_abs2, w_rel2 = max_err(warp_off, warp_ts)
        c_abs2, c_rel2 = max_err(cert_off, cert_ts)

        print(f"[official vs TS] warp abs: {w_abs2:.3e}  rel: {w_rel2:.3e}")
        print(f"[official vs TS] cert abs: {c_abs2:.3e}  rel: {c_rel2:.3e}")

    except Exception as e:
        print("(optional official check skipped:", e, ")")

if __name__ == "__main__":
    main()

