#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F

# --- Adjust these paths if yours differ ---
DINO_W = "/shared/dinov2_vitl14_pretrain.pth"
ROMA_W = "/shared/roma_outdoor.pth"
OUT_TS = "/shared/roma_outdoor_ts.pt"

# Import your implementation
# from romatch import roma_outdoor   # if your function is at romatch/__init__.py
from romatch import roma_outdoor  # <- change to where roma_outdoor(...) is defined

def load_state_dict(path):
    sd = torch.load(path, map_location="cpu")
    # handle { 'state_dict': ... } style checkpoints
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd

def main():
    assert os.path.isfile(DINO_W), f"Missing DINO weights at {DINO_W}"
    assert os.path.isfile(ROMA_W), f"Missing RoMa weights at {ROMA_W}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov2_weights = load_state_dict(DINO_W)
    roma_weights   = load_state_dict(ROMA_W)

    # Build RoMa with preloaded weights (no downloads inside)
    # Keep 560×560 to stay friendly to an 8 GB 4060 (batch=1).
    model = roma_outdoor(
        device=device,
        weights=roma_weights,
        dinov2_weights=dinov2_weights,
        coarse_res=560,         # multiple of 14
        upsample_res=560,       # can set 560 or 864 — 560 is safer for 8 GB
        amp_dtype=torch.float16 # will be ignored on CPU
    )
    model.eval().to(device)

    # Dummy normalized inputs [B,3,H,W] for scripting sanity-check
    B, C, H, W = 1, 3, 560, 560
    xA = torch.randn(B, C, H, W, device=device)
    xB = torch.randn(B, C, H, W, device=device)

    # ---- Sanity forward through TorchScript-safe entrypoint ----
    # You must have implemented RegressionMatcher.forward_ts and Decoder.forward_ts
    with torch.inference_mode():
        warp, cert = model.forward_ts(xA, xB, upsample=True)  # returns [B,H,W,4], [B,1,H,W] or [B,H,W]
        # Optional quick check to prevent size surprises
        assert warp.shape[0] == B and warp.shape[1] == H and warp.shape[2] == W and warp.shape[3] == 4

    # ---- Script & save ----
    scripted = torch.jit.script(model)
    scripted.save(OUT_TS)
    print(f"Saved TorchScript module to {OUT_TS}")

if __name__ == "__main__":
    main()
