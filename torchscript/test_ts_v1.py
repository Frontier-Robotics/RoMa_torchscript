#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import torch.nn.functional as Fnn


DINO_W = "/shared/dinov2_vitl14_pretrain.pth"
ROMA_W = "/shared/roma_outdoor.pth"
OUT_TS = "/shared/roma_outdoor_ts.pt"

# optional: TF32 on Ampere+ (faster matmul in fp32)
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

from romatch_ts.models.model_zoo import roma_outdoor


def load_sd(p):
    # Backward-compatible safe load
    try:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(p, map_location="cpu")
    return sd["state_dict"] if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict) else sd


def main():
    Path(OUT_TS).parent.mkdir(parents=True, exist_ok=True)

    dinov2_sd = load_sd(DINO_W)
    roma_sd   = load_sd(ROMA_W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = roma_outdoor(
        device=device,
        weights=roma_sd,
        dinov2_weights=dinov2_sd,
        coarse_res=(560, 560),
        upsample_res=(768, 1024),
        amp_dtype=torch.float16,
    ).eval().to(device)

    # ---- optional: sanity-check eager forward on GPU ----
    with torch.inference_mode():
        xA = torch.randn(1, 3, 768, 1024, device=device)
        xB = torch.randn(1, 3, 768, 1024, device=device)
        warp, cert = model.forward_ts(xA, xB, upsample=True)
        print("warp", tuple(warp.shape), "cert", tuple(cert.shape))

    print("Eager forward passed, TorchScripting...")
    
    # Free VRAM before scripting
    del warp, cert, xA, xB
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- script on CPU (safer) ----
    model_cpu = model.to("cpu").eval()

    # If your model has any training-only flags, ensure they are off:
    # model_cpu.decoder.embedding_decoder.train(False)
    # model_cpu.decoder.detach = True  # if you rely on detaching between scales

    scripted = torch.jit.script(model_cpu)
    scripted = torch.jit.freeze(scripted, preserved_attrs=["forward_ts"])
    # scripted = torch.jit.optimize_for_inference(scripted)  # optional
    print(scripted)          # shows exported methods
    # or:
    print(scripted.code)     # prints schemas if available

    scripted.save(OUT_TS)
    print(f"Saved TorchScript to {OUT_TS}")


if __name__ == "__main__":
    main()

