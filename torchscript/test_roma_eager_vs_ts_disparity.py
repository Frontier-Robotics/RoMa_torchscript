#!/usr/bin/env python3
import os
import argparse
import time
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


import json
from math import isfinite



DINO_P = "/shared/models/dinov2_vitl14_pretrain.pth"
ROMA_P = "/shared/models/roma_outdoor.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _json_default(o):
    import numpy as _np
    import torch as _torch
    # torch/np scalars
    if isinstance(o, (_np.generic,)):
        return o.item()
    if isinstance(o, (_torch.Tensor,)):
        return o.detach().cpu().tolist()
    # fallback: cast to str (or raise)
    try:
        return float(o)
    except Exception:
        return str(o)


def safe_load(p, device):
    try:
        return torch.load(p, map_location=device, weights_only=True)  # PyTorch ≥2.4
    except TypeError:
        return torch.load(p, map_location=device)  # fallback

ROMA_W = safe_load(ROMA_P, device)
DINO_CKPT = safe_load(DINO_P, device)
DINO_W = DINO_CKPT.get("model", DINO_CKPT)


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def masked_stats(err, mask):
    m = np.isfinite(err) & mask
    if not np.any(m):
        return dict(count=0, mae=np.nan, rmse=np.nan, med=np.nan)
    e = err[m]
    return dict(
        count=int(e.size),
        mae=float(np.mean(np.abs(e))),
        rmse=float(np.sqrt(np.mean(e**2))),
        med=float(np.median(np.abs(e))),
    )

def bad_pixel_rates(err_abs, mask, thrs=(0.5, 1.0, 2.0, 3.0)):
    m = np.isfinite(err_abs) & mask
    rates = {}
    if not np.any(m):
        for t in thrs:
            rates[f"bad@{t}px"] = np.nan
        return rates
    e = err_abs[m]
    n = e.size
    for t in thrs:
        rates[f"bad@{t}px"] = float(np.mean(e > t))
    return rates

def relative_mae(err_abs, disp_ref, mask, eps=1e-6):
    # rel error = |Δ| / (|disp_ref| + eps)
    m = np.isfinite(err_abs) & np.isfinite(disp_ref) & mask
    if not np.any(m):
        return np.nan
    rel = err_abs[m] / (np.abs(disp_ref[m]) + eps)
    return float(np.mean(rel))

def save_montage(rows, out_path):
    """
    rows: list[list[np.ndarray(BGR uint8)]]
    - Resizes images in each row to that row's min-height
    - Horizontally stacks per-row
    - Pads each row canvas on the right to max row width
    - Vertically stacks rows
    """
    def resize_to_h(img, h):
        if img.shape[0] == h:
            return img
        w = int(round(img.shape[1] * (h / img.shape[0])))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    row_imgs = []
    row_widths = []
    row_heights = []

    # 1) build each row at consistent height (per-row)
    for row in rows:
        assert len(row) > 0, "save_montage: empty row"
        h = min(im.shape[0] for im in row)
        resized = [resize_to_h(im, h) for im in row]
        stacked = np.hstack(resized)  # shape: (h, W_row, 3)
        row_imgs.append(stacked)
        row_heights.append(stacked.shape[0])
        row_widths.append(stacked.shape[1])

    # 2) pad rows to same width
    max_w = max(row_widths)
    padded_rows = []
    for img, w in zip(row_imgs, row_widths):
        if w < max_w:
            pad = max_w - w
            img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        padded_rows.append(img)

    # 3) vstack
    full = np.vstack(padded_rows)
    cv2.imwrite(out_path, full)


def colormap_float(arr, vmin=None, vmax=None, cmap=cv2.COLORMAP_JET, nan_to_zero=True):
    d = np.array(arr, dtype=np.float32, copy=True)
    nan_mask = ~np.isfinite(d)
    if vmin is None:
        vmin = np.nanpercentile(d, 5) if np.any(~nan_mask) else 0.0
    if vmax is None:
        vmax = np.nanpercentile(d, 95) if np.any(~nan_mask) else 1.0
    if vmax <= vmin: vmax = vmin + 1e-6
    if nan_to_zero:
        d[nan_mask] = vmin
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    img8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(img8, cmap)

def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 28), (0,0,0), thickness=-1)
    cv2.putText(out, text, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def binned_calibration(cert_prob, abs_err, mask, bins=10):
    # Returns per-bin: [low, high, count, mean_cert, mean_abs_err]
    m = np.isfinite(cert_prob) & np.isfinite(abs_err) & mask
    if not np.any(m):
        return []
    p = cert_prob[m].ravel()
    e = abs_err[m].ravel()
    edges = np.linspace(0.0, 1.0, bins+1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        sel = (p >= lo) & (p < hi) if i < bins-1 else (p >= lo) & (p <= hi)
        if np.any(sel):
            rows.append([float(lo), float(hi), int(sel.sum()),
                         float(np.mean(p[sel])), float(np.mean(e[sel]))])
        else:
            rows.append([float(lo), float(hi), 0, np.nan, np.nan])
    return rows

# -------------------------
# CLI
# -------------------------
def parse_res(s: str):
    h, w = s.lower().split("x")
    return int(h), int(w)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["eager", "ts", "both"], default="both")
parser.add_argument("--left", required=True)
parser.add_argument("--right", required=True)
parser.add_argument("--fx", type=float, required=True)
parser.add_argument("--baseline_cm", type=float, required=True)
parser.add_argument("--coarse_res", type=parse_res, default=(560,560))
parser.add_argument("--upsample_res", type=parse_res, default=(560,560))
parser.add_argument("--ts-backend", choices=["auto", "sdpa"], default="auto",
                    help="sdpa disables xFormers so TorchScript uses native SDPA")
parser.add_argument("--ts-export", choices=["trace", "script"], default="trace")
parser.add_argument("--ts-out", default="models/roma_fp32_onepass.ts")
parser.add_argument("--outdir", default="outputs")
args = parser.parse_args()

# If we want TS to use SDPA fallback, disable xFormers BEFORE imports.
if args.ts_backend == "sdpa":
    os.environ["XFORMERS_DISABLED"] = "1"

# Eager model (reference)
from romatch import roma_outdoor as roma_outdoor_eager
# TorchScriptable model (your traceable wrapper)
from romatch_ts2 import roma_outdoor as roma_outdoor_ts

# -------------------------
# Speed knobs
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

# -------------------------
# Helpers
# -------------------------
def im_to_tensor_rgb_01(im: Image.Image):
    t = transforms.ToTensor()(im.convert("RGB"))
    t = transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])(t)
    return t

def resize_to(t: torch.Tensor, size_hw, mode="bilinear"):
    return F.interpolate(t[None], size=size_hw, mode=mode, align_corners=False)[0]

def up_to_size(flow, cert, Ht, Wt):
    # flow: [N,2,H,W], cert: [N,1,H,W]
    if flow.shape[-2:] != (Ht, Wt):
        flow = F.interpolate(flow, size=(Ht, Wt), mode="bilinear", align_corners=False)
        cert = F.interpolate(cert, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return flow, cert

def flow_to_warp(flow: torch.Tensor):
    # flow: [N,2,H,W] in normalized [-1,1] coords of target image (A->B)
    N, _, H, W = flow.shape
    yy = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=flow.device)
    xx = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=flow.device)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')   # gy:[H,W], gx:[H,W]
    grid = torch.stack([gx, gy], dim=-1).expand(N, H, W, 2)  # [N,H,W,2] as (x,y)
    a2b = flow.permute(0, 2, 3, 1)                              # [N,H,W,2]
    return torch.cat([grid, a2b], dim=-1)                       # [N,H,W,4]
                  # [N,H,W,4]

def warp_to_dense_disparity_from_warp(warp):
    """
    warp: [H,2W,4] (eager) OR [N,H,W,4] (TS single-direction)
    Returns: disparity in pixels (same H,W as the left image)
    """
    if warp.dim() == 3:  # [H,2W,4], bidirectional packed
        H, W2, _ = warp.shape
        W = W2 // 2
        wL = warp[:, :W, :]                # [H,W,4]
        x1, x2 = wL[..., 0], wL[..., 2]
        x1_px = ((x1 + 1) / 2) * (W - 1)
        x2_px = ((x2 + 1) / 2) * (W - 1)
        disp = x1_px - x2_px
        disp[disp <= 0] = torch.nan
        return disp
    elif warp.dim() == 4:  # [N,H,W,4]
        _, H, W, _ = warp.shape
        x1, x2 = warp[..., 0], warp[..., 2]
        x1_px = ((x1 + 1) / 2) * (W - 1)
        x2_px = ((x2 + 1) / 2) * (W - 1)
        disp = x1_px - x2_px
        disp[disp <= 0] = torch.nan
        return disp  # [N,H,W]
    else:
        raise ValueError("Unexpected warp shape")

def depth_to_colormap(depth, vmin=None, vmax=None):
    d = np.copy(depth)
    nan_mask = np.isnan(d)
    if np.all(nan_mask):
        d[:] = 0
        vmin, vmax = 0, 1
    else:
        if vmin is None: vmin = np.nanpercentile(d, 5)
        if vmax is None: vmax = np.nanpercentile(d, 95)
        if vmax <= vmin: vmax = vmin + 1e-6
        d[nan_mask] = vmin
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    img_8u = (norm * 255).astype(np.uint8)
    jet = cv2.applyColorMap(img_8u, cv2.COLORMAP_JET)
    jet[nan_mask] = 0
    return jet

def prob_to_logit(p: torch.Tensor, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
(Hc, Wc) = args.coarse_res
(Hu, Wu) = args.upsample_res

for v, name in [(Hc,"Hc"), (Wc,"Wc")]:
    if v % 14 != 0:
        raise ValueError(f"{name} must be divisible by 14, got {v}")

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.dirname(args.ts_out), exist_ok=True)

L_img = Image.open(args.left).convert("RGB")
R_img = Image.open(args.right).convert("RGB")

# Coarse-res tensors (for TS input)
L = im_to_tensor_rgb_01(L_img)
R = im_to_tensor_rgb_01(R_img)
L_c = resize_to(L, (Hc, Wc)).unsqueeze(0).to(device)
R_c = resize_to(R, (Hc, Wc)).unsqueeze(0).to(device)

# -------------------------
# EAGER (reference)
# -------------------------
eager_pack = None
if args.mode in ("eager", "both"):
    print(f"Using coarse resolution {args.coarse_res}, and upsample res {args.upsample_res}")
    m_eager = roma_outdoor_eager(
        device=device,
        coarse_res=args.coarse_res,
        upsample_res=args.upsample_res,
        weights=ROMA_W,
        dinov2_weights=DINO_W,
        amp_dtype=torch.float32,
    ).eval().to(device)
    m_eager.upsample_preds = True  # two-pass (coarse + refine to upsample_res)

    # warmup
    for _ in range(2):
        _ = m_eager.match(args.left, args.right, batched=False, device=device)

    t0 = time.time()
    warp_eager, cert_eager_prob = m_eager.match(args.left, args.right, batched=False, device=device)
    t1 = time.time()
    print(f"[eager] time: {(t1 - t0) * 1000:.2f} ms")

    # warp_eager: [Hu,2*Wu,4] normalized; certainty: [Hu,Wu] probabilities
    assert warp_eager.shape[0] == Hu and warp_eager.shape[1] == 2 * Wu, \
        f"warp shape {warp_eager.shape} != expected [{Hu}, {2*Wu}, 4]"

    # disparity (left half)
    disp_eager = warp_to_dense_disparity_from_warp(warp_eager)  # [Hu,Wu]
    cert_eager_logits = prob_to_logit(cert_eager_prob)          # [Hu,Wu]

    # save eager visuals
    disp_vis = depth_to_colormap(disp_eager.cpu().numpy())
    cv2.imwrite(os.path.join(args.outdir, "eager_disp.png"), disp_vis)
    cert_vis = (np.clip(cert_eager_logits.cpu().numpy(), -10, 10) - (-10)) / 20.0
    cert_vis = (cert_vis * 255).astype(np.uint8)
    cert_vis = cv2.applyColorMap(cert_vis, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(args.outdir, "eager_cert_logits.png"), cert_vis)

    eager_pack = dict(
        disp_np=disp_eager.cpu().numpy(),
        cert_logits_np=cert_eager_logits.cpu().numpy(),
        warp=warp_eager,  # torch [Hu,2Wu,4]
    )

# -------------------------
# TORCHSCRIPT
# -------------------------
class TSRoMa2Stage(torch.nn.Module):
    def __init__(self, m, Hc, Wc, Hu, Wu):
        super().__init__()
        self.m = m
        self.Hc, self.Wc = Hc, Wc
        self.Hu, self.Wu = Hu, Wu
        # precompute scale factor like match()
        self.scale_factor = math.sqrt((Hu*Wu) / (Hc*Wc))

    def forward(self, L_up: torch.Tensor, R_up: torch.Tensor):
        # 1) coarse downsample inside TS
        L_c = F.interpolate(L_up, size=(self.Hc, self.Wc), mode="bilinear", align_corners=False)
        R_c = F.interpolate(R_up, size=(self.Hc, self.Wc), mode="bilinear", align_corners=False)

        # 2) coarse pass
        coarse = self.m.forward({"im_A": L_c, "im_B": R_c}, batched=True, upsample=False, scale_factor=1.0)
        finest = coarse[1]  # dict with 'flow' and 'certainty' tensors at finest coarse scale

        # 3) learned upsample/refine pass
        up = self.m.forward(
            {"im_A": L_up, "im_B": R_up},            # <- only tensors in the batch dict
            batched=True,
            upsample=True,
            scale_factor=self.scale_factor,
            flow_in=finest["flow"],                  # <- pass directly
            cert_in=finest["certainty"],             # <- pass directly
        )
        flow = up[1]["flow"]          # [N,2,Hu,Wu], normalized
        cert = up[1]["certainty"]     # logits [N,1,Hu,Wu] (no sigmoid)
        return flow, cert


flow_t, cert_t = None, None
if args.mode in ("ts", "both"):
    def make_model(device, coarse_res, upsample_res):
        print(f"Using coarse resolution {coarse_res}, and upsample res {coarse_res}")
        m = roma_outdoor_ts(
            device=device,
            coarse_res=(Hc,Wc),
            upsample_res=(Hu,Wu),   # one-pass at coarse; we upsample outputs manually
            weights=ROMA_W,
            dinov2_weights=DINO_W,
            amp_dtype=torch.float32,
        ).eval().to(device)
        m.upsample_preds = False
        return m

    m_ts = make_model(device, (Hc, Wc), (Hc, Wc))
    wrapper = TSRoMa2Stage(m_ts, Hc, Wc, Hu, Wu).eval().to(device)


    def try_script_each_leaf(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            try:
                # Only try to script “leafy” children first
                if not any(True for _ in child.children()):
                    torch.jit.script(child)
                else:
                    # recurse first to narrow down
                    try_script_each_leaf(child, full)
            except Exception as e:
                print(f"[JIT FAIL] {full}: {type(child).__name__}\n{e}\n")
                # For composites, also try scripting the composite to confirm
                try:
                    torch.jit.script(child)
                except Exception as e2:
                    print(f"[JIT FAIL (composite)] {full}: {type(child).__name__}\n{e2}\n")

# after you build `wrapper` / `matcher`:
    try_script_each_leaf(wrapper)  # or your top-level module

    

    with torch.inference_mode():
        if args.ts_export == "script":
            ts = torch.jit.script(wrapper)
        else:
            ts = torch.jit.trace(wrapper, (L_c, R_c), strict=False, check_trace=False)
        ts.save(args.ts_out)

        # warmup
        for _ in range(2):
            _ = ts(L_c, R_c)

        t0 = time.time()
        flow_b, cert_b = ts(L_c, R_c)
        t1 = time.time()
        print(f"[ts] time: {(t1 - t0) * 1000:.2f} ms")

        flow_t = flow_b[0]  # (2,Hc,Wc)
        cert_t = cert_b[0]  # (1,Hc,Wc)
    
# Build TS disparity/certainty if available



# -------------------------
if flow_t is not None:
    # upsample to (Hu,Wu) and build warp
    flow_u, cert_u = up_to_size(flow_t.unsqueeze(0), cert_t.unsqueeze(0), Hu, Wu)
    warp_ts = flow_to_warp(flow_u)[0]                     # [Hu,Wu,4]
    disp_ts = warp_to_dense_disparity_from_warp(warp_ts.unsqueeze(0))[0]  # [Hu,Wu]
    cert_ts_logits = cert_u[0,0]                          # [Hu,Wu] logits

    # Save TS visuals (disparity + certainty)
    disp_ts_vis = depth_to_colormap(to_np(disp_ts))
    cv2.imwrite(os.path.join(args.outdir, "ts_disp.png"), disp_ts_vis)
    cert_ts_prob = sigmoid_np(to_np(cert_ts_logits))
    cert_ts_vis = colormap_float(cert_ts_prob, vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(args.outdir, "ts_cert_prob.png"), cert_ts_vis)

# -------------------------
# Deep comparison vs EAGER
# -------------------------
if (eager_pack is not None) and (flow_t is not None):
    # Numpy views
    # Expected image size
    Hu, Wu = args.upsample_res

    # Grab raw arrays
    disp_ref   = to_np(eager_pack["disp_np"])            # [Hu,Wu]
    cert_ref_l = to_np(eager_pack["cert_logits_np"])     # [Hu,W? or 2W?]
    disp_ts_np = to_np(disp_ts)                          # [Hu,Wu]
    cert_ts_l  = to_np(cert_ts_logits)                   # [Hu,Wu]

    # --- FIX: handle packed (2W) certainty from eager ---
    def left_half_if_packed(a, W, name):
        if a.ndim < 2:
            raise ValueError(f"{name}: expected 2D+, got {a.shape}")
        H, Wgot = a.shape[:2]
        if Wgot == 2 * W:
            return a[:, :W, ...]
        if Wgot != W:
            raise ValueError(f"{name}: unexpected width {Wgot}, expected {W} or {2*W}")
        return a

    cert_ref_l = left_half_if_packed(cert_ref_l, Wu, "cert_ref_logits")

    # (Optional) sanity print
    print("[post-fix shapes]",
        "disp_ref", disp_ref.shape,
        "disp_ts", disp_ts_np.shape,
        "cert_ref", cert_ref_l.shape,
        "cert_ts", cert_ts_l.shape)

    # Certainty → probabilities
    cert_ref_p = 1.0 / (1.0 + np.exp(-cert_ref_l))
    cert_ts_p  = 1.0 / (1.0 + np.exp(-cert_ts_l))

    # Valid mask + continue with metrics...
    valid = np.isfinite(disp_ref) & np.isfinite(disp_ts_np) & (disp_ref > 0) & (disp_ts_np > 0)




    # mask: finite disparities, positive disparities, and both certainties > 0.2
    certainty_thresh = 0.2
    valid = (
        np.isfinite(disp_ref) & np.isfinite(disp_ts_np) &
        (disp_ref > 0) & (disp_ts_np > 0) &
        (cert_ref_p > certainty_thresh) &
        (cert_ts_p  > certainty_thresh)
    )


    # Errors
    err = disp_ts_np - disp_ref
    err_abs = np.abs(err)

    # Error heatmaps
    err_vis = colormap_float(err, vmin=-np.nanpercentile(np.abs(err[valid]), 95),
                                  vmax= np.nanpercentile(np.abs(err[valid]), 95),
                                  cmap=cv2.COLORMAP_COOL)
    rel_err = np.where(valid, err_abs / (np.abs(disp_ref) + 1e-6), np.nan)
    rel_vis = colormap_float(rel_err, vmin=0.0, vmax=np.nanpercentile(rel_err[valid], 95), cmap=cv2.COLORMAP_MAGMA)

    cv2.imwrite(os.path.join(args.outdir, "err_signed_heatmap.png"), err_vis)
    cv2.imwrite(os.path.join(args.outdir, "err_relative_heatmap.png"), rel_vis)

    # Disparity visual comparisons
    disp_ref_vis = depth_to_colormap(disp_ref)
    disp_ts_vis  = depth_to_colormap(disp_ts_np)
    cv2.imwrite(os.path.join(args.outdir, "eager_disp.png"), disp_ref_vis)  # re-save to match naming
    cv2.imwrite(os.path.join(args.outdir, "ts_disp.png"), disp_ts_vis)

    # Certainty visual comparisons (probabilities)
    cert_ref_vis = colormap_float(cert_ref_p, vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_VIRIDIS)
    cert_ts_vis  = colormap_float(cert_ts_p,  vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(args.outdir, "eager_cert_prob.png"), cert_ref_vis)
    cv2.imwrite(os.path.join(args.outdir, "ts_cert_prob.png"),   cert_ts_vis)

    # Labeled montages
    row1 = [put_label(disp_ref_vis, "EAGER disparity"),
            put_label(disp_ts_vis,  "TS disparity"),
            put_label(err_vis,      "Signed error (px)")]

    row2 = [put_label(cert_ref_vis, "EAGER certainty prob"),
            put_label(cert_ts_vis,  "TS certainty prob"),
            put_label(rel_vis,      "Relative |error|")]

    save_montage([row1, row2], os.path.join(args.outdir, "comparison_montage.png"))

    # Scalar metrics
    stats = masked_stats(err, valid)
    stats_abs = masked_stats(err_abs, valid)
    stats["mae"]   = stats_abs["mae"]
    stats["rmse"]  = stats_abs["rmse"]
    stats["medae"] = stats_abs["med"]

    stats.update(bad_pixel_rates(err_abs, valid, thrs=(0.5, 1.0, 2.0, 3.0)))
    stats["relative_mae"] = relative_mae(err_abs, disp_ref, valid)

    # Certainty ↔ error correlations (higher certainty should mean lower error)
    try:
        from scipy.stats import pearsonr, spearmanr
        p_ref = pearsonr(cert_ref_p[valid].ravel(), (-err_abs[valid]).ravel())[0]
        s_ref = spearmanr(cert_ref_p[valid].ravel(), (-err_abs[valid]).ravel()).correlation
        p_ts  = pearsonr(cert_ts_p[valid].ravel(),  (-err_abs[valid]).ravel())[0]
        s_ts  = spearmanr(cert_ts_p[valid].ravel(), (-err_abs[valid]).ravel()).correlation
    except Exception:
        # Fallback: simple Pearson via numpy (no tie-aware rank corr)
        def pearson_np(a, b):
            a = a - np.mean(a); b = b - np.mean(b)
            den = np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12
            return float((a*b).sum() / den)
        p_ref = pearson_np(cert_ref_p[valid].ravel(), (-err_abs[valid]).ravel())
        p_ts  = pearson_np(cert_ts_p[valid].ravel(),  (-err_abs[valid]).ravel())
        s_ref = np.nan
        s_ts  = np.nan

    stats["corr_cert_vs_neg_abs_err_eager_pearson"] = p_ref
    stats["corr_cert_vs_neg_abs_err_ts_pearson"]    = p_ts
    stats["corr_cert_vs_neg_abs_err_eager_spearman"] = s_ref
    stats["corr_cert_vs_neg_abs_err_ts_spearman"]    = s_ts
    stats["num_valid"] = int(np.sum(valid))

    # Save metrics.json
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2, default=_json_default, allow_nan=True)

    # Reliability / calibration table
    calib_rows_ref = binned_calibration(cert_ref_p, err_abs, valid, bins=10)
    calib_rows_ts  = binned_calibration(cert_ts_p,  err_abs, valid, bins=10)

    with open(os.path.join(args.outdir, "calibration.csv"), "w") as f:
        f.write("which,bin_lo,bin_hi,count,mean_cert,mean_abs_err\n")
        for lo,hi,c,mc,me in calib_rows_ref:
            f.write(f"eager,{lo:.3f},{hi:.3f},{c},{mc if np.isfinite(mc) else ''},{me if np.isfinite(me) else ''}\n")
        for lo,hi,c,mc,me in calib_rows_ts:
            f.write(f"ts,{lo:.3f},{hi:.3f},{c},{mc if np.isfinite(mc) else ''},{me if np.isfinite(me) else ''}\n")

    # Also keep your original quick delta print
    dmax = np.nanmax(err_abs[valid]) if np.any(valid) else np.nan
    dmean = np.nanmean(err_abs[valid]) if np.any(valid) else np.nan
    dmed  = np.nanmedian(err_abs[valid]) if np.any(valid) else np.nan
    print(f"[delta disparity] max={dmax:.6f} mean={dmean:.6f} median={dmed:.6f}")
    print(f"[metrics] saved to {os.path.join(args.outdir, 'metrics.json')} and calibration.csv")
