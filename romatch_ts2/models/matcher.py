import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
#from einops import rearrange
import warnings
from warnings import warn
from PIL import Image

from romatch_ts2.utils import get_tuple_transform_ops
from romatch_ts2.utils.local_correlation import local_correlation
from romatch_ts2.utils.utils import check_rgb, cls_to_flow_refine, get_autocast_params, check_not_i16
from romatch_ts2.utils.kde import kde

from typing import Optional, Tuple, Dict
from torch import Tensor

import math

class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        #self.local_corr_radius = local_corr_radius
        #self.corr_in_other = corr_in_other
        #self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        #self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype

        self.local_corr_radius: int = int(local_corr_radius) if local_corr_radius is not None else 0
        self.corr_in_other: bool = bool(corr_in_other)
        self.no_im_B_fm: bool = bool(no_im_B_fm)
        self.concat_logits: bool = bool(concat_logits)
        # if you have a displacement flag/string:
        self.has_displacement_emb: bool = displacement_emb is not None
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def _forward_body(
        self,
        x: Tensor,
        y: Tensor,
        flow: Tensor,
        scale_factor: float,
        logits: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        b, c, hs, ws = x.shape

        # 1) warp
        x_hat = Fn.grid_sample(
            y, flow.permute(0, 2, 3, 1),
            align_corners=False, mode=self.sample_mode
        )

        # 2) optional disp embedding + (optional) local correlation
        if self.has_displacement_emb:
            ys = torch.linspace(-1.0 + 1.0 / float(hs), 1.0 - 1.0 / float(hs), hs, device=x.device)
            xs = torch.linspace(-1.0 + 1.0 / float(ws), 1.0 - 1.0 / float(ws), ws, device=x.device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            im_A_coords = torch.stack((gx, gy)).unsqueeze(0).expand(b, 2, hs, ws)
            in_displacement = flow - im_A_coords
            emb_in_displacement = self.disp_emb((40.0 / 32.0) * float(scale_factor) * in_displacement)

            if self.local_corr_radius > 0:
                if self.corr_in_other:
                    local_corr = local_correlation(
                        x, y,
                        local_radius=int(self.local_corr_radius),
                        flow=flow,
                        sample_mode=self.sample_mode,
                    )
                else:
                    raise NotImplementedError("Local corr in own frame should not be used.")
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_im_B_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)

        if self.concat_logits and (logits is not None):
            d = torch.cat((d, logits), dim=1)

        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())

        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        flow: Tensor,
        scale_factor: float = 1.0,
        logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Only use autocast in eager mode, never in scripted graphs.
        if (not torch.jit.is_scripting()) and self.amp:
            if x.is_cuda:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    return self._forward_body(x, y, flow, float(scale_factor), logits)
            else:
                # CPU autocast also supported; safe guard
                with torch.cpu.amp.autocast(dtype=self.amp_dtype):
                    return self._forward_body(x, y, flow, float(scale_factor), logits)
        else:
            # Script path (or AMP off): no autocast context at all.
            return self._forward_body(x, y, flow, float(scale_factor), logits)

class CosKernel(nn.Module):
    def __init__(self, T: float, learn_temperature: bool = False):
        super().__init__()
        self.learn_temperature = learn_temperature
        base = torch.tensor(float(T))
        if learn_temperature:
            self.T_param = nn.Parameter(base)               # always present
        else:
            self.register_buffer("T_param", base, persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # cosine sim (JIT-safe: no torch.norm kwargs)
        num = torch.einsum("bnd,bmd->bnm", x, y)
        x_norm = (x * x).sum(-1).clamp_min(eps).sqrt()[..., None]     # (B,N,1)
        y_norm = (y * y).sum(-1).clamp_min(eps).sqrt()[:, None, :]    # (B,1,M)
        c = num / (x_norm * y_norm)

        T = self.T_param
        if self.learn_temperature:
            T = T.abs() + 0.01
        T = T.to(dtype=c.dtype, device=c.device)

        return ((c - 1.0) / T).exp()




class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K: int = int(self.covar_size)  # ensure python int
        b, h, w, h2, w2 = cov.shape
        # (optionally assert shapes)
        assert h == h2 and w == w2

        # Pad for local neighborhoods
        pad_each = K // 2
        cov = Fn.pad(cov, (pad_each, pad_each, pad_each, pad_each), mode="constant", value=0.0)

        device = cov.device
        dtype = cov.dtype  # not strictly needed below, but handy if you add ops

        # integer grid tensors
        delta_y = torch.arange(-pad_each, pad_each + 1, device=device)
        delta_x = torch.arange(-pad_each, pad_each + 1, device=device)
        delta = torch.stack(torch.meshgrid(delta_y, delta_x, indexing='ij'), dim=-1)  # (K,K,2)

        pos_y = torch.arange(pad_each, int(h) + pad_each, device=device)
        pos_x = torch.arange(pad_each, int(w) + pad_each, device=device)
        positions = torch.stack(torch.meshgrid(pos_y, pos_x, indexing='ij'), dim=-1)  # (h,w,2)

        neighbours = positions[:, :, None, None, :] + delta[None, :, :]  # (h,w,K,K,2)

        # flatten indices
        hw: int = int(h) * int(w)
        # make sure sizes passed to expand are ints
        points = torch.arange(hw, device=device, dtype=torch.long).unsqueeze(1).expand(hw, int(K) * int(K))

        # gather local KxK covariance patches
        # cov shape after pad: (b, h + K - 1, w + K - 1, h, w)  if yours differs, adapt the reshape/indexing accordingly
        cov_view = cov.reshape(b, hw, int(h) + K - 1, int(w) + K - 1)

        ny = neighbours[..., 0].reshape(-1)
        nx = neighbours[..., 1].reshape(-1)

        local_cov = cov_view[:, points.reshape(-1), ny, nx]  # (b, hw*K*K)
        local_cov = local_cov.reshape(b, int(h), int(w), K * K)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    
    def _noop(self):
        return  # placeholder to keep class non-empty if you need it

    def _reshape_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # "b d h w -> b (h w) d"
        b, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

    def _reshape_tokens_back(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # "b (h w) d -> b d h w"
        b, hw, d = x.shape
        return x.transpose(1, 2).reshape(b, d, h, w)

    def _reshape_gram(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # "b (h w) (r c) -> b h w r c"
        b, hw1, hw2 = x.shape
        return x.reshape(b, h, w, h, w)

    def get_pos_enc(self, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = y.shape
        yy = torch.linspace(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h), h, device=y.device)
        xx = torch.linspace(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w), w, device=y.device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')  # (h, w)
        # stack as (h, w, 2) with order (x,y) to match your previous code
        coords = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, h, w, 2)  # (b, h, w, 2)
        coords = coords.permute(0, 3, 1, 2)  # "b h w d -> b d h w"
        return self.project_to_basis(coords)  # -> (b, gp_dim, h, w)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h1, w1 = x.shape
        _, _, h2, w2 = y.shape

        f = self.get_pos_enc(y)                                 # (b, d, h2, w2)
        _, d, _, _ = f.shape

        x_flat = self._reshape_tokens(x.float())                # (b, h1*w1, c)
        y_flat = self._reshape_tokens(y.float())                # (b, h2*w2, c)
        f_flat = self._reshape_tokens(f)                        # (b, h2*w2, d)

        K_xx = self.K(x_flat, x_flat)                           # (b, h1*w1, h1*w1)
        K_yy = self.K(y_flat, y_flat)                           # (b, h2*w2, h2*w2)
        K_xy = self.K(x_flat, y_flat)                           # (b, h1*w1, h2*w2)
        K_yx = K_xy.transpose(1, 2)                             # (b, h2*w2, h1*w1)

        sigma = self.sigma_noise * torch.eye(h2 * w2, device=x.device).unsqueeze(0)
        # # suppress potential cholesky warnings during scripting
        # with warnings.catch_warnings():
        #     K_yy_inv = torch.linalg.inv(K_yy + sigma)

        M = K_yy + sigma  # (B, N, N)
        I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype).expand(M.size(0), -1, -1)
        K_yy_inv = torch.linalg.solve(M, I)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f_flat))             # (b, h1*w1, d)
        mu_x = self._reshape_tokens_back(mu_x, h1, w1)          # (b, d, h1, w1)

        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))   # (b, h1*w1, h1*w1)
            cov_x = self._reshape_gram(cov_x, h1, w1)           # (b, h1, w1, h1, w1)

            # local KxK neighbourhoods -> (b, K^2, h1, w1)
            local_cov_x = self.get_local_cov(cov_x)             # (b, h1, w1, K^2)
            local_cov_x = local_cov_x.permute(0, 3, 1, 2)       # "b h w K -> b K h w"

            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)    # (b, d+K^2, h1, w1)
        else:
            gp_feats = mu_x                                     # (b, d, h1, w1)

        return gp_feats


    # def get_pos_enc(self, y):
    #     b, c, h, w = y.shape
    #     coarse_coords = torch.meshgrid(
    #         (
    #             torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
    #             torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
    #         ),
    #         indexing = 'ij'
    #     )

    #     coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
    #         None
    #     ].expand(b, h, w, 2)
    #     coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
    #     coarse_embedded_coords = self.project_to_basis(coarse_coords)
    #     return coarse_embedded_coords

    # def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     b, c, h1, w1 = x.shape
    #     b, c, h2, w2 = y.shape
    #     f = self.get_pos_enc(y)
    #     b, d, h2, w2 = f.shape
    #     x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
    #     K_xx = self.K(x, x)
    #     K_yy = self.K(y, y)
    #     K_xy = self.K(x, y)
    #     K_yx = K_xy.permute(0, 2, 1)
    #     sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
    #     with warnings.catch_warnings():
    #         K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

    #     mu_x = K_xy.matmul(K_yy_inv.matmul(f))
    #     mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
    #     if not self.no_cov:
    #         cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
    #         cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
    #         local_cov_x = self.get_local_cov(cov_x)
    #         local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
    #         gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
    #     else:
    #         gp_feats = mu_x
    #     return gp_feats

class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        detach=False,
        scales="all",
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
        amp: bool = False,                 # <- add this
        amp_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        self.pos_embeddings = {} if pos_embeddings is None else pos_embeddings
        self.scales = ["32", "16", "8", "4", "2", "1"] if scales == "all" else scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp = amp                      # <- store it
        self.amp_dtype = amp_dtype          # already used
        
        # Mirror as fixed attributes for TorchScript:
        self.proj_16 = proj["16"] if "16" in proj else None
        self.proj_8  = proj["8"]  if "8"  in proj else None
        self.proj_4  = proj["4"]  if "4"  in proj else None
        self.proj_2  = proj["2"]  if "2"  in proj else None
        self.proj_1  = proj["1"]  if "1"  in proj else None

        self.gp_16 = gps["16"] if "16" in gps else None

        self.refiner_16 = conv_refiner["16"] if "16" in conv_refiner else None
        self.refiner_8  = conv_refiner["8"]  if "8"  in conv_refiner else None
        self.refiner_4  = conv_refiner["4"]  if "4"  in conv_refiner else None
        self.refiner_2  = conv_refiner["2"]  if "2"  in conv_refiner else None
        self.refiner_1  = conv_refiner["1"]  if "1"  in conv_refiner else None

    @torch.jit.export
    def _apply_proj(self, s: int, x: Tensor) -> Tensor:
        if s == 16 and self.proj_16 is not None:
            return self.proj_16(x)
        elif s == 8 and self.proj_8 is not None:
            return self.proj_8(x)
        elif s == 4 and self.proj_4 is not None:
            return self.proj_4(x)
        elif s == 2 and self.proj_2 is not None:
            return self.proj_2(x)
        elif s == 1 and self.proj_1 is not None:
            return self.proj_1(x)
        return x  # no-op if not present

    @torch.jit.export
    def _apply_gp(self, s: int, f1: Tensor, f2: Tensor) -> Tensor:
        if s == 16 and self.gp_16 is not None:
            return self.gp_16(f1, f2)
        # add more if you ever have other gp scales
        raise RuntimeError("GP for scale {} is not available".format(s))

    @torch.jit.export
    def _apply_refiner(
        self, s: int, f1: Tensor, f2: Tensor, flow: Tensor, scale_factor: float, logits: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if s == 16 and self.refiner_16 is not None:
            return self.refiner_16(f1, f2, flow, scale_factor=scale_factor, logits=logits)
        elif s == 8 and self.refiner_8 is not None:
            return self.refiner_8(f1, f2, flow, scale_factor=scale_factor, logits=logits)
        elif s == 4 and self.refiner_4 is not None:
            return self.refiner_4(f1, f2, flow, scale_factor=scale_factor, logits=logits)
        elif s == 2 and self.refiner_2 is not None:
            return self.refiner_2(f1, f2, flow, scale_factor=scale_factor, logits=logits)
        elif s == 1 and self.refiner_1 is not None:
            return self.refiner_1(f1, f2, flow, scale_factor=scale_factor, logits=logits)
        raise RuntimeError("Refiner for scale {} is not available".format(s))

    def get_placeholder_flow(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        # 1D coords
        ys = torch.linspace(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h), h, device=device)
        xs = torch.linspace(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w), w, device=device)
        # build 2D without meshgrid
        yy = ys.view(h, 1).expand(h, w)   # (h,w)
        xx = xs.view(1, w).expand(h, w)   # (h,w)
        # stack as (x,y), add batch, expand
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0)  # (1,2,h,w)
        return grid.expand(b, 2, h, w)                    # (b,2,h,w)

    def get_positional_embedding(
        self,
        b: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        # same grid construction
        ys = torch.linspace(-1.0 + 1.0 / float(h), 1.0 - 1.0 / float(h), h, device=device)
        xs = torch.linspace(-1.0 + 1.0 / float(w), 1.0 - 1.0 / float(w), w, device=device)
        yy = ys.view(h, 1).expand(h, w)
        xx = xs.view(1, w).expand(h, w)

        grid_bchw = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(b, 2, h, w)  # (b,2,h,w)

        # If you really need a projection, make sure `self.pos_embedding` exists and is nn.Module.
        # Otherwise, just return the coords.
        if hasattr(self, "pos_embedding") and isinstance(self.pos_embedding, nn.Module):
            return self.pos_embedding(grid_bchw)
        return grid_bchw


    def forward(
        self,
        f1: Dict[int, torch.Tensor],
        f2: Dict[int, torch.Tensor],
        gt_warp: Optional[torch.Tensor] = None,
        gt_prob: Optional[torch.Tensor] = None,
        upsample: bool = False,
        flow: Optional[torch.Tensor] = None,
        certainty: Optional[torch.Tensor] = None,
        scale_factor: float = 1.0,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        coarse_scales = self.embedding_decoder.scales()  # e.g., [16]
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]

        # --- normalize scale keys to int, and build sizes as Dict[int, Tuple[int,int]]
        sizes: Dict[int, Tuple[int, int]] = {}
        for k in f1.keys():
            ks = int(k)
            sizes[ks] = (int(f1[k].shape[-2]), int(f1[k].shape[-1]))

        # convenience handles at scale 1 (must exist)
        h1, w1 = sizes[1]
        b = int(f1[1].shape[0])
        device = f1[1].device

        # coarsest scale in this pass
        coarsest_scale = int(all_scales[0])

        # --- old_stuff tensor: allocate with explicit ints
        hc, wc = sizes[coarsest_scale]
        old_stuff = f1[coarsest_scale].new_zeros(
            (b, self.embedding_decoder.hidden_dim, hc, wc)
        )

        # --- flow/certainty at coarsest scale
        #hc, wc = sizes[coarsest_scale]
        hc = int(hc)
        wc = int(wc)

        if not upsample:
            flow = self.get_placeholder_flow(b, hc, wc, device)  # (b,2,hc,wc)
            certainty = torch.zeros((b, 1, hc, wc),
                                    device=device,
                                    dtype=f1[coarsest_scale].dtype)
        else:
            # TS-safe: ensure tensors before interpolate
            if flow is None:
                flow = torch.zeros((b, 2, hc, wc),
                                device=device,
                                dtype=f1[coarsest_scale].dtype)
            else:
                flow = Fn.interpolate(flow, size=(hc, wc),
                                    align_corners=False, mode="bilinear")

            if certainty is None:
                certainty = torch.zeros((b, 1, hc, wc),
                                        device=device,
                                        dtype=f1[coarsest_scale].dtype)
            else:
                certainty = Fn.interpolate(certainty, size=(hc, wc),
                                        align_corners=False, mode="bilinear")

        corresps: Dict[int, Dict[str, torch.Tensor]] = {}
        displacement = f1[1].new_zeros((b, 2, h1, w1))  # placeholder (shape doesn’t really matter here)

        for new_scale in all_scales:
            ins = int(new_scale)
            hi, wi = sizes[ins]
            corresps[ins] = {}

            f1_s = f1[ins]
            f2_s = f2[ins]

            ins = int(new_scale)
            if ((ins == 16 and self.proj_16 is not None) or
                (ins == 8  and self.proj_8  is not None) or
                (ins == 4  and self.proj_4  is not None) or
                (ins == 2  and self.proj_2  is not None) or
                (ins == 1  and self.proj_1  is not None)):


                if (not torch.jit.is_scripting()) and self.amp:
                    if f1_s.is_cuda:
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            f1_s = self._apply_proj(ins, f1_s.to(torch.float32))
                            f2_s = self._apply_proj(ins, f2_s.to(torch.float32))
                    else:
                        # CPU autocast is OK if you want it; otherwise just do the no-AMP branch
                        with torch.cpu.amp.autocast(dtype=self.amp_dtype):
                            f1_s = self._apply_proj(ins, f1_s.to(torch.float32))
                            f2_s = self._apply_proj(ins, f2_s.to(torch.float32))
                else:
                    # Scripted path or AMP disabled: no autocast context at all
                    f1_s = self._apply_proj(ins, f1_s.to(torch.float32))
                    f2_s = self._apply_proj(ins, f2_s.to(torch.float32))




            if ins in coarse_scales:
                
                old_stuff = Fn.interpolate(old_stuff, size=(hi, wi), mode="bilinear", align_corners=False)
                gp_posterior = self._apply_gp(ins, f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff_ret = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, ins
                )
                # Keep old_stuff as Tensor; only overwrite if we got a Tensor back
                if old_stuff_ret is not None:
                    old_stuff = old_stuff_ret

                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(gm_warp_or_cls).permute(0, 3, 1, 2)
                    if self.training:
                        corresps[ins].update({"gm_cls": gm_warp_or_cls, "gm_certainty": certainty})
                else:
                    if self.training:
                        corresps[ins].update({"gm_flow": gm_warp_or_cls, "gm_certainty": certainty})
                    flow = gm_warp_or_cls.detach()


            has_refiner = (
                (ins == 16 and self.refiner_16 is not None) or
                (ins == 8  and self.refiner_8  is not None) or
                (ins == 4  and self.refiner_4  is not None) or
                (ins == 2  and self.refiner_2  is not None) or
                (ins == 1  and self.refiner_1  is not None)
            )

            if has_refiner:
                if self.training:
                    corresps[ins].update({"flow_pre_delta": flow})

                # always run the refiner (define delta_flow/delta_certainty in all modes)
                delta_flow, delta_certainty = self._apply_refiner(
                    ins, f1_s, f2_s, flow, float(scale_factor), certainty
                )

                if self.training:
                    corresps[ins].update({"delta_flow": delta_flow})

                # compute displacement with explicit floats to keep TS happy
                displacement = torch.stack(
                    (
                        delta_flow[:, 0].float() / (self.refine_init * float(w1)),
                        delta_flow[:, 1].float() / (self.refine_init * float(h1)),
                    ),
                    dim=1,
                ) * float(ins)

                flow = flow + displacement
                certainty = certainty + delta_certainty

            # record current scale outputs
            corresps[ins].update({"certainty": certainty, "flow": flow})

            # upsample to next finer scale (ins//2) if not final
            if ins != 1:
                hn, wn = sizes[ins // 2]
                flow = Fn.interpolate(flow, size=(hn, wn), mode=self.flow_upsample_mode)
                certainty = Fn.interpolate(certainty, size=(hn, wn), mode=self.flow_upsample_mode)
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()

        return corresps

class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14*16*6, 14*16*6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(
        self,
        batch: Dict[str, Tensor],
        batched: bool = True,
        upsample: bool = False,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        # Ensure TorchScript sees a Dict[str, Tensor]
        batch = torch.jit.annotate(Dict[str, Tensor], batch)

        x_q = batch["im_A"]
        x_s = batch["im_B"]

        if batched:
            X = torch.cat((x_q, x_s), dim=0)              # (2B, C, H, W)
            feat = self.encoder(X, upsample=upsample)     # Dict[int, Tensor] with (2B, ..)

            # Split along batch dimension into query/support dicts
            f_q = torch.jit.annotate(Dict[int, Tensor], {})
            f_s = torch.jit.annotate(Dict[int, Tensor], {})
            for k, v in feat.items():
                a, b = v.chunk(2, dim=0)
                f_q[int(k)] = a
                f_s[int(k)] = b
        else:
            fq = self.encoder(x_q, upsample=upsample)     # Dict[int, Tensor]
            fs = self.encoder(x_s, upsample=upsample)     # Dict[int, Tensor]
            # Rebuild as Dict[int, Tensor] with int keys (TS friendly)
            f_q = torch.jit.annotate(Dict[int, Tensor], {int(k): v for k, v in fq.items()})
            f_s = torch.jit.annotate(Dict[int, Tensor], {int(k): v for k, v in fs.items()})

        return f_q, f_s


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(
        self,
        batch: Dict[str, Tensor],
        batched: bool = True,
        upsample: bool = False,
        scale_factor: float = 1.0,
        # IMPORTANT: pass optional flow/cert directly, not nested in batch
        flow_in: Optional[Tensor] = None,
        cert_in: Optional[Tensor] = None,
    ):
        batch = torch.jit.annotate(Dict[str, Tensor], batch)
        f_q_pyramid, f_s_pyramid = self.extract_backbone_features(batch, batched=batched, upsample=upsample)

        # If you still want to support the eager-only nested dict path, guard it:
        if (not torch.jit.is_scripting()) and ("corresps" in batch):  # eager-only
            cdict = batch["corresps"]  # type: ignore[index]
            if isinstance(cdict, dict):
                flow_in = cdict.get("flow", flow_in)
                cert_in = cdict.get("certainty", cert_in)

        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            flow=flow_in,
            certainty=cert_in,
            scale_factor=scale_factor,
        )
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        flow_in = None
        cert_in = None
        if "corresps" in batch:
            cd = batch["corresps"]       # Dict[str, Tensor]
            if "flow" in cd: flow_in = cd["flow"]
            if "certainty" in cd: cert_in = cd["certainty"]
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            flow=flow_in,
            certainty=cert_in,
            scale_factor=scale_factor,
        )

        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = Fn.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(
        self, x_A, x_B, warp, certainty, return_tuple=True, return_inds=False, max_dist = 0.005, cert_th = 0,
    ):
        x_A_to_B = Fn.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = Fn.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_A_to_B[:, None] > cert_th)
            * (D < max_dist),
            as_tuple=True,
        )

        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)
            
    @torch.jit.ignore
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input)
            check_not_i16(im_A)
            im_A = im_A.convert("RGB")
        else:
            check_rgb(im_A_input)
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input)
            check_not_i16(im_B)
            im_B = im_B.convert("RGB")
        else:
            check_rgb(im_B_input)
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = Fn.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = Fn.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = Fn.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )
                
    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = Fn.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = Fn.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch_ts2.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im

        # ---- add inside RegressionMatcher ----
 
