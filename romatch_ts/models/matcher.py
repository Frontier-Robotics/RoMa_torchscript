import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from einops import rearrange
import warnings
from warnings import warn
from PIL import Image

from romatch_ts.utils import get_tuple_transform_ops
from romatch_ts.utils.local_correlation import local_correlation, local_correlation_ts

from romatch_ts.utils.utils import check_rgb, cls_to_flow_refine, get_autocast_params, check_not_i16
from romatch_ts.utils.kde import kde
from torch.jit import is_scripting

#from typing import Dict, Optional, Tuple
from torch import Tensor
from torch.jit import annotate

from typing import Tuple, Optional, Union, Dict, List
#import torch
#import torch.nn.functional as F


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
        self.local_corr_radius: int = int(local_corr_radius) if local_corr_radius is not None else 0
        self.corr_in_other: bool = bool(corr_in_other)
        self.no_im_B_fm: bool = bool(no_im_B_fm)
        self.amp: bool = bool(amp)
        self.concat_logits: bool = bool(concat_logits)
        self.use_cosine_corr: bool = bool(use_cosine_corr)
        self.disable_local_corr_grad: bool = bool(disable_local_corr_grad)
        self.is_classifier: bool = bool(is_classifier)
        self.sample_mode = sample_mode  # string is fine
        self.amp_dtype = amp_dtype
        
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


    def _forward_core(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        flow: torch.Tensor,
        scale_factor: float,
        logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, hs, ws = x.shape
        x_hat = torch.nn.functional.grid_sample(
            y, flow.permute(0, 2, 3, 1), align_corners=False, mode=self.sample_mode
        )

        if self.has_displacement_emb:
            yy, xx = torch.meshgrid(
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                indexing="ij",
            )
            im_A_coords = torch.stack((xx, yy))[None].expand(b, 2, hs, ws)
            in_displacement = flow - im_A_coords
            emb_in_displacement = self.disp_emb(40.0 / 32.0 * scale_factor * in_displacement)

            if self.local_corr_radius > 0:
                if self.corr_in_other:
                    # TS-safe: call TS version when scripting, else Python version
                    local_corr = (
                        local_correlation_ts(x, y, int(self.local_corr_radius), flow=flow)
                        if torch.jit.is_scripting()
                        else local_correlation(
                            x, y, local_radius=self.local_corr_radius, flow=flow, sample_mode=self.sample_mode
                        )
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

        if self.concat_logits:
            if logits is None:
                logits = torch.zeros((b, 1, hs, ws), device=x.device, dtype=x.dtype)
            d = torch.cat((d, logits), dim=1)

        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        flow: torch.Tensor,
        scale_factor: float = 1.0,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # freeze-friendly autocast: branch, don’t pass dynamic enabled=
        use_amp = bool(self.amp) and bool(x.is_cuda) and (not torch.jit.is_scripting())
        if use_amp:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                return self._forward_core(x, y, flow, scale_factor, logits)
        else:
            return self._forward_core(x, y, flow, scale_factor, logits)


    # def forward(
    #     self,
    #     x: Tensor,
    #     y: Tensor,
    #     flow: Tensor,
    #     scale_factor: float = 1.0,
    #     logits: Optional[Tensor] = None,
    # ) -> Tuple[Tensor, Tensor]:

    #     b,c,hs,ws = x.shape

    #     enabled = bool(self.amp) and bool(x.is_cuda)  # x is any tensor you're already using in that scope
    #     with torch.autocast("cuda", enabled=enabled, dtype=self.amp_dtype):         
    #         x_hat = Fnn.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
    #         if self.has_displacement_emb:
    #             im_A_coords = torch.meshgrid(
    #             (
    #                 torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
    #                 torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
    #             ), indexing='ij'
    #             )
    #             im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    #             im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
    #             in_displacement = flow-im_A_coords
    #             emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
    #             if self.local_corr_radius > 0:
    #                 if self.corr_in_other:
    #                     if torch.jit.is_scripting():
    #                         local_corr = local_correlation_ts(x, y, int(self.local_corr_radius), flow=flow)
    #                     else:
    #                         local_corr = local_correlation(
    #                             x, y, local_radius=self.local_corr_radius, flow=flow, sample_mode=self.sample_mode)
    #                 else:
    #                     raise NotImplementedError("Local corr in own frame should not be used.")
    #                 if self.no_im_B_fm:
    #                     x_hat = torch.zeros_like(x)
    #                 d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
    #             else:
    #                 d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
    #         else:
    #             if self.no_im_B_fm:
    #                 x_hat = torch.zeros_like(x)
    #             d = torch.cat((x, x_hat), dim=1)

    #         if self.concat_logits:
    #             if logits is None:
    #                 logits = torch.zeros(
    #                     (x.shape[0], 1, x.shape[2], x.shape[3]),
    #                     device=x.device, dtype=x.dtype
    #                 )
    #             d = torch.cat((d, logits), dim=1)
    #         d = self.block1(d)
    #         d = self.hidden_blocks(d)
    #     d = self.out_conv(d.float())
    #     displacement, certainty = d[:, :-1], d[:, -1:]
    #     return displacement, certainty

class CosKernel(nn.Module):
    def __init__(self, T: float, learn_temperature: bool = False):
        super().__init__()
# -        self.learn_temperature = learn_temperature
# -        if learn_temperature:
# -            self.T = nn.Parameter(Tensor(float(T)))
# -        else:
# -            self.register_buffer("Tbuf", Tensor(float(T)))
        self.learn_temperature: bool = bool(learn_temperature)
        t = torch.tensor(float(T))
        # Always define both for TorchScript
        self.T = nn.Parameter(t.clone(), requires_grad=self.learn_temperature)
        if self.learn_temperature:
            self.register_buffer("Tbuf", torch.tensor(0.0))  # placeholder, unused
        else:
            self.register_buffer("Tbuf", t)
 
    def forward(self, x: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
# -        c = torch.einsum("bnd,bmd->bnm", x, y) / (x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps)
# -        T = (self.T.abs() + 0.01) if self.learn_temperature else self.Tbuf
        # TS-friendly cosine similarity + temperature
        dots = torch.matmul(x, y.transpose(1, 2))                              # (b,n,m)
        x_n = torch.norm(x, p=2, dim=-1, keepdim=True)                         # (b,n,1)
        y_n = torch.norm(y, p=2, dim=-1, keepdim=True)                         # (b,m,1)
        denom = x_n * y_n.transpose(1, 2) + eps                                # (b,n,m)
        c = dots / denom
        T = (self.T.abs() + 0.01) if self.learn_temperature else self.Tbuf
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
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = int(covar_size)
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features
        self.amp = amp
        self.amp_dtype = amp_dtype

    def get_local_cov(self, cov):
        K: int = int(self.covar_size)
        K2: int = K * K

        b, h, w, _, _ = cov.shape
        hw = h * w
        pad: int = K // 2

        # pad last 4 dims
        cov = Fnn.pad(cov, 4 * (pad,))  # (b, h+K-1, w+K-1, h+K-1, w+K-1)

        # window deltas (relative)
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-pad, pad + 1, device=cov.device),
                torch.arange(-pad, pad + 1, device=cov.device),
                indexing='ij'
            ),
            dim=-1,  # [K,K,2]
        )

        # base positions (absolute) inside the padded plane
        row_base = torch.arange(pad, pad + h, device=cov.device)
        col_base = torch.arange(pad, pad + w, device=cov.device)
        positions = torch.stack(torch.meshgrid(row_base, col_base, indexing='ij'), dim=-1)  # [h,w,2]

        # per-pixel window coordinates
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]  # [h,w,K,K,2]

        points = torch.arange(hw, device=cov.device, dtype=torch.long)[:, None].expand(hw, K2)

        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].reshape(-1),
            neighbours[..., 1].reshape(-1),
        ].reshape(b, h, w, K2)

        return local_cov
    def _flatten_hw(self, x: Tensor) -> Tensor:
        # [b, d, h, w] -> [b, h*w, d]
        b, d, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, d)

    def _unflatten_hw(self, x: Tensor, h: int, w: int) -> Tensor:
        # [b, h*w, d] -> [b, d, h, w]
        b, n, d = x.shape
        return x.reshape(b, h, w, d).permute(0, 3, 1, 2)

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)
        coarse_coords = coarse_coords.unsqueeze(0).expand(b, h, w, 2)
        coarse_coords = coarse_coords.permute(0, 3, 1, 2)
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords


    # def forward(self, x: Tensor, y: Tensor) -> Tensor:
    #     b, c, h1, w1 = x.shape
    #     _, _, h2, w2 = y.shape

    #     f = self.get_pos_enc(y)                # [b, d, h2, w2]
    #     _, d, _, _ = f.shape

    #     x = self._flatten_hw(x.float())        # [b, h1*w1, c]
    #     y = self._flatten_hw(y.float())        # [b, h2*w2, c]
    #     f = self._flatten_hw(f)                # [b, h2*w2, d]

    #     # autocast reduces memory a LOT; TS-safe guard
    #     enabled = bool(self.amp) and bool(x.is_cuda)  # x is any tensor you're already using in that scope
    #     with torch.autocast("cuda", enabled=enabled, dtype=self.amp_dtype):

    #     #dev, ena, dt = get_autocast_params(y.device, enabled=True, dtype=torch.float16)
    #     #with torch.autocast(dev, enabled=(ena and not is_scripting()), dtype=dt):
    #         K_yy = self.K(y, y)                # [b, n2, n2]
    #         K_xy = self.K(x, y)                # [b, n1, n2]

    #         sigma = self.sigma_noise * torch.eye(h2 * w2, device=y.device, dtype=K_yy.dtype)[None]
    #         K_yy_plus = K_yy + sigma
    #         I = torch.eye(K_yy_plus.size(-1), device=K_yy_plus.device, dtype=K_yy_plus.dtype)
    #         I = I.expand(K_yy_plus.size(0), -1, -1)
    #         K_yy_inv = torch.linalg.solve(K_yy_plus, I)

    #         mu_x = K_xy.matmul(K_yy_inv.matmul(f))         # [b, n1, d]
    #         mu_x = self._unflatten_hw(mu_x, h1, w1)        # [b, d, h1, w1]

    #         if not self.no_cov:
    #             K_xx = self.K(x, x)                        # compute only when needed
    #             K_yx = K_xy.transpose(1, 2)
    #             cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
    #             cov_x = cov_x.reshape(b, h1, w1, h1, w1)
    #             local_cov_x = self.get_local_cov(cov_x).permute(0, 3, 1, 2)
    #             return torch.cat((mu_x, local_cov_x), dim=1)

    #         return mu_x
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h1, w1 = x.shape
        _, _, h2, w2 = y.shape

        f = self.get_pos_enc(y)              # [b, d, h2, w2]
        _, d, _, _ = f.shape

        xf = self._flatten_hw(x.float())     # [b, h1*w1, c]
        yf = self._flatten_hw(y.float())     # [b, h2*w2, c]
        ff = self._flatten_hw(f)             # [b, h2*w2, d]

        # branch around autocast (no dynamic `enabled=` arg)
        use_amp = bool(getattr(self, "amp", False)) and bool(x.is_cuda) and (not torch.jit.is_scripting())
        if use_amp:
            with torch.autocast("cuda", dtype=self.amp_dtype):
                K_yy = self.K(yf, yf)                    # [b, n2, n2]
                K_xy = self.K(xf, yf)                    # [b, n1, n2]
                sigma = self.sigma_noise * torch.eye(h2 * w2, device=y.device, dtype=K_yy.dtype)[None]
                K_yy_plus = K_yy + sigma
                I = torch.eye(K_yy_plus.size(-1), device=K_yy_plus.device, dtype=K_yy_plus.dtype)
                I = I.expand(K_yy_plus.size(0), -1, -1)
                K_yy_inv = torch.linalg.solve(K_yy_plus, I)
                mu_x = K_xy.matmul(K_yy_inv.matmul(ff))   # [b, n1, d]
        else:
            K_yy = self.K(yf, yf)
            K_xy = self.K(xf, yf)
            sigma = self.sigma_noise * torch.eye(h2 * w2, device=y.device, dtype=K_yy.dtype)[None]
            K_yy_plus = K_yy + sigma
            I = torch.eye(K_yy_plus.size(-1), device=K_yy_plus.device, dtype=K_yy_plus.dtype)
            I = I.expand(K_yy_plus.size(0), -1, -1)
            K_yy_inv = torch.linalg.solve(K_yy_plus, I)
            mu_x = K_xy.matmul(K_yy_inv.matmul(ff))

        mu_x = self._unflatten_hw(mu_x, h1, w1)          # [b, d, h1, w1]

        if not self.no_cov:
            K_xx = self.K(xf, yf if False else xf)       # compute only when needed
            K_yx = K_xy.transpose(1, 2)
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = cov_x.reshape(b, h1, w1, h1, w1)
            local_cov_x = self.get_local_cov(cov_x).permute(0, 3, 1, 2)
            return torch.cat((mu_x, local_cov_x), dim=1)

        return mu_x

class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder: nn.Module,
        gps: nn.ModuleDict,
        proj: nn.ModuleDict,
        conv_refiner: nn.ModuleDict,
        detach: bool = False,
        scales: str = "all",
        pos_embeddings: Optional[Dict[str, Tensor]] = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0,
        displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.detach = detach
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype

        # ---- keep scales as ints
        if scales == "all":
            self.scales: List[int] = [32, 16, 8, 4, 2, 1]
        else:
            self.scales = [int(s) for s in scales]

        # ---- optional pos embeddings (unchanged)
        if pos_embeddings is not None:
            for k, v in pos_embeddings.items():
                self.register_buffer(f"pos_emb_{k}", v, persistent=True)
        self.has_pos_embeddings: bool = pos_embeddings is not None

        # ===== NEW: TS-friendly fast lookup structures =====
        # PROJ
        self.proj_list = nn.ModuleList()
        self.proj_names: List[str] = []
        for name, mod in proj.items():
            self.proj_list.append(mod)
            self.proj_names.append(name)

        # GPS
        self.gps_list = nn.ModuleList()
        self.gps_names: List[str] = []
        for name, mod in gps.items():
            self.gps_list.append(mod)
            self.gps_names.append(name)

        # REFINER
        self.ref_list = nn.ModuleList()
        self.ref_names: List[str] = []
        for name, mod in conv_refiner.items():
            self.ref_list.append(mod)
            self.ref_names.append(name)

    
# in class Decoder

    def get_placeholder_flow(self, b: int, h: int, w: int, ref: torch.Tensor) -> torch.Tensor:
        """Identity sampling grid in [-1, 1] with same device/dtype as ref."""
        dev = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32

        xs = torch.linspace(-1.0 + 1.0/float(w), 1.0 - 1.0/float(w), w, device=dev, dtype=dtype)
        ys = torch.linspace(-1.0 + 1.0/float(h), 1.0 - 1.0/float(h), h, device=dev, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')        # [h,w], [h,w]
        grid = torch.stack((gx, gy), dim=0)                   # [2,h,w]
        grid = grid.unsqueeze(0).expand(b, 2, h, w)           # [B,2,h,w]
        return grid

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

        # TS-stable dict types
        f1 = annotate(Dict[int, torch.Tensor], f1)
        f2 = annotate(Dict[int, torch.Tensor], f2)

        # Spatial sizes by scale (int keys)
        sizes: Dict[int, List[int]] = {}
        for k in self.scales:  # self.scales: List[int]
            if k in f1:
                sizes[k] = [int(f1[k].shape[-2]), int(f1[k].shape[-1])]

        all_scales: List[int] = self.scales if not upsample else [8, 4, 2, 1]
        coarsest_scale: int = all_scales[0]

        # Which scales the transformer runs on (e.g., ["16"] -> [16])
        coarse_scales: List[int] = [int(s) for s in self.embedding_decoder.scales()]

        # Reference resolution for NDC scaling of displacement
        ref_scale = 1 if 1 in sizes else min(sizes.keys())
        h, w = sizes[ref_scale]
        b = int(f1[ref_scale].shape[0])

        # State carried across transformer blocks @ coarsest
        Hc, Wc = sizes[coarsest_scale]
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, Hc, Wc,
            device=f1[coarsest_scale].device
        )

        # Initialize flow / certainty at coarsest
        if not upsample:
            ref = f1[coarsest_scale]                  # a Tensor
            flow = self.get_placeholder_flow(b, Hc, Wc, ref)
            certainty = torch.zeros((b, 1, Hc, Wc), device=ref.device, dtype=flow.dtype)
        else:
            if flow is None:
                raise RuntimeError("Decoder.forward: upsample=True requires a coarse 'flow'.")
            if certainty is None:
                raise RuntimeError("Decoder.forward: upsample=True requires a coarse 'certainty'.")
            flow = Fnn.interpolate(flow, size=(Hc, Wc), mode="bilinear", align_corners=False)
            certainty = Fnn.interpolate(certainty, size=(Hc, Wc), mode="bilinear", align_corners=False)

        corresps = annotate(Dict[int, Dict[str, torch.Tensor]], {})

        for ins in all_scales:
            s = str(ins)
            inner = annotate(Dict[str, torch.Tensor], {})
            corresps[ins] = inner

            f1_s, f2_s = f1[ins], f2[ins]
            Hs, Ws = int(f1_s.shape[-2]), int(f1_s.shape[-1])

            # ---- Optional projection (freeze-safe AMP branch) ----
            for idx, mod in enumerate(self.proj_list):
                if self.proj_names[idx] == s:
                    use_amp = (f1_s.is_cuda and (not torch.jit.is_scripting()))
                    if use_amp:
                        with torch.autocast("cuda", dtype=self.amp_dtype):
                            f1_s = mod(f1_s)
                            f2_s = mod(f2_s)
                    else:
                        f1_s = mod(f1_s.to(torch.float32))
                        f2_s = mod(f2_s.to(torch.float32))

            # ---- GP + Transformer only on coarse_scales ----
            if ins in coarse_scales:
                if old_stuff.shape[-2:] != (Hs, Ws):
                    old_stuff = Fnn.interpolate(old_stuff, size=(Hs, Ws), mode="bilinear", align_corners=False)

                found_gp = False
                gp_posterior = f1_s  # placeholder type
                for idx, gp in enumerate(self.gps_list):
                    if self.gps_names[idx] == s:
                        gp_posterior = gp(f1_s, f2_s)
                        found_gp = True
                if not found_gp:
                    gp_posterior = self.gps_list[0](f1_s, f2_s)

                gm_out, cert_logits, old_stuff = self.embedding_decoder(gp_posterior, f1_s, old_stuff, ins)

                if self.embedding_decoder.is_classifier:
                    # TS-safe sanity on channel count (should be square: side^2)
                    C = int(gm_out.size(1))
                    side = int(math.sqrt(C))
                    if side * side != C:
                        raise RuntimeError(f"warp head channels must be square (got {C}).")

                    flow = cls_to_flow_refine(gm_out).permute(0, 3, 1, 2)  # [B,2,Hs,Ws]
                    if self.training:
                        inner["gm_cls"] = gm_out
                        inner["gm_certainty"] = cert_logits
                else:
                    if self.training:
                        inner["gm_flow"] = gm_out
                        inner["gm_certainty"] = cert_logits
                    flow = gm_out.detach()

            # Ensure certainty matches current scale
            if certainty.shape[-2:] != (Hs, Ws):
                certainty = Fnn.interpolate(certainty, size=(Hs, Ws), mode="bilinear", align_corners=False)

            # ---- Conv refiner (present at multiple scales) ----
            for idx, mod in enumerate(self.ref_list):
                if self.ref_names[idx] == s:
                    if flow.shape[-2:] != (Hs, Ws):
                        flow = Fnn.interpolate(flow, size=(Hs, Ws), mode=self.flow_upsample_mode, align_corners=False)
                    if certainty.shape[-2:] != (Hs, Ws):
                        certainty = Fnn.interpolate(certainty, size=(Hs, Ws), mode=self.flow_upsample_mode, align_corners=False)

                    if self.training:
                        inner["flow_pre_delta"] = flow

                    delta_flow, delta_certainty = mod(
                        f1_s, f2_s, flow, scale_factor=scale_factor, logits=certainty
                    )
                    disp = torch.stack(
                        (
                            delta_flow[:, 0].float() / (self.refine_init * w),
                            delta_flow[:, 1].float() / (self.refine_init * h),
                        ),
                        dim=1,
                    )
                    flow = flow + ins * disp
                    certainty = certainty + delta_certainty

            inner["flow"] = flow
            inner["certainty"] = certainty

            # Prep for next finer scale
            if ins != 1:
                next_ins = ins // 2
                if next_ins in f1:
                    next_H, next_W = int(f1[next_ins].shape[-2]), int(f1[next_ins].shape[-1])
                    flow = Fnn.interpolate(flow, size=(next_H, next_W), mode=self.flow_upsample_mode, align_corners=False)
                    certainty = Fnn.interpolate(certainty, size=(next_H, next_W), mode=self.flow_upsample_mode, align_corners=False)
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
        if not torch.jit.is_scripting():
            self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)  # python-only
        #self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
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
    
    @torch.jit.export
    def extract_backbone_features(
        self,
        batch: Dict[str, Tensor],
        batched: bool = True,
        upsample: bool = False,
    ) -> Dict[int, Tensor]:
        # TS now knows 'batch' is Dict[str, Tensor]
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        # Always return a Dict for TS (avoid union types)
        X = torch.cat((x_q, x_s), dim=0)
        feats: Dict[int, Tensor] = self.encoder(X, upsample=upsample)
        return feats

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

    #def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
    
    @torch.jit.export
    def forward(
        self,
        batch: Dict[str, Tensor],
        upsample: bool = False,
        scale_factor: float = 1.0,
        flow: Optional[Tensor] = None,
        certainty: Optional[Tensor] = None,
    ) -> Dict[int, Dict[str, Tensor]]:
        batch = annotate(Dict[str, Tensor], batch)
        feature_pyramid = self.extract_backbone_features(batch, batched=True, upsample=upsample)
        f_q_pyramid = {s: f.chunk(2)[0] for s, f in feature_pyramid.items()}
        f_s_pyramid = {s: f.chunk(2)[1] for s, f in feature_pyramid.items()}

        corresps: Dict[int, Dict[str, Tensor]] = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            flow=flow,
            certainty=certainty,
            scale_factor=scale_factor,
        )
        return corresps


    @torch.jit.export
    def forward_ts(self, im_A: Tensor, im_B: Tensor, upsample: bool = True
                ) -> Tuple[Tensor, Tensor]:
        # ---- constants / buffers (built on the fly for device safety) ----
        mean = torch.tensor([0.485, 0.456, 0.406], device=im_A.device, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=im_A.device, dtype=torch.float32).view(1, 3, 1, 1)

        # ---- input to float32, optional 0..1 scaling if looks like 8-bit ----
        A = im_A.to(torch.float32)
        B = im_B.to(torch.float32)
        if float(A.max().item()) > 1.5:
            A = A / 255.0
        if float(B.max().item()) > 1.5:
            B = B / 255.0

        # ---- 1) coarse pass at (h_resized, w_resized) ----
        Hc: int = int(self.h_resized)
        Wc: int = int(self.w_resized)

        A_c = Fnn.interpolate(A, size=(Hc, Wc), mode="bilinear", align_corners=False)
        B_c = Fnn.interpolate(B, size=(Hc, Wc), mode="bilinear", align_corners=False)
        A_c = (A_c - mean) / std
        B_c = (B_c - mean) / std

        batch0: Dict[str, Tensor] = {"im_A": A_c, "im_B": B_c}
        none_t: Optional[Tensor] = None
        coarse = self.forward(batch0, upsample=False, scale_factor=1.0, flow=none_t, certainty=none_t)

        finest = coarse[1]
        flow   = finest["flow"]        # [B,2,Hc,Wc]
        cert   = finest["certainty"]   # [B,1,Hc,Wc]

        # prepare low-res certainty attenuation (matches match())
        low_res_cert: Optional[Tensor] = None
        if self.attenuate_cert and self.upsample_preds and upsample:
            if 16 in coarse:
                Hu_att, Wu_att = self.upsample_res
                low_res_cert = Fnn.interpolate(coarse[16]["certainty"], size=(Hu_att, Wu_att),
                                            mode="bilinear", align_corners=False)
                # keep negative logits only, scale by factor
                factor = 0.5
                zero = torch.tensor(0.0, device=low_res_cert.device, dtype=low_res_cert.dtype)
                low_res_cert = factor * (low_res_cert * (low_res_cert < zero))

        # ---- 2) optional refine at upsample_res (CNN only; no ViT) ----
        if self.upsample_preds and upsample:
            Hu, Wu = self.upsample_res
            A_hi = Fnn.interpolate(A, size=(Hu, Wu), mode="bilinear", align_corners=False)
            B_hi = Fnn.interpolate(B, size=(Hu, Wu), mode="bilinear", align_corners=False)
            A_hi = (A_hi - mean) / std
            B_hi = (B_hi - mean) / std

            scale = (float(Hu * Wu) / float(self.h_resized * self.w_resized)) ** 0.5
            batch1: Dict[str, Tensor] = {"im_A": A_hi, "im_B": B_hi}
            hi = self.forward(batch1, upsample=True, scale_factor=scale, flow=flow, certainty=cert)
            finest = hi[1]
            flow   = finest["flow"]        # [B,2,Hu,Wu]
            cert   = finest["certainty"]   # [B,1,Hu,Wu]
            if low_res_cert is not None:
                cert = cert - low_res_cert

        # ---- 3) build warp + OOB masking/clamp like match() ----
        Bsz, _, H, W = flow.shape
        xs = torch.linspace(-1.0 + 1.0/float(W), 1.0 - 1.0/float(W), W, device=flow.device, dtype=flow.dtype)
        ys = torch.linspace(-1.0 + 1.0/float(H), 1.0 - 1.0/float(H), H, device=flow.device, dtype=flow.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(Bsz, H, W, 2)

        flow_hw = flow.permute(0, 2, 3, 1)
        if (flow_hw.abs() > 1).any():
            wrong = (flow_hw.abs() > 1).sum(dim=-1) > 0
            cert = cert.clone()
            cert[wrong[:, None]] = 0.0
            flow_hw = torch.clamp(flow_hw, -1.0, 1.0)

        warp = torch.cat((coords, flow_hw), dim=-1)  # [B,H,W,4]
        cert = cert.sigmoid()
        return warp, cert


    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
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
        coords_fb = Fnn.grid_sample(
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
        x_A_to_B = Fnn.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = Fnn.grid_sample(
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
                low_res_certainty = Fnn.interpolate(
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
                im_A_to_im_B = Fnn.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = Fnn.interpolate(
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
        if not isinstance(im_A, Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (Tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (Tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = Fnn.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = Fnn.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch_ts.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
