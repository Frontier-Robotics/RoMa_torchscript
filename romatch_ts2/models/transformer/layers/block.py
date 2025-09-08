# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

# at top
import os
import logging
from typing import Callable, List, Any, Tuple, Dict, Optional

import torch
from torch import nn, Tensor

from .attention import Attention  # <- keep plain Attention
# from .attention import MemEffAttention  # we'll gate this behind a flag
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")




class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        # attn_class: Callable[..., nn.Module] = Attention,  # <- remove kw default to keep TS simpler
        ffn_layer: Callable[..., nn.Module] = Mlp,
        attn_class: Callable[..., nn.Module] = Attention,   # <-- default to TS-friendly Attention
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def attn_residual(self, x: Tensor) -> Tensor:
        return self.ls1(self.attn(self.norm1(x)))

    def ffn_residual(self, x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))

    # TorchScript-safe SD helper: which==0 -> attn branch, which==1 -> ffn branch
    @torch.jit.export
    def drop_add_residual_sd(self, x: Tensor, which: int) -> Tensor:
        b, n, d = x.shape
        sdr = self.sample_drop_ratio
        sample_subset_size = max(int(b * (1.0 - sdr)), 1)
        brange = torch.randperm(b, device=x.device)[:sample_subset_size]
        x_subset = x.index_select(0, brange)

        if which == 0:
            residual = self.attn_residual(x_subset)
        else:
            residual = self.ffn_residual(x_subset)

        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        scale = float(b) / float(sample_subset_size)
        x_plus = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=scale)
        return x_plus.view_as(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.sample_drop_ratio > 0.1:
            x = self.drop_add_residual_sd(x, 0)  # attn
            x = self.drop_add_residual_sd(x, 1)  # ffn
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(self.attn_residual(x))
            x = x + self.drop_path1(self.ffn_residual(x))  # FIXME kept
        else:
            x = x + self.attn_residual(x)
            x = x + self.ffn_residual(x)
        return x



# def get_branges_scales(x, sample_drop_ratio=0.0):
#     b, n, d = x.shape
#     sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
#     brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
#     residual_scale_factor = b / sample_subset_size
#     return brange, residual_scale_factor


# def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
#     if scaling_vector is None:
#         x_flat = x.flatten(1)
#         residual = residual.flatten(1)
#         x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
#     else:
#         x_plus_residual = scaled_index_add(
#             x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
#         )
#     return x_plus_residual


# attn_bias_cache: Dict[Tuple, Any] = {}


# def get_attn_bias_and_cat(x_list, branges=None):
#     """
#     this will perform the index select, cat the tensors, and provide the attn_bias from cache
#     """
#     batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
#     all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
#     if all_shapes not in attn_bias_cache.keys():
#         seqlens = []
#         for b, x in zip(batch_sizes, x_list):
#             for _ in range(b):
#                 seqlens.append(x.shape[1])
#         attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
#         attn_bias._batch_sizes = batch_sizes
#         attn_bias_cache[all_shapes] = attn_bias

#     if branges is not None:
#         cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
#     else:
#         tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
#         cat_tensors = torch.cat(tensors_bs1, dim=1)

#     return attn_bias_cache[all_shapes], cat_tensors




# class NestedTensorBlock(Block):
#     def attn_residual_nested(self, x: Tensor, attn_bias) -> Tensor:
#         return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

#     def ffn_residual_nested(self, x: Tensor) -> Tensor:
#         return self.ls2(self.mlp(self.norm2(x)))

#     @torch.jit.export
#     def drop_add_residual_sd_list(
#         self,
#         x_list: List[Tensor],
#         which: int,              # 0 = attn, 1 = ffn
#         attn_bias_any=None       # bias needed for attn pass; unused for ffn
#     ) -> List[Tensor]:
#         # 1) pick subsets per-tensor
#         branges: List[Tensor] = []
#         scales: List[float] = []
#         for x in x_list:
#             b = x.shape[0]
#             sample_subset_size = max(int(b * (1.0 - self.sample_drop_ratio)), 1)
#             brange = torch.randperm(b, device=x.device)[:sample_subset_size]
#             branges.append(brange)
#             scales.append(float(b) / float(sample_subset_size))

#         # 2) build bias and cat inputs (keeps behavior when using MemEffAttention)
#         attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

#         # 3) compute residual on concatenated subset and split back
#         if which == 0:
#             # attention residual needs bias
#             residual_cat = self.attn(self.norm1(x_cat), attn_bias=attn_bias)
#             if isinstance(self.ls1, LayerScale):
#                 residual_cat = self.ls1(residual_cat)
#         else:
#             residual_cat = self.mlp(self.norm2(x_cat))
#             if isinstance(self.ls2, LayerScale):
#                 residual_cat = self.ls2(residual_cat)

#         residual_list = attn_bias.split(residual_cat)  # split by original batch sizes

#         # 4) add residuals back per tensor
#         out: List[Tensor] = []
#         for x, brange, residual, scale in zip(x_list, branges, residual_list, scales):
#             x_flat = x.flatten(1)
#             r_flat = residual.to(dtype=x.dtype).flatten(1)
#             x_plus = torch.index_add(x_flat, 0, brange, r_flat, alpha=scale)
#             out.append(x_plus.view_as(x))
#         return out

#     def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
#         assert isinstance(self.attn, MemEffAttention)
#         if self.training and self.sample_drop_ratio > 0.0:
#             x_list = self.drop_add_residual_sd_list(x_list, which=0)  # attn
#             x_list = self.drop_add_residual_sd_list(x_list, which=1)  # ffn
#             return x_list
#         else:
#             attn_bias, x = get_attn_bias_and_cat(x_list)
#             x = x + self.attn_residual_nested(x, attn_bias=attn_bias)
#             x = x + self.ffn_residual_nested(x)
#             return attn_bias.split(x)

#     def forward(self, x_or_x_list):
#         if isinstance(x_or_x_list, Tensor):
#             return super().forward(x_or_x_list)
#         elif isinstance(x_or_x_list, list):
#             # requires xFormers path as before
#             return self.forward_nested(x_or_x_list)
#         else:
#             raise AssertionError