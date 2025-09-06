# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

from typing import List 
logger = logging.getLogger("dinov2")


# romatch_ts/models/transformer/layers/block.py
from typing import Callable, List, Any, Tuple, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:
    from xformers.ops import fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# ---- add this: keep the cache and helpers out of TorchScript
if not XFORMERS_AVAILABLE:
    attn_bias_cache = {}  # type: ignore[var-annotated]

    @torch.jit.ignore
    def get_attn_bias_and_cat(x_list, branges=None):
        raise RuntimeError("Nested-tensor path requires xFormers; disabled for TorchScript.")

    @torch.jit.ignore
    def drop_add_residual_stochastic_depth_list(*args, **kwargs):
        raise RuntimeError("Nested-tensor path requires xFormers; disabled for TorchScript.")



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
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
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

    def forward(self, x: Tensor) -> Tensor:
            if self.training and self.sample_drop_ratio > 0.1:
                # ---- attention residual with stochastic-depth (inline, no closures)
                b, n, d = x.shape
                keep = max(int(b * (1 - self.sample_drop_ratio)), 1)
                idx = torch.randperm(b, device=x.device)[:keep]
                x_sub = x[idx]
                res = self.ls1(self.attn(self.norm1(x_sub)))
                x_flat = x.flatten(1)
                res_flat = res.flatten(1)
                scale = b / keep
                x = torch.index_add(x_flat, 0, idx, res_flat.to(dtype=x.dtype), alpha=scale).view_as(x)

                # ---- ffn residual with stochastic-depth (inline)
                b, n, d = x.shape
                keep = max(int(b * (1 - self.sample_drop_ratio)), 1)
                idx = torch.randperm(b, device=x.device)[:keep]
                x_sub = x[idx]
                res = self.ls2(self.mlp(self.norm2(x_sub)))
                x_flat = x.flatten(1)
                res_flat = res.flatten(1)
                scale = b / keep
                x = torch.index_add(x_flat, 0, idx, res_flat.to(dtype=x.dtype), alpha=scale).view_as(x)
                return x

            if self.training and self.sample_drop_ratio > 0.0:
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path1(self.ls2(self.mlp(self.norm2(x))))  # same drop path as original
                return x

            # eval / no drop-path
            x = x + self.ls1(self.attn(self.norm1(x)))
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


# class NestedTensorBlock(Block):
#     def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
#         # TorchScript-friendly version; no nested defs
#         assert isinstance(self.attn, MemEffAttention)
#         if self.training and self.sample_drop_ratio > 0.0:
#             # Simple per-tensor path (no stochastic subset tricks during scripting)
#             out_list: List[Tensor] = []
#             for x in x_list:
#                 x = x + self.ls1(self.attn(self.norm1(x)))  # attn_bias not used in per-tensor path
#                 x = x + self.ls2(self.mlp(self.norm2(x)))
#                 out_list.append(x)
#             return out_list
#         else:
#             # Fast batched path with xFormers bias (no inner defs)
#             attn_bias, x = get_attn_bias_and_cat(x_list)
#             x = x + self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
#             x = x + self.ls2(self.mlp(self.norm2(x)))
#             return attn_bias.split(x)


class NestedTensorBlock(Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            # TorchScript-safe: call base class directly, not super()
            return Block.forward(self, x_or_x_list)
        # keep nested-tensor path disabled for JIT
        raise RuntimeError("Nested-tensor path is disabled for TorchScript on this build.")
