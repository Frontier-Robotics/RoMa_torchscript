# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor  # <-- needed for type annotations
from typing import Optional
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, use_sdpa: bool = True):
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # TorchScript-safe backend choice
        self.use_sdpa: bool = bool(use_sdpa)

    def _shape_qkv(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)
        return q, k, v

    def _attn_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # dropout only during training
        p = float(self.attn_drop.p) if self.training else 0.0
        # PyTorch native SDPA (scriptable)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=False)
        return out  # (B, H, N, D)

    def _attn_naive(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Manual matmul softmax path, also scriptable
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) if self.training else attn
        out = torch.matmul(attn, v)                                   # (B,H,N,D)
        return out

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TorchScript-friendly: no globals, no lambdas
        # NOTE: attn_bias not supported in the SDPA path here; keep None
        if attn_bias is not None:
            # If you truly need bias, convert it to a dense (B,H,N,N) mask and
            # pass via 'attn_mask' argument to sdpa, but keep it out for TS simplicity.
            raise RuntimeError("attn_bias unsupported without xFormers")

        q, k, v = self._shape_qkv(x)
        if self.use_sdpa:
            y = self._attn_sdpa(q, k, v)
        else:
            y = self._attn_naive(q, k, v)

        B, H, N, D = y.shape
        y = y.transpose(1, 2).reshape(B, N, H * D)  # (B,N,C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class MemEffAttention(Attention):
    """
    Memory-efficient attention that:
      - uses xFormers memory_efficient_attention in eager mode (if available),
      - otherwise (and for TorchScript/tracing/scripting) uses PyTorch SDPA:
            torch.nn.functional.scaled_dot_product_attention
    Notes:
      * `attn_bias` is only supported in the xFormers path, matching prior behavior.
      * SDPA path ignores `attn_bias` (asserts None), matching your previous fallback.
    """

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape

        # prepare QKV shaped for both paths
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # TorchScript / tracing / scripting OR xFormers not available:
        # -> use native SDPA (fast on recent PyTorch/CUDA)


        # q,k,v as (B, H, N, Hd)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, N, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scale outside to match your Attention scaling
        q = q * self.scale

        # dropout only in training
        drop_p = self.attn_drop.p if self.training else 0.0

        # SDPA: expects (..., L, E) with batch dims in front; (B, H, N, Hd) is fine
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=False,
        )  # (B, H, N, Hd)

        x = x.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        # # Eager path with xFormers
        # q, k, v = unbind(qkv, 2)  # each is (B, N, H, Hd)
        # # xFormers expects (B, N, H, Hd) and handles scaling internally if desired.
        # # We keep your explicit scaling for parity with the softmax path.
        # q = q * self.scale

        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)  # (B, N, H, Hd)
        # x = x.reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x








