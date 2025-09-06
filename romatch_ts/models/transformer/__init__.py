import torch
import torch.nn as nn
import torch.nn.functional as F

from romatch_ts.utils.utils import get_grid, get_autocast_params
from .layers.block import Block
from .layers.attention import MemEffAttention
from .dinov2 import vit_large

from typing import Optional, List, Tuple



class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks: nn.Sequential,
        hidden_dim: int,
        # allow either out_dim or num_classes for BC:
        out_dim: Optional[int] = None,
        *,
        num_classes: Optional[int] = None,
        is_classifier: bool = False,
        amp: bool = False,
        pos_enc: bool = True,
        learned_embeddings: bool = False,
        embedding_dim: Optional[int] = None,
        amp_dtype: torch.dtype = torch.float16,
        **_: dict,
    ) -> None:
        super().__init__()
        # Map num_classes -> out_dim if provided
        if out_dim is None and num_classes is not None:
            out_dim = num_classes
        if out_dim is None:
            raise ValueError("TransformerDecoder: one of out_dim or num_classes must be provided")

        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = int(out_dim)
        self._scales: List[int] = [16]
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings

        # ✅ Declare once in __init__, never in forward
        self.learned_pos_embeddings: Optional[torch.Tensor] = None
        if self.learned_embeddings:
            if embedding_dim is None:
                raise ValueError("embedding_dim is required when learned_embeddings=True")
            pe = torch.empty((1, hidden_dim, embedding_dim, embedding_dim))
            nn.init.kaiming_normal_(pe)
            self.learned_pos_embeddings = nn.Parameter(pe)

    def scales(self) -> List[int]:
        return self._scales.copy()


    def forward(
        self,
        gp_posterior: torch.Tensor,   # [B, Cg, H, W]
        features: torch.Tensor,       # [B, Cf, H, W]
        old_stuff: torch.Tensor,      # kept for TS-stable signatures
        new_scale: int,               # unused but TS-safe
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = gp_posterior.shape
        x = torch.cat((gp_posterior, features), dim=1)   # [B, C, H, W]
        C = x.shape[1]
        if C != self.hidden_dim:
            raise RuntimeError(f"Decoder: channel dim {C} != hidden_dim {self.hidden_dim}")

        if torch.jit.is_scripting():
            # ❌ no autocast in scripted graph
            tokens = x.flatten(2).permute(0, 2, 1)       # [B, H*W, C]
            if self.learned_embeddings and self.learned_pos_embeddings is not None:
                pe = F.interpolate(self.learned_pos_embeddings, size=(H, W),
                                mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1).reshape(1, H*W, C).to(dtype=tokens.dtype)
                tokens = tokens + pe
            z = self.blocks(tokens)
            out = self.to_out(z)                          # [B, H*W, out_dim]
        else:
            # ✅ eager-only autocast
            with torch.autocast("cuda", enabled=bool(self.amp), dtype=self.amp_dtype):
                tokens = x.flatten(2).permute(0, 2, 1)
                if self.learned_embeddings and self.learned_pos_embeddings is not None:
                    pe = F.interpolate(self.learned_pos_embeddings, size=(H, W),
                                    mode="bilinear", align_corners=False
                        ).permute(0, 2, 3, 1).reshape(1, H*W, C).to(dtype=tokens.dtype)
                    tokens = tokens + pe
                z = self.blocks(tokens)
                out = self.to_out(z)

        out = out.permute(0, 2, 1).reshape(B, self.out_dim, H, W)
        warp, certainty = out[:, :-1], out[:, -1:]
        return warp, certainty, old_stuff