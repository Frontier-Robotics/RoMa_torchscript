import torch
import torch.nn as nn
import torch.nn.functional as F

from romatch_ts2.utils.utils import get_grid, get_autocast_params
from .layers.block import Block
from .layers.attention import MemEffAttention
from .dinov2 import vit_large

from typing import Tuple, Optional, List, Dict
Tensor = torch.Tensor

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks: nn.Module,
        hidden_dim: int,
        out_dim: int,
        is_classifier: bool = False,
        *args,
        amp: bool = False,
        pos_enc: bool = True,
        learned_embeddings: bool = False,
        embedding_dim: Optional[int] = None,
        amp_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales: List[int] = [16]
        self.is_classifier = is_classifier
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings

        if self.learned_embeddings:
            assert embedding_dim is not None, "embedding_dim must be set when learned_embeddings=True"
            pe = torch.empty((1, hidden_dim, embedding_dim, embedding_dim))
            nn.init.kaiming_normal_(pe)
            self.learned_pos_embeddings = nn.Parameter(pe)
        else:
            # keep attribute defined for TS
            self.register_buffer("learned_pos_embeddings", torch.zeros(1, hidden_dim, 1, 1), persistent=False)

    @torch.jit.export
    def scales(self) -> List[int]:
        # return directly; avoid list.copy() in TS
        return self._scales

    def _build_pos_enc(self, B: int, C: int, H: int, W: int, x: Tensor) -> Tensor:
        # TS-safe positional encoding branch
        if self.learned_embeddings:
            pe = F.interpolate(
                self.learned_pos_embeddings, size=(H, W), mode='bilinear', align_corners=False
            )  # (1, C, H, W)
            return pe.permute(0, 2, 3, 1).reshape(1, H * W, C).to(dtype=x.dtype)
        # no learned pos enc
        return torch.zeros(1, H * W, C, device=x.device, dtype=x.dtype)

    def forward(
        self,
        gp_posterior: Tensor,
        features: Tensor,
        old_stuff: Tensor,
        new_scale: int,  # TS: keep as int
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:

        # Drop autocast in scripted graph; keep a clean eager path if you ever use it.
        if torch.jit.is_scripting() or not self.amp:
            x = torch.cat((gp_posterior, features), dim=1)     # (B, C, H, W)
            B, C, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])

            pos_enc = self._build_pos_enc(B, C, H, W, x)       # (1, HW, C)
            tokens = x.reshape(B, C, H * W).permute(0, 2, 1)   # (B, HW, C)
            tokens = tokens + pos_enc                          # broadcast

            z = self.blocks(tokens)                            # (B, HW, hidden_dim) or similar
            out = self.to_out(z)                               # (B, HW, out_dim)
            out = out.permute(0, 2, 1).reshape(B, self.out_dim, H, W)
            warp, certainty = out[:, :-1], out[:, -1:]
            return warp, certainty, None

        # # (Optional eager-only AMP path, excluded from scripted graph)
        # device_type = 'cuda' if gp_posterior.is_cuda else 'cpu'
        # with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
        #     x = torch.cat((gp_posterior, features), dim=1)
        #     B, C, H, W = x.shape
        #     pos_enc = self._build_pos_enc(int(B), int(C), int(H), int(W), x)
        #     tokens = x.reshape(B, C, H * W).permute(0, 2, 1) + pos_enc
        #     z = self.blocks(tokens)
        #     out = self.to_out(z).permute(0, 2, 1).reshape(B, self.out_dim, H, W)
        #     warp, certainty = out[:, :-1], out[:, -1:]
        #     return warp, certainty, None
