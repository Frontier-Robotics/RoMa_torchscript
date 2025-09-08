from typing import Optional, Union, Dict, Optional
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from romatch_ts2.utils.utils import get_autocast_params
from .transformer import vit_large

from typing import Dict
from torch import Tensor

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res=False, weights=None,
                 dilation=None, freeze_bn=True, anti_aliased=False,
                 early_exit=False, amp=False, amp_dtype=torch.float16):
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]

        if weights is not None:
            self.net = tvm.resnet50(weights=weights, replace_stride_with_dilation=dilation)
        else:
            self.net = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=dilation)

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor, upsample: bool = False) -> Dict[int, Tensor]:
        net = self.net

        # TorchScript path: NO autocast inside the scripted graph
        if torch.jit.is_scripting() or not self.amp:
            feats: Dict[int, Tensor] = {1: x}
            x = net.conv1(x); x = net.bn1(x); x = net.relu(x); feats[2] = x
            x = net.maxpool(x)
            x = net.layer1(x); feats[4] = x
            x = net.layer2(x); feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x); feats[16] = x
            x = net.layer4(x); feats[32] = x
            return feats

        # Eager-only AMP branch (won’t be included in the scripted graph)
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            feats: Dict[int, Tensor] = {1: x}
            x = net.conv1(x); x = net.bn1(x); x = net.relu(x); feats[2] = x
            x = net.maxpool(x)
            x = net.layer1(x); feats[4] = x
            x = net.layer2(x); feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x); feats[16] = x
            x = net.layer4(x); feats[32] = x
            return feats

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp=False, amp_dtype=torch.float16):
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor, upsample: bool = False) -> Dict[int, Tensor]:
        # Scripted path: no autocast
        if torch.jit.is_scripting() or not self.amp:
            feats: Dict[int, Tensor] = {}
            scale: int = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale *= 2
                x = layer(x)
            feats[scale] = x
            return feats

        # Eager-only AMP path
        device_type = 'cuda' if x.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            feats: Dict[int, Tensor] = {}
            scale: int = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale *= 2
                x = layer(x)
            feats[scale] = x
            return feats


class CNNandDinov2(nn.Module):
    def __init__(
        self,
        cnn_kwargs: Dict = None,
        amp: bool = False,
        use_vgg: bool = False,
        dinov2_weights: Dict[str, torch.Tensor] = None,
        amp_dtype: torch.dtype = torch.float16,
        dinov2_img_size: int = 560,   # kept for API symmetry
    ):
        super().__init__()


        # --- ViT (single module, TS-friendly) ---
        vit_kwargs = dict(
            img_size=518,      # matches the DINOv2-L/14 ckpt; pos embed will interpolate at runtime
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )
        
        self.dinov2_vitl14: nn.ModuleList = nn.ModuleList([vit_large(**vit_kwargs).eval()])

        # load DINOv2-L/14 weights into index 0
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu"
            )
        self.dinov2_vitl14[0].load_state_dict(dinov2_weights, strict=True)

        # --- CNN backbone ---
        cnn_kwargs = {} if cnn_kwargs is None else cnn_kwargs
        self.cnn = VGG19(**cnn_kwargs) if use_vgg else ResNet50(**cnn_kwargs)

        # misc
        self.amp = amp
        self.amp_dtype = amp_dtype

        if amp_dtype is not None:
            self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(amp_dtype)

        # Optional: freeze ViT just like upstream
        for p in self.dinov2_vitl14[0].parameters():
            p.requires_grad = False
        self.dinov2_vitl14[0].eval()

    def train(self, mode: bool = True):
        # keep ViT in eval (frozen), toggle only CNN
        self.cnn.train(mode)
        self.dinov2_vitl14[0].eval()
        return self


    @torch.no_grad()
    def forward(self, x: torch.Tensor, upsample: bool = False) -> Dict[int, torch.Tensor]:
        # dino = self.dinov2_vitl14[0]
        # if dino.device != x.device:
        #     self.dinov2_vitl14[0] = dino.to(x.device)
        B, C, H, W = x.shape
        feature_pyramid: Dict[int, torch.Tensor] = self.cnn(x)  # must be Dict[int, Tensor]

        if not upsample:
            x_vit = x.to(self.amp_dtype)
            dino = self.dinov2_vitl14[0]
            feats = dino.forward_features(x_vit)   # {'x_norm_patchtokens': [B, N, 1024], ...}
            tokens = feats["x_norm_patchtokens"]
            feat16 = tokens.permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
            feature_pyramid[16] = feat16

        return feature_pyramid