from typing import Optional, Union
import torch
from torch import Tensor
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from romatch_ts.utils.utils import get_autocast_params
# ⬇️ make sure vit_large is imported
from .transformer.dinov2 import vit_large
from typing import Dict



class ResNet50(nn.Module):
    def __init__(self, pretrained: bool = False, high_res: bool = False, weights=None,
                 dilation=None, freeze_bn: bool = True, anti_aliased: bool = False,
                 early_exit: bool = False, amp: bool = False, amp_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        # torchvision >= 0.13 deprecates `pretrained` – your call handles both
        if weights is not None:
            self.net = tvm.resnet50(weights=weights, replace_stride_with_dilation=dilation)
        else:
            self.net = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=dilation)

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        feats = torch.jit.annotate(Dict[int, Tensor], {1: x})
        net = self.net

        if torch.jit.is_scripting():
            # ❌ no autocast in scripted branch
            x = net.conv1(x); x = net.bn1(x); x = net.relu(x); feats[2] = x
            x = net.maxpool(x); x = net.layer1(x); feats[4] = x
            x = net.layer2(x); feats[8] = x
            if self.early_exit: return feats
            x = net.layer3(x); feats[16] = x
            x = net.layer4(x); feats[32] = x
            return feats
        else:
            # ✅ eager-only autocast; enabled can be dynamic here
            with torch.autocast("cuda", enabled=bool(self.amp), dtype=self.amp_dtype):
                x = net.conv1(x); x = net.bn1(x); x = net.relu(x); feats[2] = x
                x = net.maxpool(x); x = net.layer1(x); feats[4] = x
                x = net.layer2(x); feats[8] = x
                if self.early_exit: return feats
                x = net.layer3(x); feats[16] = x
                x = net.layer4(x); feats[32] = x
            return feats


    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



class VGG19(nn.Module):
    def __init__(self, pretrained: bool = False, amp: bool = False, amp_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        feats = torch.jit.annotate(Dict[int, Tensor], {})
        scale = 1

        if torch.jit.is_scripting():
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale *= 2
                x = layer(x)
            return feats
        else:
            with torch.autocast("cuda", enabled=bool(self.amp), dtype=self.amp_dtype):
                for layer in self.layers:
                    if isinstance(layer, nn.MaxPool2d):
                        feats[scale] = x
                        scale *= 2
                    x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs=None, amp=False, use_vgg=False,
                 dinov2_weights=None, amp_dtype=torch.float16):
        super().__init__()

        # ---- CNN backbone
        cnn_kwargs = cnn_kwargs or {}
        self.cnn = VGG19(**cnn_kwargs) if use_vgg else ResNet50(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype

        # ---- DINOv2 ViT-L/14
        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )
        self.dinov2 = vit_large(**vit_kwargs).eval()

        # unwrap common checkpoint wrappers
        if isinstance(dinov2_weights, dict) and "state_dict" in dinov2_weights:
            dinov2_weights = dinov2_weights["state_dict"]
        if isinstance(dinov2_weights, dict) and "model" in dinov2_weights:
            dinov2_weights = dinov2_weights["model"]
        if isinstance(dinov2_weights, dict):
            dinov2_weights = {k.replace("module.", ""): v for k, v in dinov2_weights.items()}

        # load weights (strict=True should work for official vit-l14 ckpt)
        if dinov2_weights is not None:
            _inc = self.dinov2.load_state_dict(dinov2_weights, strict=True)
            print("[DINOv2] loaded vit-large weights (strict=True)")
        else:
            print("[DINOv2] WARNING: no weights provided; backbone will be random.")
        #self.dinov2.load_state_dict(dinov2_weights, strict=True)

    def train(self, mode: bool = True):
        # keep same behavior as before: training flag only affects CNN BN
        return self.cnn.train(mode)


    def forward(self, x: torch.Tensor, upsample: bool = False) -> Dict[int, torch.Tensor]:
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample:
            with torch.no_grad():
                # Keep ViT in fp32 (safe for TS); cast input if needed.
                x_vi = x if x.dtype == torch.float32 else x.float()
                if torch.jit.is_scripting():
                    # TS-friendly path: get patch tokens directly as a Tensor
                    x_norm_patchtokens = self.dinov2.forward_patchtokens(x_vi)
                else:
                    # Eager path: keep the nice Python dict API
                    dinov2_features_16 = self.dinov2.forward_features(x_vi)
                    x_norm_patchtokens = dinov2_features_16["x_norm_patchtokens"]

                f16 = x_norm_patchtokens.permute(0, 2, 1).reshape(B, 1024, H // 14, W // 14)
                feature_pyramid[16] = f16

        return feature_pyramid
