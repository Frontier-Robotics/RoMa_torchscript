import torch
import torch.nn.functional as F
from romatch import roma_outdoor

NET_H, NET_W = 768, 1024
MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1)

def preprocess_u8_rgb_to_net(x_u8_hw3):
    t = torch.from_numpy(x_u8_hw3).to('cuda', non_blocking=True)
    t = t.permute(2,0,1).contiguous().unsqueeze(0).to(torch.float16).mul_(1/255.0)  # [1,3,H,W]
    # scale short side to NET_H
    _,_,h,w = t.shape
    s = NET_H / min(h,w)
    nh, nw = int(round(h*s)), int(round(w*s))
    t = F.interpolate(t, size=(nh,nw), mode='bilinear', align_corners=False)
    # letterbox to NET_H×NET_W
    ph, pw = NET_H - nh, NET_W - nw
    t = F.pad(t, (pw//2, pw - pw//2, ph//2, ph - ph//2))
    # normalize
    return (t - MEAN) / STD  # [1,3,NET_H,NET_W], half

class TSRoMa(torch.nn.Module):
    def __init__(self, roma_model):
        super().__init__()
        self.roma = roma_model  # this is a RegressionMatcher
    def forward(self, L, R):
        # L, R: [1,3,NET_H,NET_W] half, normalized (ImageNet)
        # call the model’s *tensor* path. If forward() isn’t tensor-ready,
        # see Option B below to add it once in the library.
        return self.roma(L, R)  # many nn.Modules use forward(L,R)
        # If that raises, replace with whichever internal call returns (warp, cert)

# Build once
base = roma_outdoor(device='cuda', upsample_res=(NET_H, NET_W)).eval().cuda().half()

# Trace with static shapes
L = torch.zeros(1,3,NET_H,NET_W, device='cuda', dtype=torch.half)
R = torch.zeros(1,3,NET_H,NET_W, device='cuda', dtype=torch.half)
ts = torch.jit.trace(TSRoMa(base), (L, R), strict=False)
ts.save(f"/models/roma_matcher_fp16_{NET_W}x{NET_H}.ts")
