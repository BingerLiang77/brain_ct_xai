"""Frozen CT-FM SegResEncoder + trainable SegResNet-style decoder -> 1-channel lesion logit map; case score from the map.

Input  x: (B,1,96,128,128) float, brain-masked brain window in [0,1] (as in the segcls_pack cases).
Output   : logit map (B,1,96,128,128).  score(): differentiable scalar per case from the logit map inside the brain mask.
"""
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = (32, 64, 128, 256, 512)          # encoder stage channels (stage 0 = full resolution, stage 4 = 1/16)


def build_encoder(weights_dir=None):
    """Frozen CT-FM encoder. lighter_zoo first (what the pack was verified with); MONAI fallback (random weights if no dir)."""
    enc = None
    if weights_dir is not None:
        try:
            from lighter_zoo import SegResEncoder
            enc = SegResEncoder.from_pretrained(str(weights_dir))
        except Exception as e:
            print("lighter_zoo load failed, falling back to MONAI + safetensors:", repr(e)[:200])
    if enc is None:
        from monai.networks.nets.segresnet_ds import SegResEncoder as ME
        enc = ME(spatial_dims=3, in_channels=1, init_filters=32, blocks_down=[1, 2, 2, 4, 4], norm=("GROUP", {"num_groups": 8}))
        if weights_dir is not None:
            from safetensors.torch import load_file
            enc.load_state_dict(load_file(str(Path(weights_dir) / "model.safetensors")), strict=True)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.eval()


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv3d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class Decoder(nn.Module):
    """SegResNet up path: 1x1 conv to the skip's channels, upsample, add skip, ResBlock; x4 stages, then 1x1 head."""
    def __init__(self, ch=CH):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv3d(ch[i], ch[i - 1], 1) for i in range(len(ch) - 1, 0, -1)])
        self.blocks = nn.ModuleList([ResBlock(ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out = nn.Conv3d(ch[0], 1, 1)

    def forward(self, fs):
        x = fs[-1]
        for k, i in enumerate(range(len(fs) - 1, 0, -1)):
            skip = fs[i - 1]
            x = self.red[k](x)
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False) + skip
            x = self.blocks[k](x)
        return self.out(F.relu(self.out_norm(x)))


class SegCls(nn.Module):
    def __init__(self, weights_dir=None):
        super().__init__()
        self.encoder = build_encoder(weights_dir)
        self.decoder = Decoder()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()               # frozen encoder always in eval mode
        return self

    def features(self, x):
        # no torch.no_grad(): encoder params are frozen, so training builds no encoder graph anyway,
        # but gradients w.r.t. the INPUT (IG, Grad-CAM) must still flow through the encoder.
        return list(self.encoder(x))

    def forward(self, x, feats=None):
        fs = self.features(x) if feats is None else feats
        return self.decoder(fs).float()


# ---------------------------------------------------------------- case scores (differentiable, computed on LOGITS inside the brain mask)
NEG = -1e4

def case_scores(z, brain):
    """z: (B,1,D,H,W) logits, brain: (B,1,D,H,W) bool/0-1. Returns dict name -> (B,) tensor. Ranking only (AUC) is needed, no calibration."""
    B = z.shape[0]
    zf = z.reshape(B, -1)
    bm = brain.reshape(B, -1).bool()
    zm = torch.where(bm, zf, torch.full_like(zf, NEG))
    out = {"max": zm.max(1).values}
    for tau in (0.5, 1.0, 2.0):
        out[f"lse{tau:g}"] = torch.logsumexp(tau * zm, 1) / tau
    for k in (20, 50, 200):
        out[f"top{k}"] = zm.topk(k, dim=1).values.mean(1)
    out["softvol"] = (torch.sigmoid(zf) * bm).sum(1)
    return out


def seg_loss(z, y, brain, pos_weight=20.0, eps=1.0):
    """BCE (pos_weight) + batch-pooled soft Dice, both restricted to the brain mask."""
    m = brain.float()
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=torch.tensor(pos_weight, device=z.device))
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(z) * m
    inter = (p * y).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + (y * m).sum() + eps)
    return bce + dice, bce.detach(), dice.detach()
