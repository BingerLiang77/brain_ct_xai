"""Deletion / insertion faithfulness test on 3-D blocks.

The volume is cut into cubic blocks (default 8^3 voxels = 20 mm). Blocks that contain >= 25 % head voxels are
"eligible"; a method scores each eligible block by the mean of its attribution over head voxels and ranks
blocks from most to least supportive of the haemorrhage logit.

* deletion : replace the top-k blocks by a baseline; a faithful map makes the logit drop fast.
* insertion: start from the baseline volume and reveal the top-k blocks; a faithful map makes the logit rise fast.

Baselines: `zero` (air, -1024 HU) and `shuffle` (head voxels randomly permuted: same intensity histogram, no structure).
Random block orders are run alongside as the reference; the headline numbers are gains over random ordering.
Both methods are evaluated on the same blocks, so the coarse Grad-CAM and the voxel-level IG are compared fairly.
"""
import numpy as np
import torch

BLOCK = 8
FRACS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75)      # interior points; 0 and 1 are the original / baseline volumes
GRID = np.array((0.0,) + FRACS + (1.0,))


def block_ids(shape, b=BLOCK):
    d, h, w = [torch.arange(s) for s in shape]
    nd, nh, nw = (shape[0] + b - 1) // b, (shape[1] + b - 1) // b, (shape[2] + b - 1) // b
    return ((d // b)[:, None, None] * nh + (h // b)[None, :, None]) * nw + (w // b)[None, None, :], nd * nh * nw


def block_stats(head):
    """head: (D, H, W) bool. Returns bid (D, H, W) long, n_blocks, eligible (n_blocks,) bool, head voxel count per block."""
    bid, nb = block_ids(head.shape)
    cnt = torch.bincount(bid.flatten(), weights=head.flatten().float(), minlength=nb)
    full = torch.bincount(bid.flatten(), minlength=nb).float()
    return bid, nb, cnt >= 0.25 * full, cnt


def block_scores(amap, head, bid, nb, cnt):
    s = torch.bincount(bid.flatten(), weights=(amap * head).flatten().float(), minlength=nb)
    return s / cnt.clamp(min=1)


def order_from_scores(scores, eligible, gen):
    """Eligible blocks sorted by descending score, ties broken at random. Returns block ids."""
    idx = torch.where(eligible)[0]
    tie = torch.rand(len(idx), generator=gen)
    o = np.lexsort((tie.numpy(), -scores[idx].numpy()))
    return idx[torch.from_numpy(o)]


def baseline_volumes(x, meanneg, sigma=2.0):
    """x: (1, D, H, W) cpu. Replacement volumes chosen for neutrality (their logit sits near the negative class):
    `meanneg` = mean training-negative head; `blur2` = this case blurred with sigma = 2 voxels (5 mm).
    An all-air baseline (logit +3.8) and a head-voxel shuffle (logit up to +3 on positives) were rejected."""
    from scipy.ndimage import gaussian_filter
    return {"meanneg": meanneg.clone(), "blur2": torch.from_numpy(gaussian_filter(x[0].numpy(), sigma)).float()[None]}


@torch.no_grad()
def _logits(model, vols, device, chunk=7):
    out = []
    for i in range(0, len(vols), chunk):
        out.append(model(torch.stack(vols[i:i + chunk]).to(device)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def curves_for_order(model, x, bvol, order, bid, nb, n_elig, l_orig, l_base, device):
    """x, bvol: (1, D, H, W) cpu. order: block ids in importance order. Returns {'del': [...], 'ins': [...]} over GRID."""
    rank = torch.full((nb,), 10 ** 9, dtype=torch.long)
    rank[order] = torch.arange(len(order))
    rank_vol = rank[bid][None]
    dels, inss = [], []
    for f in FRACS:
        k = max(1, int(round(f * n_elig)))
        sel = rank_vol < k
        dels.append(torch.where(sel, bvol, x))
        inss.append(torch.where(sel, x, bvol))
    ld, li = _logits(model, dels, device), _logits(model, inss, device)
    return {"del": [l_orig] + ld.tolist() + [l_base], "ins": [l_base] + li.tolist() + [l_orig]}


def auc(y):
    return float(np.trapezoid(y, GRID))


def evaluate_case(model, x, maps, head, meanneg, lesion=None, seed=0, baselines=("meanneg", "blur2"), sigma=2.0, n_random=2, device="cpu"):
    """x: (1, D, H, W) cpu; maps: {name: (D, H, W) cpu}; head, lesion: (D, H, W) bool cpu; meanneg: (1, D, H, W) cpu.
    Returns {baseline: {ordering: {'del': [...], 'ins': [...]}}, 'n_elig': int, 'logit': float}."""
    gen = torch.Generator().manual_seed(seed)
    bid, nb, elig, cnt = block_stats(head)
    n_elig = int(elig.sum())
    orders = {name: order_from_scores(block_scores(m, head, bid, nb, cnt), elig, gen) for name, m in maps.items()}
    for r in range(n_random):
        orders[f"random{r}"] = order_from_scores(torch.rand(nb, generator=gen), elig, gen)
    if lesion is not None and lesion.any():
        les_frac = torch.bincount(bid.flatten(), weights=lesion.flatten().float(), minlength=nb) / cnt.clamp(min=1)
        orders["lesion_first"] = order_from_scores(les_frac, elig, gen)
    l_orig = float(_logits(model, [x], device)[0])
    res = {"n_elig": n_elig, "logit": l_orig}
    for b, bvol in baseline_volumes(x, meanneg, sigma=sigma).items():
        if b not in baselines:
            continue
        l_base = float(_logits(model, [bvol], device)[0])
        res[b] = {"logit_baseline": l_base}
        for name, order in orders.items():
            res[b][name] = curves_for_order(model, x, bvol, order, bid, nb, n_elig, l_orig, l_base, device)
    return res
