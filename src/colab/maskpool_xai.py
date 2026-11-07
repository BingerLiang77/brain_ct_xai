"""Step 1 (Colab): attributions for the MASKED-POOL model (frozen CT-FM encoder -> head-masked average pooling of 5 stages -> linear head)
with FOUR IG baselines and Grad-CAM, brain/head-only.   Self-contained: needs only segcls_pack (data/cases, weights, ref).

  python maskpool_xai.py validate --pack PACK --out OUT              # 2 cases: does this port reproduce the stored logits / maps?
  python maskpool_xai.py converge --pack PACK --out OUT              # IG step-count check on 3 cases (24 / 64 / 128 steps), all 4 baselines
  python maskpool_xai.py attr     --pack PACK --out OUT [--steps 64] # the 55 cases of ref/attributions (35 test pos + 20 test neg)

IG baselines (all keep the outside of the head at 0, as the model only ever sees 0 there):
  IG_zero   0 inside the head too  (= -20 HU, the same value as outside the brain)
  IG_u      constant 0.5 inside the head  (= 40 HU)                           [same as the earlier IG_u]
  IG_bl     this case blurred inside the head (sigma 2 voxels, normalised convolution)   [same as the earlier IG_bl]
  IG_neg    head-conditional voxelwise mean of the 763 TRAINING negatives: for each voxel, the mean of x over the negatives whose head
            contains that voxel (the earlier unconditional mean was diluted by zeros at the head edge); voxels seen by < 20 negatives
            get the global mean head intensity.  Written only inside this case's head.
IMPORTANT: the model computes its pooling mask as (x > 0.01).  With a zero baseline that mask would vanish along the IG path, so here the
pooling mask is FIXED to the case's original head for every call (identical to the old behaviour for u / bl / neg, which stay > 0.01).
Grad-CAM maps are multiplied by the head mask (outside the head they do not affect the logit); the share of raw mass outside is stored.
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

SIGMA = 2.0
BASES = ("IG_zero", "IG_u", "IG_bl", "IG_neg")


# ------------------------------------------------------------------ model
def build_encoder(weights_dir):
    if os.environ.get("MPOOL_NOPRE"):
        torch.manual_seed(0)
        from monai.networks.nets.segresnet_ds import SegResEncoder as ME
        enc = ME(spatial_dims=3, in_channels=1, init_filters=32, blocks_down=[1, 2, 2, 4, 4], norm=("GROUP", {"num_groups": 8}))
    else:
        from lighter_zoo import SegResEncoder
        enc = SegResEncoder.from_pretrained(str(weights_dir))
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.eval()


class ICHMasked(nn.Module):
    """Frozen encoder -> masked average pooling (mask = `head` if given, else x > 0.01) -> standardise -> linear head. Same as the project's class."""
    def __init__(self, backbone, ck):
        super().__init__()
        self.backbone, self.stages = backbone, list(ck["stages"])
        for n in ("mean", "std", "w"):
            self.register_buffer(n, ck[n].float())
        self.register_buffer("b", torch.tensor(float(ck["b"])))
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x, return_feats=False, head=None):
        fs = self.backbone(x)
        m = (x > 0.01).to(x.dtype) if head is None else head.to(x.dtype).expand(x.shape[0], *head.shape[1:])
        zs = []
        for s in self.stages:
            f = fs[s]
            w = F.adaptive_avg_pool3d(m, f.shape[2:])
            zs.append((f * w).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3, 4)).clamp(min=1e-6))
        logit = ((torch.cat(zs, 1) - self.mean) / self.std) @ self.w + self.b
        return (logit, fs) if return_feats else logit


def load_model(pack, dev):
    ck = torch.load(Path(pack) / "weights" / "ich_head_masked_pool_stages_0_1_2_3_4.pt", map_location="cpu", weights_only=True)
    return ICHMasked(build_encoder(Path(pack) / "weights" / "ct_fm_feature_extractor"), ck).to(dev)


# ------------------------------------------------------------------ data and baselines
def load_case(pack, cid):
    d = torch.load(Path(pack) / "data" / "cases" / f"{cid}.pt", weights_only=False)
    return dict(id=cid, x=d["x"].float(), head=d["head"].bool(), brain=d["brain"].bool(), lesion=d["lesion"].bool(), label=int(d["meta"]["label"]))


def xai_ids(pack):
    return sorted(p.stem for p in (Path(pack) / "ref" / "attributions").glob("*.pt"))


def blur_in_head(x, head, s=SIGMA):
    xa, h = x[0].numpy(), head.numpy().astype(np.float32)
    num, den = ndi.gaussian_filter(xa * h, s), ndi.gaussian_filter(h, s)
    return torch.from_numpy(np.where(h > 0, num / np.maximum(den, 1e-6), 0.0)).float()[None]


def negmean_volume(pack, out, min_n=20):
    f = Path(out) / "negmean_volume.pt"
    if f.exists():
        return torch.load(f, weights_only=True)
    coh = pd.read_csv(Path(pack) / "cohort.csv")
    ids = list(coh[(coh.split == "train") & (coh.label == 0)].id)
    s = n = None
    for i in ids:
        c = load_case(pack, i)
        h = c["head"].float()[None]
        s = c["x"] * h if s is None else s + c["x"] * h
        n = h if n is None else n + h
    glob = float(s.sum() / n.sum())
    vol = torch.where(n >= min_n, s / n.clamp(min=1), torch.tensor(glob))
    print(f"negative-mean volume from {len(ids)} training negatives; global head mean {glob:.4f} ({glob * 120 - 20:.1f} HU); "
          f"voxels with < {min_n} contributing cases (inside any head) -> global mean: {float(((n < min_n) & (n > 0)).sum()):.0f}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(vol, f)
    return vol


def make_baselines(c, negvol):
    h = c["head"][None]
    return {"IG_zero": torch.zeros_like(c["x"]), "IG_u": torch.where(h, torch.tensor(0.5), torch.tensor(0.0)),
            "IG_bl": blur_in_head(c["x"], c["head"]), "IG_neg": torch.where(h, negvol, torch.tensor(0.0))}


# ------------------------------------------------------------------ Grad-CAM and IG
def gradcam(model, x, head):
    x = x.clone().requires_grad_(True)
    logit, fs = model(x, return_feats=True, head=head)
    grads = torch.autograd.grad(logit.sum(), fs)
    out, outside = {}, {}
    for k, (f, g) in enumerate(zip(fs, grads)):
        a = g.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((a * f).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        hm = head[0, 0].to(cam.device)
        outside[k] = float(cam[~hm].sum() / cam.sum().clamp(min=1e-12))
        out[f"GradCAM_s{k}"] = (cam * hm).cpu()
    return float(logit), out, outside


def integrated_gradients(model, x, base, head, steps=64, chunk=4, power=1):
    """Gauss-Legendre IG on the straight path.  power > 1 warps the path parameter (alpha = t**power, weight p*t**(p-1)) so that the
    nodes crowd towards the baseline; needed for the ZERO baseline, where GroupNorm makes the network nearly scale-invariant for alpha > 0
    and the whole change of the logit happens in a tiny neighbourhood of alpha = 0."""
    nodes, w = np.polynomial.legendre.leggauss(steps)
    t, w = (nodes + 1) / 2, w / 2
    al, w = t ** power, w * power * t ** (power - 1)
    diff, tot = x - base, torch.zeros_like(x)
    for i in range(0, steps, chunk):
        a = torch.tensor(al[i:i + chunk], dtype=x.dtype, device=x.device).view(-1, 1, 1, 1, 1)
        ww = torch.tensor(w[i:i + chunk], dtype=x.dtype, device=x.device).view(-1, 1, 1, 1, 1)
        xs = (base + a * diff).requires_grad_(True)
        g, = torch.autograd.grad(model(xs, head=head).sum(), xs)
        tot += (ww * g).sum(0, keepdim=True)
    attr = (diff * tot).detach()
    with torch.no_grad():
        l_x, l_0 = float(model(x, head=head)), float(model(base, head=head))
    return attr[0, 0].cpu(), dict(logit=l_x, base_logit=l_0, err=float(attr.sum()) - (l_x - l_0))


# ------------------------------------------------------------------ commands
def cmd_attr(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / "attributions"; out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.pack, dev)
    negvol = negmean_volume(a.pack, a.out)
    ids = xai_ids(a.pack)[:a.max_cases] if a.max_cases else xai_ids(a.pack)
    use = tuple(b.strip() for b in a.bases.split(",")) if a.bases else BASES
    print(f"{len(ids)} cases, steps {a.steps}, baselines {use}", flush=True)
    t0, rows = time.time(), []
    for n, cid in enumerate(ids):
        f = out / f"{cid}.pt"
        d = torch.load(f, weights_only=True) if f.exists() else {"id": cid}
        c = load_case(a.pack, cid)
        x, head = c["x"][None].to(dev), c["head"][None, None].to(dev)
        if "GradCAM_s4" not in d:
            logit, maps, outside = gradcam(model, x, head)
            d["logit"] = logit
            for k, v in maps.items():
                d[k] = v.half()
            for k, v in outside.items():
                d[f"gc_s{k}_outside_mass"] = v
        bases = make_baselines(c, negvol)
        for key in use:
            if key in d and d.get(f"{key}_steps") == a.steps:
                continue
            attr, info = integrated_gradients(model, x, bases[key][None].to(dev), head, a.steps, a.chunk, a.zero_power if key == "IG_zero" else 1)
            d[key] = attr.half(); d[f"{key}_err"], d[f"{key}_base_logit"], d[f"{key}_steps"] = info["err"], info["base_logit"], a.steps
            d[f"{key}_sum16"] = float(attr.half().float().sum())
            rows.append(dict(id=cid, label=c["label"], base=key, logit=info["logit"], base_logit=info["base_logit"], err=info["err"], sum16=d[f"{key}_sum16"]))
        torch.save(d, f)
        print(f"  {n + 1}/{len(ids)} {cid} ({time.time() - t0:.0f}s)", flush=True)
    if rows:
        df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "ig_completeness.csv", index=False)
        df["gap"] = (df.logit - df.base_logit).abs()
        df["rel"] = df.err.abs() / df.gap.clip(lower=1e-6)
        print("\nIG completeness: |sum(attr) - (logit - base_logit)| per baseline (absolute, and relative to |logit - base_logit|)")
        print(df.groupby("base").agg(err_med=("err", lambda s: s.abs().median()), err_p95=("err", lambda s: s.abs().quantile(.95)), err_max=("err", lambda s: s.abs().max()),
                                       gap_med=("gap", "median"), rel_med=("rel", "median"), base_logit_med=("base_logit", "median")).round(3).to_string())
        print("baseline logit of negatives (median):", df[df.label == 0].groupby("base").base_logit.median().round(2).to_dict())


def cmd_validate(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.pack, dev); negvol = negmean_volume(a.pack, a.out)
    for cid in xai_ids(a.pack)[:a.max_cases or 2]:
        ref = torch.load(Path(a.pack) / "ref" / "attributions" / f"{cid}.pt", weights_only=True)
        c = load_case(a.pack, cid); x, head = c["x"][None].to(dev), c["head"][None, None].to(dev)
        logit, maps, outside = gradcam(model, x, head)
        print(f"\n{cid}: logit recomputed {logit:.4f} vs stored {float(ref['logit']):.4f}  |  Grad-CAM mass outside head (raw): " + ", ".join(f"s{k} {v:.2f}" for k, v in outside.items()))
        for k in ("GradCAM_s3", "GradCAM_s4"):
            r = ref[k].float() * c["head"]; m = maps[k]
            print(f"  {k}: Pearson with stored (head only) {np.corrcoef(m[c['head']].numpy(), r[c['head']].numpy())[0, 1]:.4f}")
        bases = make_baselines(c, negvol)
        for key in ("IG_bl", "IG_u"):
            attr, info = integrated_gradients(model, x, bases[key][None].to(dev), head, 24, a.chunk)
            r = ref[key].float()
            print(f"  {key} (24 steps): Pearson with stored {np.corrcoef(attr.flatten().numpy(), r.flatten().numpy())[0, 1]:.4f}; err {info['err']:+.3f} vs stored {float(ref[key + '_err']):+.3f}; "
                  f"base logit {info['base_logit']:.3f} vs stored {float(ref[key + '_base_logit']):.3f}")


def cmd_converge(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.pack, dev); negvol = negmean_volume(a.pack, a.out)
    from scipy.stats import spearmanr
    for cid in xai_ids(a.pack)[:a.max_cases or 3]:
        c = load_case(a.pack, cid); x, head = c["x"][None].to(dev), c["head"][None, None].to(dev)
        bases = make_baselines(c, negvol)
        with torch.no_grad():
            prof = {al: float(model(x * al, head=head)) for al in (0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)}
        print(f"\n{cid[:12]} logit along the ZERO path, logit(alpha * x): " + ", ".join(f"{k:g}: {v:.2f}" for k, v in prof.items()))
        for key in BASES:
            for pw in ((1, 4) if key == "IG_zero" else (1,)):
                res = {s: integrated_gradients(model, x, bases[key][None].to(dev), head, s, a.chunk, pw) for s in (24, 64, 128)}
                ref = res[128][0][c["head"]].numpy()
                line = "  ".join(f"{s}: err {res[s][1]['err']:+.3f}, rho vs 128 {spearmanr(res[s][0][c['head']].numpy(), ref)[0]:.3f}" for s in (24, 64))
                tag = key + (f"(p{pw})" if key == "IG_zero" else "")
                print(f"  {tag:11s} |logit-base| {abs(res[128][1]['logit'] - res[128][1]['base_logit']):.2f}  128: err {res[128][1]['err']:+.3f}   {line}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attr", "validate", "converge"])
    ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=64); ap.add_argument("--zero-power", type=int, default=1, help="path warp for the zero baseline (1 = plain straight path; `converge` tests 1 and 4)"); ap.add_argument("--chunk", type=int, default=4); ap.add_argument("--max-cases", type=int, default=None); ap.add_argument("--bases", default=None, help="comma list, e.g. IG_zero (default: all four)")
    a = ap.parse_args()
    {"attr": cmd_attr, "validate": cmd_validate, "converge": cmd_converge}[a.cmd](a)


if __name__ == "__main__":
    main()
