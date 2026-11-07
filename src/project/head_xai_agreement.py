"""Grad-CAM vs IG on the head-CT ICH project: agreement and complementarity.

Put this file in head_ct_ich_ctfm/src and run   python head_xai_agreement.py
It only READS cached attributions (data/xai_masked/attributions/*.pt) and runs forward passes of the model
(no training, no new attributions).  Cases without all four maps (GradCAM_s4, GradCAM_s3, IG, IG_uniform) are skipped.

What it does (all at block level, BLOCK = 8 voxels = 20 mm, only blocks with >= 25 % head voxels):
 1. Agreement of two maps A, B on a case:  Spearman over blocks and top-k overlap (IoU, k = 5/10/20 % of blocks).
    Two controls:
      - population prior: the same agreement between A_i and the leave-one-out MEAN map of B over the other cases.
        If agreement(A_i, B_i) is not above agreement(A_i, mean B_-i), the two methods only share a case-independent
        pattern (brain shape, skull base ...), not case-specific information.
      - self ceiling: same-family pairs (IG meanneg vs IG uniform, GradCAM s3 vs s4).
 2. Complementarity: top-k sets of A and B are split into  A-only, B-only, intersection.  Each set is replaced by a
    baseline volume (deletion) and the logit drop is compared with random block sets of the same size.
    If A-only AND B-only both drop the logit more than random, each carries information the other lacks.
    Done under three replacement volumes (meanneg, blur2, uniform brain 0.5 inside the head) because the replacement
    value that equals one method's IG baseline favours that method.
 3. Where the three sets lie (normalised z and AP position of block centres).

Outputs go to  <XAI_DIR>/agreement/  (csv + summary.txt).
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon

import xai_common as xc
import xai_faith as xf

OUT = Path(xc.XAI_DIR) / "agreement"
MAPS = {"GC4": "GradCAM_s4", "GC3": "GradCAM_s3", "IG_mn": "IG", "IG_u": "IG_uniform"}
PAIRS_AGREE = [("GC4", "IG_u"), ("GC4", "IG_mn"), ("GC3", "IG_u"), ("GC3", "IG_mn")]   # across methods
PAIRS_CEIL = [("IG_mn", "IG_u"), ("GC3", "GC4")]                                       # within a family
PAIRS_COMP = [("GC4", "IG_u"), ("GC4", "IG_mn")]                                       # deletion test (costs forwards)
KS = (0.05, 0.10, 0.20)
REPL = ("meanneg", "blur2", "uniform")
N_RAND = 5
SEED = 0
BLOCK = xf.BLOCK
MAX_CASES = None            # set e.g. 10 for a quick trial


# ----------------------------------------------------------------------------- loading
def head_of(x):
    """x: (1, D, H, W) -> (D, H, W) bool.  xc.head_mask may take (1,1,D,H,W) and return any broadcastable shape."""
    h = xc.head_mask(x[None])
    return torch.as_tensor(h).reshape(x.shape[1:]).bool()


def load_case(cid):
    f = Path(xc.XAI_DIR) / "attributions" / f"{cid}.pt"
    if not f.exists():
        return None
    d = torch.load(f, weights_only=True)
    if any(k not in d for k in MAPS.values()):
        return None
    x = xc.load_tensor(cid)
    head = head_of(x)
    maps = {n: d[k].float().clamp(min=0) for n, k in MAPS.items()}        # IG: positive part; Grad-CAM already >= 0
    bid, nb, elig, cnt = xf.block_stats(head)
    sc = {n: xf.block_scores(m, head, bid, nb, cnt) for n, m in maps.items()}
    return dict(id=cid, x=x, head=head, bid=bid, nb=nb, elig=elig, cnt=cnt, sc=sc)


# ----------------------------------------------------------------------------- agreement
def top_set(score, elig, k, gen):
    """Top-k fraction of eligible blocks; ties (Grad-CAM has many zeros) are broken at random, not by position."""
    order = xf.order_from_scores(score, elig, gen)
    n = max(1, int(round(k * int(elig.sum()))))
    return set(order[:n].tolist())


def iou(a, b):
    return len(a & b) / max(1, len(a | b))


def rho(a, b, elig):
    a, b = a[elig].numpy(), b[elig].numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def normalised(sc, elig):
    s = sc.clone() * elig.float()
    return s / s.sum().clamp(min=1e-12)


def agreement_rows(cases):
    names = list(MAPS)
    S = {n: torch.stack([normalised(c["sc"][n], c["elig"]) for c in cases]) for n in names}   # (N, nb)
    tot = {n: S[n].sum(0) for n in names}
    N = len(cases)
    rows = []
    for i, c in enumerate(cases):
        gen = torch.Generator().manual_seed(SEED + i)
        loo = {n: (tot[n] - S[n][i]) / max(1, N - 1) for n in names}
        for kind, pairs in (("across", PAIRS_AGREE), ("ceiling", PAIRS_CEIL)):
            for a, b in pairs:
                r = dict(id=c["id"], kind=kind, A=a, B=b, n_elig=int(c["elig"].sum()),
                         rho=rho(c["sc"][a], c["sc"][b], c["elig"]),
                         rho_pop=np.nanmean([rho(c["sc"][a], loo[b], c["elig"]), rho(c["sc"][b], loo[a], c["elig"])]))
                for k in KS:
                    ta, tb = top_set(c["sc"][a], c["elig"], k, gen), top_set(c["sc"][b], c["elig"], k, gen)
                    pa, pb = top_set(loo[a], c["elig"], k, gen), top_set(loo[b], c["elig"], k, gen)
                    r[f"iou{int(k*100)}"] = iou(ta, tb)
                    r[f"iou{int(k*100)}_pop"] = 0.5 * (iou(ta, pb) + iou(tb, pa))
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- complementarity
def repl_volumes(c, meanneg):
    bv = xf.baseline_volumes(c["x"], meanneg)
    bv["uniform"] = torch.where(c["head"][None], torch.tensor(0.5), torch.tensor(0.0))
    return {k: bv[k] for k in REPL}


def block_centres(bid, nb, shape, b=BLOCK):
    nh, nw = (shape[1] + b - 1) // b, (shape[2] + b - 1) // b
    ids = torch.arange(nb)
    iz, rem = ids // (nh * nw), ids % (nh * nw)
    return (iz * b + (b - 1) / 2).float(), ((rem // nw) * b + (b - 1) / 2).float()


@torch.no_grad()
def logit_drops(model, c, bvol, sets, l_orig, device):
    vols = [torch.where(torch.isin(c["bid"], torch.tensor(sorted(s), dtype=torch.long))[None], bvol, c["x"]) for s in sets]
    return (l_orig - xf._logits(model, vols, device)).tolist()


def complement_rows(model, cases, meanneg, device):
    rows, loc_rows = [], []
    for i, c in enumerate(cases):
        gen = torch.Generator().manual_seed(SEED + 1000 + i)
        elig_idx = torch.where(c["elig"])[0]
        l_orig = float(xf._logits(model, [c["x"]], device)[0])
        bvols = repl_volumes(c, meanneg)
        zc, yc = block_centres(c["bid"], c["nb"], c["head"].shape)
        hz = torch.where(c["head"].any(dim=(1, 2)))[0]
        hy = torch.where(c["head"].any(dim=(0, 2)))[0]
        zn = (zc - hz.min()) / max(1.0, float(hz.max() - hz.min() + 1))
        yn = (yc - hy.min()) / max(1.0, float(hy.max() - hy.min() + 1))
        for a, b in PAIRS_COMP:
            for k in KS:
                ta, tb = top_set(c["sc"][a], c["elig"], k, gen), top_set(c["sc"][b], c["elig"], k, gen)
                parts = {"A_only": ta - tb, "B_only": tb - ta, "inter": ta & tb}
                if len(parts["A_only"]) == 0 or len(parts["B_only"]) == 0:
                    continue
                sets, tags = [], []
                for nm, s in parts.items():
                    if s:
                        sets.append(s); tags.append(nm)
                for size, nm in ((len(parts["A_only"]), "rand_ab"), (len(parts["inter"]), "rand_inter")):
                    if size == 0:
                        continue
                    for _ in range(N_RAND):
                        sets.append(set(elig_idx[torch.randperm(len(elig_idx), generator=gen)[:size]].tolist()))
                        tags.append(nm)
                for rname, bvol in bvols.items():
                    d = logit_drops(model, c, bvol, sets, l_orig, device)
                    agg = {}
                    for t, v in zip(tags, d):
                        agg.setdefault(t, []).append(v)
                    rows.append(dict(id=c["id"], A=a, B=b, k=k, repl=rname, logit=l_orig,
                                     n_ab=len(parts["A_only"]), n_inter=len(parts["inter"]),
                                     **{t: float(np.mean(v)) for t, v in agg.items()}))
                for nm, s in parts.items():
                    if s:
                        idx = torch.tensor(sorted(s))
                        loc_rows.append(dict(id=c["id"], A=a, B=b, k=k, part=nm, z=float(zn[idx].mean()), ap=float(yn[idx].mean())))
        for nm, idx in (("all_eligible", elig_idx),):
            loc_rows.append(dict(id=c["id"], A="-", B="-", k=0, part=nm, z=float(zn[idx].mean()), ap=float(yn[idx].mean())))
        print(f"  complementarity {i + 1}/{len(cases)} {c['id']}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(loc_rows)


# ----------------------------------------------------------------------------- summaries
def wil(x, y=None):
    x = np.asarray(x, float)
    if y is not None:
        x = x - np.asarray(y, float)
    x = x[~np.isnan(x)]
    if len(x) < 6 or np.all(x == 0):
        return np.nan
    return float(wilcoxon(x)[1])


def summarise(agr, comp, loc, labels, f):
    def P(*a):
        print(*a); print(*a, file=f)
    agr = agr.merge(labels, on="id")
    comp = comp.merge(labels, on="id") if len(comp) else comp
    P("=" * 100); P("1. AGREEMENT (block level; 'pop' = same agreement against the leave-one-out mean map of the other cases)")
    for grp, g in [("ALL", agr)] + [(f"label={l}", d) for l, d in agr.groupby("label")]:
        P(f"\n-- {grp}: n cases = {g['id'].nunique()}")
        t = g.groupby(["kind", "A", "B"]).agg(rho=("rho", "mean"), rho_pop=("rho_pop", "mean"),
                                               iou10=("iou10", "mean"), iou10_pop=("iou10_pop", "mean"),
                                               iou20=("iou20", "mean"), iou20_pop=("iou20_pop", "mean"))
        t["rho_gain"] = t["rho"] - t["rho_pop"]
        t["iou10_gain"] = t["iou10"] - t["iou10_pop"]
        P(t.round(3).to_string())
        P("   random expectation of IoU for a top-k fraction p:  p/(2-p):  "
          + ", ".join(f"k={int(k*100)}%: {k/(2-k):.3f}" for k in KS))
        a = g[g.kind == "across"]
        for (A, B), d in a.groupby(["A", "B"]):
            P(f"   {A} vs {B}: paired Wilcoxon  rho vs rho_pop  p = {wil(d['rho'], d['rho_pop']):.3g};  "
              f"iou10 vs iou10_pop  p = {wil(d['iou10'], d['iou10_pop']):.3g}")
    P("\n   Reading: gain ~ 0  -> the methods only share a case-independent pattern.   "
      "gain > 0 and close to the ceiling rows -> they agree on case-specific content.")
    if len(comp) == 0:
        return
    P("\n" + "=" * 100)
    P("2. COMPLEMENTARITY: logit drop when the set is replaced by the baseline (mean over cases; margin = drop - random of same size)")
    comp["m_A"], comp["m_B"], comp["m_I"] = comp["A_only"] - comp["rand_ab"], comp["B_only"] - comp["rand_ab"], comp["inter"] - comp["rand_inter"]
    for grp, g in [("ALL", comp)] + [(f"label={l}", d) for l, d in comp.groupby("label")]:
        P(f"\n-- {grp}")
        rows = []
        for (A, B, k, r), d in g.groupby(["A", "B", "k", "repl"]):
            rows.append(dict(A=A, B=B, k=k, repl=r, n=d["id"].nunique(),
                             A_only=d["A_only"].mean(), B_only=d["B_only"].mean(), inter=d["inter"].mean(), rand=d["rand_ab"].mean(),
                             mA=d["m_A"].mean(), pA=wil(d["m_A"]), mB=d["m_B"].mean(), pB=wil(d["m_B"]),
                             mI=d["m_I"].mean(), p_A_vs_B=wil(d["A_only"], d["B_only"])))
        P(pd.DataFrame(rows).round(3).to_string(index=False))
    P("\n   Reading: mA>0 and mB>0 (small p) -> each method's private blocks carry information the other lacks (complementary).")
    P("            only one of them > 0 -> the other adds nothing beyond it.   both ~ 0 -> the private blocks are noise.")
    P("            Check that the sign is the same under meanneg / blur2 / uniform (baseline-matching bias).")
    if len(loc):
        P("\n" + "=" * 100)
        P("3. WHERE: normalised block-centre position (0 = lowest array index of the head, 1 = highest). "
          "z: with MONAI 'SPL' a LOW index is INFERIOR (skull base); verify once on an image. ap: low index = anterior.")
        t = loc.groupby(["A", "B", "k", "part"]).agg(z=("z", "mean"), ap=("ap", "mean")).round(3)
        P(t.to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = xc.pick_device()
    model = xc.load_model(device)
    meanneg = xc.mean_negative_baseline()
    cases_df = xc.load_cases()
    labels = cases_df[["id", "label"]]
    ids = [i for i in cases_df["id"] if (Path(xc.XAI_DIR) / "attributions" / f"{i}.pt").exists()]
    if MAX_CASES:
        ids = ids[:MAX_CASES]
    cases = [c for c in (load_case(i) for i in ids) if c is not None]
    print(f"{len(cases)} cases with all four maps; label counts: "
          f"{labels[labels['id'].isin([c['id'] for c in cases])]['label'].value_counts().to_dict()}")
    agr = agreement_rows(cases)
    agr.to_csv(OUT / "agreement_cases.csv", index=False)
    comp, loc = complement_rows(model, cases, meanneg, device)
    comp.to_csv(OUT / "complement_cases.csv", index=False)
    loc.to_csv(OUT / "complement_location.csv", index=False)
    with open(OUT / "summary.txt", "w") as f:
        summarise(agr, comp, loc, labels, f)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
