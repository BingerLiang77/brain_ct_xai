"""Why do the 'mean' and the 'max' block rankings behave differently?  CPU only, reads the stored IG attributions.

Blocks (20 mm, eligible = >= 25 % in head) are scored two ways:
  mean   signed net attribution per head voxel      max   largest POSITIVE voxel attribution inside the block
The top-k % sets are formed exactly as in the deletion / insertion runs (highest-ranked blocks holding k % of the head voxels of the eligible blocks).
Per method and k (median over cases [IQR]) it reports

  1. overlap   |top_mean n top_max| / |top_mean| (chance = the share of eligible blocks in the other set), Jaccard, and the Spearman rho of the two block scores
  2. NEGATIVE CONTENT INSIDE THE TOP BLOCKS, mean-ranked | max-ranked | all eligible blocks (reference):
       neg voxels      share of the head voxels in the set with attribution < 0
       material neg    share with attribution < -5 % of the case's largest |attribution|   (ignores numerical noise around 0)
       material pos    share with attribution > +5 % of it
       neg mass        negative attribution / (negative + positive attribution)  (0 = no negative evidence, 0.5 = equal amounts)
       net / gross     (positive - negative) / (positive + negative): how much of the attribution survives cancellation
       peak ratio      median over blocks of |most negative voxel| / (largest positive voxel)   (> 1: the block's strongest voxel is a NEGATIVE one although it was ranked by its positive peak)
  3. what else differs: head fraction of the blocks (edge blocks), mean intensity (HU), lesion-block share (base rate = share among eligible blocks)

Usage:  python mpool_rank_compare.py --pack PACK --attr DIR [DIR2 ...] --out OUT [--methods ...] [--include-neg]     (needs mpool_faith.py, mpool_loc.py, maskpool_xai.py alongside)
Writes OUT/rank_compare_cases.csv, rank_compare.txt, rank_compare.png.
"""
import argparse
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import mpool_faith as mf
import mpool_loc as ml
import maskpool_xai as mx

warnings.filterwarnings("ignore")
KS = (5, 10, 20, 30, 50)
MAT = 0.05          # "material" = beyond 5 % of the case's largest |attribution|


def set_stats(S, p, hc, a):
    """negative content of a block set S (block ids); p = dict of per-block sums"""
    pos, neg, hcnt = p["pos"][S].sum(), p["neg"][S].sum(), hc[S].sum()
    r = dict(neg_vox=float(p["negv"][S].sum() / hcnt), mat_neg=float(p["matneg"][S].sum() / hcnt), mat_pos=float(p["matpos"][S].sum() / hcnt),
             neg_mass=float(neg / (pos + neg).clamp(min=1e-12)), net_gross=float((pos - neg) / (pos + neg).clamp(min=1e-12)))
    pr = p["minabs"][S] / p["maxpos"][S].clamp(min=1e-12)
    r["peak_ratio"] = float(pr.median())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--attr", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["IG_zero", "IG_u", "IG_bl", "IG_neg"]); ap.add_argument("--include-neg", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids = [i for i in ml.case_ids(a.attr) if a.include_neg or mx.load_case(a.pack, i)["label"] == 1]
    print(f"{len(ids)} cases; methods {a.methods}", flush=True)
    rows = []
    for n, cid in enumerate(ids):
        c = mx.load_case(a.pack, cid); d = ml.load_attr(a.attr, cid)
        head = c["head"]
        bid, nb, elig, cnt = ml.block_stats(head)
        flat = bid.flatten()
        size = float(torch.bincount(flat, minlength=nb).max())
        hfrac = cnt / size
        xmean = ml.block_mean(c["x"][0], head, bid, nb, cnt) * 120 - 20
        lc = torch.zeros(nb).scatter_add_(0, flat, (c["lesion"] & head).flatten().float())
        lesblk = (lc >= 5) & elig
        base = float(lesblk[elig].float().mean())
        allb = torch.nonzero(elig).flatten()
        for m in a.methods:
            if m not in d:
                continue
            att = d[m].float() * head
            hf = head.flatten()
            av = att.flatten()
            th = MAT * float(av.abs().max())
            sc = lambda v: torch.zeros(nb).scatter_add_(0, flat, (v * hf).float())
            p = dict(pos=sc(av.clamp(min=0)), neg=sc((-av).clamp(min=0)), negv=sc((av < 0).float()), matneg=sc((av < -th).float()), matpos=sc((av > th).float()),
                     maxpos=torch.zeros(nb).scatter_reduce(0, flat, (av.clamp(min=0) * hf), "amax", include_self=True),
                     minabs=torch.zeros(nb).scatter_reduce(0, flat, ((-av).clamp(min=0) * hf), "amax", include_self=True))
            s_mean = mf.block_scores(att, head, bid, nb, cnt, "mean"); s_max = mf.block_scores(att, head, bid, nb, cnt, "max")
            rho = float(spearmanr(s_mean[elig].numpy(), s_max[elig].numpy())[0])
            o_mean = mf.order_from_scores(s_mean, elig, torch.Generator().manual_seed(n)); o_max = mf.order_from_scores(s_max, elig, torch.Generator().manual_seed(n))
            k_mean = mf.cutoffs(o_mean, cnt, list(KS)); k_max = mf.cutoffs(o_max, cnt, list(KS))
            ref = set_stats(allb, p, cnt, a)
            for k, km, kx in zip(KS, k_mean, k_max):
                A, Bm = set(o_mean[:km].tolist()), set(o_max[:kx].tolist())
                inter = len(A & Bm)
                r = dict(id=cid, method=m, k=k, rho=rho, overlap=inter / max(1, km), jaccard=inter / max(1, len(A | Bm)), chance=kx / int(elig.sum()), lesion_base=base)
                for tag, S in (("mean", o_mean[:km]), ("max", o_max[:kx])):
                    S = S.long()
                    for kk, v in set_stats(S, p, cnt, a).items():
                        r[f"{kk}_{tag}"] = v
                    r[f"headfrac_{tag}"] = float(hfrac[S].median()); r[f"HU_{tag}"] = float(xmean[S].median()); r[f"lesion_{tag}"] = float(lesblk[S].float().mean())
                for kk, v in ref.items():
                    r[f"{kk}_all"] = v
                rows.append(r)
        print(f"  {n + 1}/{len(ids)} {cid}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "rank_compare_cases.csv", index=False)

    log = []
    P = lambda s="": (print(s, flush=True), log.append(s))
    q = lambda v: f"{v.median():.2f} [{v.quantile(.25):.2f}, {v.quantile(.75):.2f}]"
    P(f"{df.id.nunique()} positives. Medians over cases [IQR]. A top set holds k % of the head voxels of the eligible blocks. 'all' = every eligible block (reference).\n")
    P("=== 1. OVERLAP of the top-k % blocks, mean-ranked vs max-ranked (chance = expected overlap of two unrelated rankings) ===")
    P(f"{'method':8s} {'k %':>4s} {'overlap':>20s} {'jaccard':>20s} {'chance':>7s}")
    for m in a.methods:
        for k in KS:
            g = df[(df.method == m) & (df.k == k)]
            if len(g):
                P(f"{m:8s} {k:4d} {q(g.overlap):>20s} {q(g.jaccard):>20s} {g.chance.median():7.2f}")
    P("\nSpearman rho between the mean score and the max score over all eligible blocks (one value per case):")
    for m in a.methods:
        g = df[(df.method == m) & (df.k == KS[0])]
        if len(g):
            P(f"   {m:8s} {q(g.rho)}")
    names = [("neg_vox", "share of voxels with attribution < 0"), ("mat_neg", "share of voxels < -5 % of the case's largest |attribution|"),
             ("mat_pos", "share of voxels > +5 % of it"), ("neg_mass", "negative / (negative + positive) attribution"),
             ("net_gross", "(positive - negative) / (positive + negative)"), ("peak_ratio", "median |most negative voxel| / largest positive voxel, per block")]
    for k in (10, 30):
        P(f"\n=== 2. NEGATIVE CONTENT inside the top-{k} % blocks:  mean-ranked | max-ranked | all eligible blocks ===")
        for key, desc in names:
            P(f"\n-- {key}: {desc}")
            for m in a.methods:
                g = df[(df.method == m) & (df.k == k)]
                if len(g):
                    P(f"   {m:8s} {g[key + '_mean'].median():6.3f} | {g[key + '_max'].median():6.3f} | {g[key + '_all'].median():6.3f}"
                      f"    max - mean (paired, median) {np.median(g[key + '_max'] - g[key + '_mean']):+.3f}")
    P("\n=== 3. OTHER DIFFERENCES of the top-10 % blocks: head fraction | HU | lesion-block share  (mean-ranked | max-ranked);  lesion base rate ===")
    for m in a.methods:
        g = df[(df.method == m) & (df.k == 10)]
        if len(g):
            f = lambda c_: f"{g[c_ + '_mean'].median():.2f} | {g[c_ + '_max'].median():.2f}"
            P(f"   {m:8s} headfrac {f('headfrac'):>13s}   HU {f('HU'):>15s}   lesion {f('lesion'):>13s}   base {g.lesion_base.median():.3f}")
    P("\nReading: a larger negative share / neg mass / lower net-gross in the max column means the max ranking selects blocks whose strong positive peak sits next to strong negative attribution (cancelling); "
      "peak ratio > 1 means the block's strongest voxel is negative. Compare the max column with the 'all' reference, not only with the mean column.")
    (out / "rank_compare.txt").write_text("\n".join(log))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=150)
    for m in a.methods:
        g = df[df.method == m].groupby("k")
        ax[0].plot(list(KS), [g.get_group(k).overlap.median() for k in KS], color=mf.COL[m], marker=mf.MARK[m], label=m, lw=1.6, ms=5)
        ax[1].plot(list(KS), [g.get_group(k).neg_mass_max.median() for k in KS], color=mf.COL[m], marker=mf.MARK[m], lw=1.6, ms=5)
        ax[1].plot(list(KS), [g.get_group(k).neg_mass_mean.median() for k in KS], color=mf.COL[m], ls=":", lw=1.2)
        ax[2].plot(list(KS), [g.get_group(k).net_gross_max.median() for k in KS], color=mf.COL[m], marker=mf.MARK[m], lw=1.6, ms=5)
        ax[2].plot(list(KS), [g.get_group(k).net_gross_mean.median() for k in KS], color=mf.COL[m], ls=":", lw=1.2)
    ax[0].plot(list(KS), [k / 100 for k in KS], color="k", ls="--", lw=1, label="chance")
    ax[0].set_ylabel("overlap, mean-ranked vs max-ranked"); ax[0].set_title("Overlap of the two top sets", fontsize=9)
    ax[1].set_ylabel("negative / (negative + positive) attribution"); ax[1].set_title("Negative content in the top blocks (solid = max-ranked, dotted = mean-ranked)", fontsize=8)
    ax[2].set_ylabel("(positive - negative) / (positive + negative)"); ax[2].set_title("Net / gross in the top blocks (solid = max, dotted = mean)", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    for q_ in ax:
        q_.set_xlabel("top k % of head voxels (blocks)", fontsize=8); q_.spines[["top", "right"]].set_visible(False); q_.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(out / "rank_compare.png"); plt.close(fig)
    print(f"\nwrote {out}/rank_compare_cases.csv, rank_compare.txt, rank_compare.png")


if __name__ == "__main__":
    main()
