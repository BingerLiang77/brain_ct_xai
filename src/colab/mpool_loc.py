"""Step 2 (Colab or CPU): LOCALIZATION of the masked-pool model's attributions on the positives, head-only.

Methods (any that are missing from the attribution files are skipped): GC3, GC4 (Grad-CAM stages 3/4), IG_zero, IG_u (constant 40 HU), IG_bl (blur), IG_neg (negative mean).
Controls: analytic chance (no simulation) and the method's own leave-one-out POPULATION PRIOR (location only; same block machinery as before).

Metrics (per positive case; lesion := lesion AND head; cases with empty lesion-in-head are excluded and listed):
  pointing   peak voxel of the map (inside head) lies within r voxels of the lesion, r = 0 (exact), 2 (5 mm), 4 (10 mm); chance = share of head voxels within r of the lesion.
  energy     enrichment ratio = (share of positive attribution inside the lesion) / (lesion volume / head volume); 1 = chance.  Voxel contrast d = (mean in lesion - mean in rest of head) / SD in head.
  dice       Dice of the top-N head voxels with N = lesion voxel count (so Dice = precision = recall); chance = N / head volume.  Also Dice at the top 5 % of head voxels (dice5).
  block      lesion-block AUC (8-voxel = 20 mm blocks, eligible >= 25 % head, lesion block >= 5 lesion voxels) and gain over the prior's AUC.
Voxel metrics use the POSITIVE part of the map (Grad-CAM is already >= 0; IG is clipped at 0); block metrics use the signed block mean for IG (the approved ordering), the ReLU block mean for Grad-CAM.

Usage:
  python mpool_loc.py --pack PACK --attr DIR [DIR2 ...] --out OUT [--max-cases N]
  Several --attr dirs are merged per case (later dirs override keys of earlier ones), e.g. the first run for IG_u/IG_bl/IG_neg/Grad-CAM and the v2 run for IG_zero.
Writes OUT/loc_cases.csv, OUT/loc_summary.csv, OUT/loc_summary.txt, OUT/loc_figure.png.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
B = 8                                  # block = 8 voxels = 20 mm
RADII = (0, 2, 4)
METHODS = {"GC3": ("GradCAM_s3", "relu"), "GC4": ("GradCAM_s4", "relu"), "IG_zero": ("IG_zero", "net"),
           "IG_u": ("IG_u", "net"), "IG_bl": ("IG_bl", "net"), "IG_neg": ("IG_neg", "net")}
IGS = ["IG_zero", "IG_u", "IG_bl", "IG_neg"]


# ------------------------------------------------------------------ data
def load_case(pack, cid):
    d = torch.load(Path(pack) / "data" / "cases" / f"{cid}.pt", weights_only=False)
    return dict(head=d["head"].bool(), lesion=d["lesion"].bool(), label=int(d["meta"]["label"]))


def load_attr(dirs, cid):
    out = {}
    for dd in dirs:
        f = Path(dd) / f"{cid}.pt"
        if f.exists():
            out.update(torch.load(f, weights_only=True))
    return out


def case_ids(dirs):
    s = set()
    for dd in dirs:
        s |= {p.stem for p in Path(dd).glob("*.pt")}
    return sorted(s)


# ------------------------------------------------------------------ blocks (same machinery as the earlier analysis)
def block_stats(head):
    D, H, W = head.shape
    iz, iy, ix = torch.meshgrid(torch.arange(D) // B, torch.arange(H) // B, torch.arange(W) // B, indexing="ij")
    nh, nw = (H + B - 1) // B, (W + B - 1) // B
    bid = (iz * nh + iy) * nw + ix
    nb = int(bid.max()) + 1
    cnt = torch.zeros(nb).scatter_add_(0, bid.flatten(), head.flatten().float())
    size = torch.zeros(nb).scatter_add_(0, bid.flatten(), torch.ones(bid.numel()))
    return bid, nb, cnt >= 0.25 * size, cnt


def block_mean(m, head, bid, nb, cnt):
    s = torch.zeros(nb).scatter_add_(0, bid.flatten(), (m * head).flatten().float())
    return s / cnt.clamp(min=1)


def zscore(s, elig):
    v = s[elig]
    return torch.where(elig, (s - v.mean()) / v.std().clamp(min=1e-9), torch.zeros_like(s))


def block_auc(score, elig, lesblk):
    y, s = lesblk[elig].numpy(), score[elig].numpy()
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


# ------------------------------------------------------------------ voxel metrics
def topn_mask(m, head, n, gen):
    """Boolean mask of the n highest head voxels (random tie-break, so flat / zero regions are not ordered by memory layout)."""
    v = m[head].double()
    v = v + torch.rand(len(v), generator=gen).double() * 1e-9 * (v.abs().max() + 1e-12)
    idx = torch.topk(v, min(n, len(v))).indices
    sel = torch.zeros(len(v), dtype=torch.bool); sel[idx] = True
    out = torch.zeros_like(head); out[head] = sel
    return out


def voxel_metrics(m, head, les, dist, gen):
    """m: map >= 0 (float), head/les bool, dist: Euclidean distance (voxels) of every voxel to the lesion."""
    nH, nL = int(head.sum()), int(les.sum())
    r = {}
    mh = m.clone(); mh[~head] = -1.0
    flat = int(torch.argmax(mh + torch.rand(m.shape, generator=gen) * 1e-9 * (float(m[head].max()) + 1e-12)))
    peak_ok = float(m[head].max()) > 0
    pz = np.unravel_index(flat, tuple(m.shape))
    for rr in RADII:
        r[f"point{rr}"] = float(peak_ok and dist[pz] <= rr)
        r[f"point{rr}_chance"] = float(((dist <= rr) & head.numpy()).sum() / nH)
    tot = float(m[head].sum())
    r["energy_share"] = float(m[les].sum()) / tot if tot > 0 else np.nan
    r["energy_ratio"] = r["energy_share"] / (nL / nH) if tot > 0 else np.nan
    sd = float(m[head].std())
    r["vox_d"] = float((m[les].mean() - m[head & ~les].mean()) / sd) if sd > 0 else np.nan
    sel = topn_mask(m, head, nL, gen)
    r["dice"] = float((sel & les).sum()) * 2 / (int(sel.sum()) + nL)
    r["dice_chance"] = nL / nH
    k5 = int(0.05 * nH)
    sel5 = topn_mask(m, head, k5, gen)
    r["dice5"] = float((sel5 & les).sum()) * 2 / (int(sel5.sum()) + nL)
    r["dice5_chance"] = (2 * nL * k5 / nH) / (k5 + nL)
    return r


# ------------------------------------------------------------------ stats helpers
def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    try:
        return float(wilcoxon(a[ok], b[ok]).pvalue)
    except Exception:
        return np.nan


def fmt(x, p=3):
    return "nan" if x is None or np.isnan(x) else f"{x:.{p}f}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--attr", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    log = []
    P = lambda s="": (print(s, flush=True), log.append(s))

    # ---- select positives with lesion inside the head
    ids, excl, info = [], [], {}
    for cid in case_ids(a.attr):
        c = load_case(a.pack, cid)
        if c["label"] != 1:
            continue
        les = c["lesion"] & c["head"]
        frac = float(les.sum() / max(1, c["lesion"].sum()))
        if les.sum() == 0:
            excl.append(cid); continue
        ids.append(cid); info[cid] = dict(frac_in_head=frac, lesion_vox=int(les.sum()), head_vox=int(c["head"].sum()))
    if a.max_cases:
        ids = ids[:a.max_cases]
    P(f"{len(ids)} positives with lesion inside the head; excluded (empty lesion-in-head): {len(excl)}  {excl}")
    P(f"lesion fraction inside head: median {np.median([info[i]['frac_in_head'] for i in ids]):.3f}; "
      f"< 0.9: {sum(info[i]['frac_in_head'] < 0.9 for i in ids)}; < 0.5: {sum(info[i]['frac_in_head'] < 0.5 for i in ids)}  (lesion := lesion AND head throughout)")

    # ---- pass 1: block scores -> leave-one-out priors
    blk, avail = {}, set()
    for cid in ids:
        c, d = load_case(a.pack, cid), load_attr(a.attr, cid)
        bid, nb, elig, cnt = block_stats(c["head"])
        les = c["lesion"] & c["head"]
        lcnt = torch.zeros(nb).scatter_add_(0, bid.flatten(), les.flatten().float())
        sc = {}
        for n, (k, kind) in METHODS.items():
            if k in d:
                m = d[k].float(); m = m.clamp(min=0) if kind == "relu" else m
                sc[n] = block_mean(m, c["head"], bid, nb, cnt); avail.add(n)
        blk[cid] = dict(bid=bid, nb=nb, elig=elig, sc=sc, lesblk=(lcnt >= 5) & elig)
    methods = [n for n in METHODS if n in avail]
    P(f"methods found: {methods}")
    prior = {}
    for n in methods:
        cs = [i for i in ids if n in blk[i]["sc"]]
        Z = torch.stack([zscore(blk[i]["sc"][n], blk[i]["elig"]) for i in cs]); E = torch.stack([blk[i]["elig"].float() for i in cs])
        for j, i in enumerate(cs):
            prior[(i, n)] = (Z.sum(0) - Z[j]) / (E.sum(0) - E[j]).clamp(min=1)

    # ---- pass 2: per-case metrics for each method and for its own prior
    rows, comp = [], []
    for cid in ids:
        c, d = load_case(a.pack, cid), load_attr(a.attr, cid)
        head, les = c["head"], c["lesion"] & c["head"]
        dist = distance_transform_edt(~les.numpy())
        b = blk[cid]
        for n in methods:
            k, kind = METHODS[n]
            if k not in d:
                continue
            m = d[k].float(); m = m.clamp(min=0)
            pm = prior[(cid, n)][b["bid"]]
            for tag, mm in (("method", m), ("prior", pm - pm[head].min())):   # prior is a z-score (has negatives): shift to >= 0 so the energy ratio is defined
                r = voxel_metrics(mm, head, les, dist, gen)
                s = b["sc"][n] if tag == "method" else prior[(cid, n)]
                r["block_auc"] = block_auc(s, b["elig"], b["lesblk"])
                rows.append(dict(id=cid, method=n, kind=tag, frac_in_head=info[cid]["frac_in_head"], lesion_vox=info[cid]["lesion_vox"],
                                 lesion_share_of_head=info[cid]["lesion_vox"] / info[cid]["head_vox"], **r))
            if n.startswith("IG") and f"{n}_err" in d:
                gap = abs(float(d["logit"]) - float(d[f"{n}_base_logit"]))
                comp.append(dict(method=n, rel_err=abs(float(d[f"{n}_err"])) / max(gap, 1e-9)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "loc_cases.csv", index=False)

    # ---- summary
    P("\nIG completeness (relative error |sum - (logit - base)| / |logit - base|, median over cases):")
    if comp:
        P(pd.DataFrame(comp).groupby("method").rel_err.median().round(3).to_string())
    M = df[df.kind == "method"].set_index(["id", "method"]); R = df[df.kind == "prior"].set_index(["id", "method"])
    summ = []
    metrics = [("point0", "pointing r=0 (exact peak)"), ("point2", "pointing r=2 (5 mm)"), ("point4", "pointing r=4 (10 mm)"),
               ("energy_ratio", "energy ratio (1 = chance)"), ("vox_d", "voxel contrast d"),
               ("dice", "Dice, top-N (N = lesion voxels)"), ("dice5", "Dice, top 5 % of head"), ("block_auc", "block AUC (lesion vs other blocks)")]
    P("\n=== LOCALIZATION, positives (mean [bootstrap 95 % CI], median). chance = analytic; prior = the method's own leave-one-out population prior. "
      "p = paired Wilcoxon method vs prior, UNADJUSTED ===")
    for mk, title in metrics:
        P(f"\n-- {title}")
        ch = {"point0": "point0_chance", "point2": "point2_chance", "point4": "point4_chance", "dice": "dice_chance", "dice5": "dice5_chance"}.get(mk)
        P(f"{'method':8s} {'n':>3s} {'mean [CI]':>24s} {'median':>8s} {'prior mean':>11s} {'diff vs prior':>14s} {'p':>8s}" + (f" {'chance':>8s}" if ch else ""))
        for n in methods:
            x = M.xs(n, level="method")[mk]; y = R.xs(n, level="method")[mk].reindex(x.index)
            lo, hi = boot_ci(x.values)
            line = f"{n:8s} {len(x):3d} {fmt(x.mean()):>7s} [{fmt(lo)}, {fmt(hi)}] {fmt(x.median()):>8s} {fmt(y.mean()):>11s} {fmt((x - y).mean()):>14s} {fmt(wil(x.values, y.values), 3):>8s}"
            if ch:
                line += f" {fmt(M.xs(n, level='method')[ch].mean()):>8s}"
            P(line)
            summ.append(dict(metric=mk, method=n, n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi, median=x.median(), prior_mean=y.mean(),
                             diff_vs_prior=(x - y).mean(), p_vs_prior=wil(x.values, y.values), chance=M.xs(n, level="method")[ch].mean() if ch else np.nan))
    # ---- across the IG baselines (paired)
    igs = [n for n in IGS if n in methods]
    if len(igs) >= 2:
        P("\n=== IG BASELINES compared (paired over the same positives): Friedman test, then pairwise Wilcoxon (unadjusted) ===")
        for mk in ("energy_ratio", "dice", "block_auc", "point2"):
            W = pd.concat({n: M.xs(n, level="method")[mk] for n in igs}, axis=1).dropna()
            if len(igs) >= 3 and len(W) > 5:
                try:
                    pf = friedmanchisquare(*[W[n].values for n in igs]).pvalue
                except Exception:
                    pf = np.nan
            else:
                pf = np.nan
            P(f"\n{mk}: n = {len(W)}, Friedman p = {fmt(pf, 3)}")
            for i, n1 in enumerate(igs):
                for n2 in igs[i + 1:]:
                    P(f"   {n1:8s} - {n2:8s}: mean diff {fmt((W[n1] - W[n2]).mean()):>7s}   p = {fmt(wil(W[n1].values, W[n2].values), 3)}")
    # ---- sensitivity: lesion mostly in head; size strata
    for title, sub in (("lesion >= 90 % inside head", df[df.frac_in_head >= 0.9]),):
        P(f"\n=== SENSITIVITY: {title} (n = {sub.id.nunique()} positives); mean energy ratio / Dice / point2 ===")
        for n in methods:
            g = sub[(sub.kind == 'method') & (sub.method == n)]
            P(f"   {n:8s} energy {fmt(g.energy_ratio.mean(), 2)}  dice {fmt(g.dice.mean())}  point2 {fmt(g.point2.mean(), 2)}")
    P("\n=== LESION SIZE: Spearman(metric, lesion share of head) and metric by size tertile (small / mid / large) ===")
    for mk in ("dice", "energy_ratio", "point2"):
        for n in methods:
            g = M.xs(n, level="method")
            rs = spearmanr(g.lesion_share_of_head, g[mk], nan_policy="omit")[0]
            t = pd.qcut(g.lesion_share_of_head, 3, labels=["S", "M", "L"])
            P(f"   {mk:12s} {n:8s} rho {fmt(rs, 2):>6s}   " + "  ".join(f"{k} {fmt(v, 3)}" for k, v in g[mk].groupby(t).mean().items()))
    P("\nReading: values are against the same-case chance level (analytic) and against the method's own location-only prior. 35 positives, p-values unadjusted. "
      "Positives with lesions outside the brain mask were removed upstream (selection toward fewer skull-adjacent SDH/EDH); Dice depends strongly on lesion size.")
    (out / "loc_summary.txt").write_text("\n".join(log))
    pd.DataFrame(summ).to_csv(out / "loc_summary.csv", index=False)
    figure(df, methods, out / "loc_figure.png")
    P(f"\nwrote {out}/loc_cases.csv, loc_summary.csv, loc_summary.txt, loc_figure.png")


# ------------------------------------------------------------------ figure (Okabe-Ito; Grad-CAM = blue family, IG = warm family; fixed order)
COL = {"GC3": "#56B4E9", "GC4": "#0072B2", "IG_zero": "#999999", "IG_u": "#E69F00", "IG_bl": "#D55E00", "IG_neg": "#009E73"}


def figure(df, methods, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = df[df.kind == "method"]; R = df[df.kind == "prior"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    x = np.arange(len(methods))
    # (a) pointing
    a = ax[0, 0]; w = 0.26
    for j, rr in enumerate(RADII):
        for i, n in enumerate(methods):
            v = M[M.method == n][f"point{rr}"].values; lo, hi = boot_ci(v)
            a.bar(i + (j - 1) * w, v.mean(), w * 0.9, color=COL[n], alpha=[1, 0.65, 0.35][j], yerr=[[v.mean() - lo], [hi - v.mean()]], error_kw=dict(lw=1, capsize=2))
        ch = M[M.method == methods[0]][f"point{rr}_chance"].mean()
        a.axhline(ch, color="k", lw=0.8, ls=(0, (2, 2)))
    a.set_xticks(x); a.set_xticklabels(methods, fontsize=8); a.set_ylabel("peak inside lesion (hit rate)"); a.set_title("Pointing game: r = 0 / 2 / 4 voxels (dark to light); dashed = chance", fontsize=9)
    # (b) energy ratio
    a = ax[0, 1]
    for i, n in enumerate(methods):
        v = M[M.method == n].energy_ratio.dropna().values
        a.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.18, .18, len(v)), v, s=12, color=COL[n], alpha=.6, edgecolor="none")
        lo, hi = boot_ci(v); a.plot([i, i], [lo, hi], color="k", lw=2); a.plot(i, v.mean(), "o", color="white", mec="k", ms=6)
        pv = R[R.method == n].energy_ratio.mean(); a.plot([i - .3, i + .3], [pv, pv], color="k", lw=1, ls=":")
    a.axhline(1, color="k", lw=0.8, ls=(0, (2, 2))); a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(methods, fontsize=8)
    a.set_ylabel("energy ratio (log)"); a.set_title("Lesion enrichment: mean + 95 % CI; dotted = prior; dashed = chance", fontsize=9)
    # (c) dice
    a = ax[1, 0]
    for i, n in enumerate(methods):
        v = M[M.method == n].dice.values
        a.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-.18, .18, len(v)), v, s=12, color=COL[n], alpha=.6, edgecolor="none")
        lo, hi = boot_ci(v); a.plot([i, i], [lo, hi], color="k", lw=2); a.plot(i, v.mean(), "o", color="white", mec="k", ms=6)
        pv = R[R.method == n].dice.mean(); a.plot([i - .3, i + .3], [pv, pv], color="k", lw=1, ls=":")
    a.axhline(M[M.method == methods[0]].dice_chance.mean(), color="k", lw=0.8, ls=(0, (2, 2)))
    a.set_xticks(x); a.set_xticklabels(methods, fontsize=8); a.set_ylabel("Dice (top-N voxels)"); a.set_title("Dice, N = lesion size: mean + 95 % CI; dotted = prior; dashed = chance", fontsize=9)
    # (d) block AUC gain
    a = ax[1, 1]
    for i, n in enumerate(methods):
        g = (M[M.method == n].set_index("id").block_auc - R[R.method == n].set_index("id").block_auc).dropna().values
        a.scatter(np.full(len(g), i) + np.random.default_rng(2).uniform(-.18, .18, len(g)), g, s=12, color=COL[n], alpha=.6, edgecolor="none")
        lo, hi = boot_ci(g); a.plot([i, i], [lo, hi], color="k", lw=2); a.plot(i, g.mean(), "o", color="white", mec="k", ms=6)
    a.axhline(0, color="k", lw=0.8, ls=(0, (2, 2))); a.set_xticks(x); a.set_xticklabels(methods, fontsize=8)
    a.set_ylabel("block AUC - prior AUC"); a.set_title("Lesion specificity beyond location (0 = location only)", fontsize=9)
    for q in ax.flat:
        q.spines[["top", "right"]].set_visible(False); q.tick_params(labelsize=8)
    fig.suptitle(f"Localization of the masked-pool model's attributions on {M.id.nunique()} positives (head only)", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
