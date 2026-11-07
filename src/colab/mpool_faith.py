"""Step 3 (Colab, GPU): FAITHFULNESS of the IG attributions by perturbation: DELETION and INSERTION on the model's actual logit.

Methods: IG_zero, IG_u (constant 40 HU), IG_bl (blur), IG_neg (negative mean)  [Grad-CAM can be added later with --methods GC3 GC4 once its maps are wanted here].
Ranking (the unit is the 20 mm block; eligible = >= 25 % of the block inside the head; two block scores, both run by default):
  mean   signed net attribution per head voxel in the block                (the approved default)
  max    the largest POSITIVE voxel attribution inside the block           (peak evidence)
Blocks are ranked high -> low.  Fraction f % = the highest-ranked blocks that together hold f % of the head voxels of all eligible blocks.
Replacement volumes (written only inside the head; the pooling mask stays fixed to the original head): zero (-20 HU), const (40 HU = 0.5), blur (case blurred, sigma 2 voxels), negmean (training-negative head-conditional mean).
  deletion(f):  the top-f % blocks are replaced.            deletion(0) = original logit.
  insertion(f): ALL eligible blocks are replaced, then the top-f % blocks are restored.   insertion(100) = original logit, insertion(0) = deletion(100).
Control: RANDOM block orders (same permutations for every method and replacement; --random draws per case, default 5).
Optional controls (--controls): reverse = the same ranking read from the lowest-ranked block (label <rank>_rev);  ball = compact random region grown from a random seed block (--ball-draws, default 3).
The scattered random order makes many patchwork boundaries, the compact one very few; ranked sets lie in between, so both are shown.
Output = the actual logit (not a difference) at f = 0:5:100, mean / median / 95 % bootstrap CI (2000, seed 0) over cases.

Usage (needs maskpool_xai.py and mpool_loc.py in the same folder):
  python mpool_faith.py run       --pack PACK --attr DIR [DIR2 ...] --out OUT [--methods ...] [--ranks mean max] [--repl zero const blur negmean] [--grid 0:5:100] [--random 5] [--bs 4] [--max-cases N] [--include-neg]
  python mpool_faith.py summarize --out OUT          (CPU; tables, areas, figures)
`run` is resumable: each case is saved to OUT/faith_cases/<id>.pt and finished configurations are skipped.  Default cases = the label-1 cases that have attributions (35).
"""
import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
B = 8                                  # block = 8 voxels = 20 mm (same as mpool_loc)
METHOD_KEYS = {"IG_zero": "IG_zero", "IG_u": "IG_u", "IG_bl": "IG_bl", "IG_neg": "IG_neg", "GC3": "GradCAM_s3", "GC4": "GradCAM_s4"}
REPL_OF = {"zero": "IG_zero", "const": "IG_u", "blur": "IG_bl", "negmean": "IG_neg"}      # name of the matching entry of make_baselines
COL = {"random_ball": "#555555", "IG_zero": "#999999", "IG_u": "#E69F00", "IG_bl": "#D55E00", "IG_neg": "#009E73", "GC3": "#56B4E9", "GC4": "#0072B2", "random": "#000000"}
MARK = {"random_ball": None, "IG_zero": "o", "IG_u": "s", "IG_bl": "^", "IG_neg": "D", "GC3": "v", "GC4": "P", "random": None}


def parse_grid(s):
    a, b, c = (int(v) for v in s.split(":"))
    return list(range(a, c + 1, b))


# ------------------------------------------------------------------ ranking
def block_scores(m, head, bid, nb, cnt, rank):
    import mpool_loc as ml
    if rank == "mean":
        return ml.block_mean(m, head, bid, nb, cnt)
    vals = (m.clamp(min=0) * head).flatten().float()
    return torch.zeros(nb).scatter_reduce(0, bid.flatten(), vals, "amax", include_self=True)


def order_from_scores(s, elig, gen):
    s = s.double() + torch.rand(len(s), generator=gen).double() * 1e-9 * (s.abs().max() + 1e-12)
    s[~elig] = float("-inf")
    return torch.argsort(s, descending=True)[:int(elig.sum())]


def ball_order(nb, elig, head_shape, gen):
    """eligible blocks ordered by distance from a random eligible seed block: a compact growing region (most clustered control)"""
    D, H, W = head_shape
    nh, nw = (H + B - 1) // B, (W + B - 1) // B
    b = torch.arange(nb)
    iz, iy, ix = b // (nh * nw), (b // nw) % nh, b % nw
    cand = torch.nonzero(elig).flatten()
    s = cand[int(torch.randint(len(cand), (1,), generator=gen))]
    dist = ((iz - iz[s]) ** 2 + (iy - iy[s]) ** 2 + (ix - ix[s]) ** 2).double().sqrt() + torch.rand(nb, generator=gen).double() * 1e-6
    dist[~elig] = float("inf")
    return torch.argsort(dist)[:int(elig.sum())]


def cutoffs(order, cnt, fracs):
    """number of blocks k(f) that hold f % of the head voxels of the eligible blocks"""
    cum = torch.cumsum(cnt[order].double(), 0); tot = float(cum[-1])
    out = []
    for f in fracs:
        out.append(0 if f <= 0 else len(order) if f >= 100 else min(len(order), int(torch.searchsorted(cum, torch.tensor([f / 100 * tot * (1 - 1e-9)], dtype=torch.double))) + 1))
    return out


@torch.no_grad()
def logits_of(model, vols, head, bs):
    out = []
    for i in range(0, len(vols), bs):
        out += model(torch.cat(vols[i:i + bs], 0), head=head).float().flatten().tolist()
    return out


def curve(model, x, repl, full, E, bid, head, order, ks, mode, orig, fullL, fracs, bs):
    """actual logits at the fractions; the endpoints that are known exactly are not recomputed"""
    vols, idx = [], []
    res = [None] * len(fracs)
    for j, (f, k) in enumerate(zip(fracs, ks)):
        if mode == "deletion" and f <= 0:
            res[j] = orig; continue
        if mode == "deletion" and f >= 100:
            res[j] = fullL; continue
        if mode == "insertion" and f <= 0:
            res[j] = fullL; continue
        if mode == "insertion" and f >= 100:
            res[j] = orig; continue
        sel = torch.zeros(int(bid.max()) + 1, dtype=torch.bool, device=bid.device); sel[order[:k].to(bid.device)] = True
        vm = sel[bid] & head[0, 0]
        v = torch.where(vm, repl, x) if mode == "deletion" else torch.where(E & ~vm, repl, x)
        vols.append(v[None, None]); idx.append(j)
    for j, l in zip(idx, logits_of(model, vols, head, bs)):
        res[j] = l
    return np.array(res, dtype=np.float32)


# ------------------------------------------------------------------ run
def cmd_run(a):
    import maskpool_xai as mx
    import mpool_loc as ml
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); (out / "faith_cases").mkdir(parents=True, exist_ok=True)
    fracs = parse_grid(a.grid)
    model = mx.load_model(a.pack, dev)
    negdir = a.negvol_dir or str(Path(a.attr[0]).parent)
    negvol = mx.negmean_volume(a.pack, negdir)
    ids = ml.case_ids(a.attr)
    ids = [i for i in ids if a.include_neg or mx.load_case(a.pack, i)["label"] == 1]
    if a.max_cases:
        ids = ids[:a.max_cases]
    print(f"{len(ids)} cases on {dev}; methods {a.methods}; ranks {a.ranks}; replacements {a.repl}; grid {fracs}; random draws {a.random}", flush=True)
    t0, worst = time.time(), 0.0
    for n, cid in enumerate(ids):
        c = mx.load_case(a.pack, cid)
        d_attr = ml.load_attr(a.attr, cid)
        f = out / "faith_cases" / f"{cid}.pt"
        S = torch.load(f, weights_only=False) if f.exists() else {"id": cid, "label": c["label"], "fracs": fracs, "curves": {}, "random": {}, "full": {}}
        if S.get("fracs") != fracs:
            S = {"id": cid, "label": c["label"], "fracs": fracs, "curves": {}, "random": {}, "full": {}}
        x, head = c["x"][None].to(dev), c["head"][None, None].to(dev)
        bases = mx.make_baselines(c, negvol)
        bid, nb, elig, cnt = ml.block_stats(c["head"])
        E = (elig[bid] & c["head"]).to(dev)
        bid_d = bid.to(dev)
        orig = float(model(x, head=head))
        S["orig"] = orig
        if "logit" in d_attr:
            worst = max(worst, abs(orig - float(d_attr["logit"])))
        for r in a.repl:
            repl = bases[REPL_OF[r]][0].to(dev)
            full = torch.where(E, repl, x[0, 0])
            if r not in S["full"]:
                S["full"][r] = logits_of(model, [full[None, None]], head, 1)[0]
            R = repl
            gen_r = torch.Generator().manual_seed(1000 + n)
            for mode in ("deletion", "insertion"):
                for d in range(a.random):
                    key = f"{r}|{mode}"
                    if key in S["random"] and S["random"][key].shape[0] > d:
                        continue
                    g = torch.Generator().manual_seed(7919 * n + d)
                    order = torch.randperm(nb, generator=g)
                    order = order[elig[order]]
                    ks = cutoffs(order, cnt, fracs)
                    cur = curve(model, x[0, 0], R, full, E, bid_d, head, order, ks, mode, orig, S["full"][r], fracs, a.bs)
                    prev = S["random"].get(key)
                    S["random"][key] = cur[None] if prev is None else np.vstack([prev, cur[None]])
            if "ball" in a.controls:
                for mode in ("deletion", "insertion"):
                    key = f"{r}|{mode}|ball"
                    for d in range(a.ball_draws):
                        if key in S["random"] and S["random"][key].shape[0] > d:
                            continue
                        order = ball_order(nb, elig, tuple(c["head"].shape), torch.Generator().manual_seed(31 * n + d))
                        ks = cutoffs(order, cnt, fracs)
                        cur = curve(model, x[0, 0], R, full, E, bid_d, head, order, ks, mode, orig, S["full"][r], fracs, a.bs)
                        prev = S["random"].get(key)
                        S["random"][key] = cur[None] if prev is None else np.vstack([prev, cur[None]])
            for m in a.methods:
                if METHOD_KEYS[m] not in d_attr:
                    continue
                att = d_attr[METHOD_KEYS[m]].float() * c["head"]
                for rk in a.ranks:
                    sc = block_scores(att, c["head"], bid, nb, cnt, rk)
                    order = order_from_scores(sc, elig, torch.Generator().manual_seed(n))
                    variants = [(rk, order)] + ([(rk + "_rev", order.flip(0))] if "reverse" in a.controls else [])
                    for rname, od in variants:
                        ks = cutoffs(od, cnt, fracs)
                        for mode in ("deletion", "insertion"):
                            key = f"{m}|{rname}|{r}|{mode}"
                            if key in S["curves"]:
                                continue
                            S["curves"][key] = curve(model, x[0, 0], R, full, E, bid_d, head, od, ks, mode, orig, S["full"][r], fracs, a.bs)
        torch.save(S, f)
        el = time.time() - t0
        print(f"  {n + 1}/{len(ids)} {cid}  ({el:.0f}s, ~{el / (n + 1) * (len(ids) - n - 1) / 60:.0f} min left)", flush=True)
    print(f"max |recomputed logit - stored logit| over cases: {worst:.2e}", flush=True)


# ------------------------------------------------------------------ summarize
def boot_ci_curve(M, n=2000, seed=0):
    """M: (cases, fractions) -> mean, lo, hi per fraction (bootstrap over cases)"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M.shape[0], (n, M.shape[0]))
    bs = M[idx].mean(1)
    return M.mean(0), np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def cmd_summarize(a):
    from scipy.stats import friedmanchisquare, wilcoxon
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out)
    files = sorted((out / "faith_cases").glob("*.pt"))
    cases = [torch.load(f, weights_only=False) for f in files]
    fr = np.array(cases[0]["fracs"], float)
    keys = sorted({k for c in cases for k in c["curves"]})
    methods = sorted({k.split("|")[0] for k in keys}, key=lambda m: list(METHOD_KEYS).index(m))
    ranks = sorted({k.split("|")[1] for k in keys}); repls = sorted({k.split("|")[2] for k in keys}, key=lambda r: list(REPL_OF).index(r))
    log, rows, arows = [], [], []
    P = lambda s="": (print(s, flush=True), log.append(s))
    P(f"{len(cases)} cases ({sum(c['label'] == 1 for c in cases)} positive); methods {methods}; ranks {ranks}; replacements {repls}; grid {fr.astype(int).tolist()}")
    P(f"mean original logit {np.mean([c['orig'] for c in cases]):+.3f} (median {np.median([c['orig'] for c in cases]):+.3f}); fully replaced (all eligible blocks): "
      + ", ".join(f"{r} {np.mean([c['full'][r] for c in cases if r in c['full']]):+.2f}" for r in repls))
    trap = getattr(np, 'trapezoid', None) or np.trapz
    area = lambda M: trap(M, fr, axis=1) / (fr[-1] - fr[0])          # mean logit along the curve

    def stack(key, rnd=False):
        L = []
        for c in cases:
            src = c["random"] if rnd else c["curves"]
            if key in src:
                v = src[key]; L.append(v.mean(0) if rnd else v)
        return np.array(L)

    for mode in ("deletion", "insertion"):
        for r in repls:
            R = stack(f"{r}|{mode}", rnd=True)
            if len(R):
                m_, lo, hi = boot_ci_curve(R)
                for j, f in enumerate(fr):
                    rows.append(dict(mode=mode, repl=r, rank="-", method="random", frac=f, n=len(R), mean=m_[j], ci_lo=lo[j], ci_hi=hi[j], median=np.median(R[:, j])))
            Bl = stack(f"{r}|{mode}|ball", rnd=True)
            if len(Bl):
                m_, lo, hi = boot_ci_curve(Bl)
                for j, f in enumerate(fr):
                    rows.append(dict(mode=mode, repl=r, rank="-", method="random_ball", frac=f, n=len(Bl), mean=m_[j], ci_lo=lo[j], ci_hi=hi[j], median=np.median(Bl[:, j])))
            for rk in ranks:
                A = {}
                for m in methods:
                    M = stack(f"{m}|{rk}|{r}|{mode}")
                    if not len(M):
                        continue
                    A[m] = M
                    m_, lo, hi = boot_ci_curve(M)
                    for j, f in enumerate(fr):
                        rows.append(dict(mode=mode, repl=r, rank=rk, method=m, frac=f, n=len(M), mean=m_[j], ci_lo=lo[j], ci_hi=hi[j], median=np.median(M[:, j])))
                    ar = area(M); lo_a, hi_a = np.percentile(ar[np.random.default_rng(0).integers(0, len(ar), (2000, len(ar)))].mean(1), [2.5, 97.5])
                    rr = area(R) if len(R) == len(M) else None
                    p = np.nan
                    if rr is not None:
                        try:
                            p = float(wilcoxon(ar, rr).pvalue)
                        except Exception:
                            pass
                    rb = area(Bl) if len(Bl) == len(M) else None
                    pb = np.nan
                    if rb is not None:
                        try:
                            pb = float(wilcoxon(ar, rb).pvalue)
                        except Exception:
                            pass
                    arows.append(dict(mode=mode, repl=r, rank=rk, method=m, n=len(M), area=ar.mean(), ci_lo=lo_a, ci_hi=hi_a, area_random=rr.mean() if rr is not None else np.nan,
                                      diff_vs_random=(ar - rr).mean() if rr is not None else np.nan, p_vs_random=p,
                                      area_ball=rb.mean() if rb is not None else np.nan, diff_vs_ball=(ar - rb).mean() if rb is not None else np.nan, p_vs_ball=pb))
    pd.DataFrame(rows).to_csv(out / "faith_curves_summary.csv", index=False)
    A = pd.DataFrame(arows); A.to_csv(out / "faith_area_summary.csv", index=False)
    P("\n=== AREA under the logit curve (mean logit along 0-100 %; deletion: LOWER = removing the top blocks hurts more = better; insertion: HIGHER = better) ===")
    P("diff = area(method) - area(random), paired over cases; p = paired Wilcoxon, UNADJUSTED.  Random curves are NOT flat: replacement volumes are not neutral.")
    for mode in ("deletion", "insertion"):
        for rk in ranks:
            for r in repls:
                g = A[(A["mode"] == mode) & (A["rank"] == rk) & (A.repl == r)]
                if g.empty:
                    continue
                P(f"\n-- {mode}, rank = {rk}, replacement = {r}   (random area {g.area_random.iloc[0]:+.3f})")
                for _, q in g.iterrows():
                    extra = f"   | vs compact-random (area {q.area_ball:+.3f}): {q.diff_vs_ball:+.3f}  p = {q.p_vs_ball:.3g}" if not np.isnan(q.area_ball) else ""
                    P(f"   {q.method:8s} n={int(q.n):3d} area {q.area:+.3f} [{q.ci_lo:+.3f}, {q.ci_hi:+.3f}]   diff vs random {q.diff_vs_random:+.3f}  p = {q.p_vs_random:.3g}{extra}")
    P("\n=== IG baselines compared on the AREA (paired, Friedman over the methods present) ===")
    for mode in ("deletion", "insertion"):
        for rk in ranks:
            for r in repls:
                D = {}
                for m in methods:
                    M = stack(f"{m}|{rk}|{r}|{mode}")
                    if len(M):
                        D[m] = area(M)
                if len(D) >= 3 and len({len(v) for v in D.values()}) == 1:
                    try:
                        p = friedmanchisquare(*D.values()).pvalue
                    except Exception:
                        p = np.nan
                    P(f"   {mode:9s} rank {rk:4s} repl {r:8s} Friedman p = {p:.3g}")
    (out / "faith_summary.txt").write_text("\n".join(log))

    # figures: one per (mode, rank); 4 panels (replacement); mean +- 95 % CI of the actual logit; random = black dashed; same markers for the same method everywhere
    S = pd.DataFrame(rows)
    for mode in ("deletion", "insertion"):
        for rk in ranks:
            fig, ax = plt.subplots(1, len(repls), figsize=(4.2 * len(repls), 3.8), dpi=150, squeeze=False)
            for q, r in zip(ax[0], repls):
                for m in methods + ["random", "random_ball"]:
                    g = S[(S["mode"] == mode) & (S.repl == r) & (S.method == m) & ((S["rank"] == rk) | S.method.isin(["random", "random_ball"]))].sort_values("frac")
                    if g.empty:
                        continue
                    q.plot(g.frac, g["mean"], color=COL[m], lw=1.6 if not m.startswith("random") else 1.2, ls={"random": "--", "random_ball": ":"}.get(m, "-"), marker=MARK[m], ms=4,
                           label={"random": "random (scattered)", "random_ball": "random (compact)"}.get(m, m), markevery=2)
                    q.fill_between(g.frac, g.ci_lo, g.ci_hi, color=COL[m], alpha=.12, lw=0)
                q.set_title(f"{mode}, replacement = {r}", fontsize=9); q.set_xlabel(f"{'removed' if mode == 'deletion' else 'restored'} (% of head voxels, {rk}-ranked blocks)", fontsize=8)
                q.set_ylabel("actual logit (mean, 95 % CI)", fontsize=8); q.tick_params(labelsize=8); q.spines[["top", "right"]].set_visible(False)
            ax[0, 0].legend(frameon=False, fontsize=7)
            fig.tight_layout(); fig.savefig(out / f"faith_{mode}_{rk}.png"); plt.close(fig)
    P(f"\nwrote {out}/faith_curves_summary.csv, faith_area_summary.csv, faith_summary.txt, faith_<mode>_<rank>.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--pack"); ap.add_argument("--attr", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["IG_zero", "IG_u", "IG_bl", "IG_neg"])
    ap.add_argument("--ranks", nargs="+", default=["mean", "max"], choices=["mean", "max"])
    ap.add_argument("--repl", nargs="+", default=["zero", "const", "blur", "negmean"], choices=list(REPL_OF))
    ap.add_argument("--grid", default="0:5:100"); ap.add_argument("--random", type=int, default=5)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--include-neg", action="store_true"); ap.add_argument("--negvol-dir", default=None)
    ap.add_argument("--controls", nargs="*", default=[], choices=["reverse", "ball"], help="reverse = lowest-ranked blocks first (per method and rank, label <rank>_rev); ball = compact random region growing from a random seed block")
    ap.add_argument("--ball-draws", type=int, default=3)
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
