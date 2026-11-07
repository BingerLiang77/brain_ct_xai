"""How much of the masked-pool logit comes from each encoder stage, and how much of it a Grad-CAM at stage 3 / stage 4 can see.

The head is linear on standardised, head-masked MEAN-pooled stage features, so the logit splits EXACTLY by stage:
    logit = b + sum_s c_s,    c_s = sum_{channels of stage s} w_c * (z_c - mean_c) / std_c
(c_s = contribution of stage s relative to the average training case).
Grad-CAM at stage k only sees the paths through stage k's feature map: stage 3 sees c_3 + c_4 (stage 4 is computed from stage 3), stage 4 sees c_4 only;
c_0..c_2 (and c_3 for a stage-4 CAM) bypass it.  IG starts at the input and sees everything.

Usage:  python mpool_stage_decomp.py --pack PACK --out OUT [--all] [--max-cases N]
  default: the 55 cases that have attributions in PACK/ref/attributions;  --all: every case in PACK/data/cases (forward passes only; use a GPU).
Writes OUT/stage_contrib.csv, OUT/stage_decomp.txt, OUT/stage_decomp.png.   Needs maskpool_xai.py in the same folder.
"""
import argparse
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import maskpool_xai as mx

warnings.filterwarnings("ignore")


@torch.no_grad()
def stage_contrib(model, x, head):
    """Returns (logit, [c_0..c_4]) with c_s the stage-s share of the logit (bias excluded)."""
    fs = model.backbone(x)
    m = head.to(x.dtype).expand(x.shape[0], *head.shape[1:])
    zs = []
    for s in model.stages:
        f = fs[s]
        w = F.adaptive_avg_pool3d(m, f.shape[2:])
        zs.append((f * w).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3, 4)).clamp(min=1e-6))
    sizes = [z.shape[1] for z in zs]
    t = ((torch.cat(zs, 1) - model.mean) / model.std) * model.w          # per-channel terms, (1, C)
    cs = [float(p.sum()) for p in torch.split(t, sizes, dim=1)]
    return float(t.sum() + model.b), cs, float(model.b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--max-cases", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mx.load_model(a.pack, dev)
    ids = sorted(p.stem for p in (Path(a.pack) / "data" / "cases").glob("*.pt")) if a.all else mx.xai_ids(a.pack)
    if a.max_cases:
        ids = ids[:a.max_cases]
    print(f"{len(ids)} cases on {dev}", flush=True)
    rows = []
    for n, cid in enumerate(ids):
        c = mx.load_case(a.pack, cid)
        x, head = c["x"][None].to(dev), c["head"][None, None].to(dev)
        logit, cs, b = stage_contrib(model, x, head)
        ref = float(model(x, head=head))
        assert abs(ref - logit) < 1e-3, f"decomposition does not reproduce the logit: {ref} vs {logit}"
        rows.append(dict(id=cid, label=c["label"], logit=logit, bias=b, **{f"c{s}": v for s, v in zip(model.stages, cs)}))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows)
    df["c012"] = df.c0 + df.c1 + df.c2
    df["c34"] = df.c3 + df.c4
    df["gap_total"] = df.logit - df.bias                                    # = sum of c_s
    df.to_csv(out / "stage_contrib.csv", index=False)

    log = []
    P = lambda s="": (print(s, flush=True), log.append(s))
    S = ["c0", "c1", "c2", "c3", "c4"]
    P(f"{len(df)} cases ({int((df.label == 1).sum())} positive, {int((df.label == 0).sum())} negative); decomposition reproduces the model logit to < 1e-3 on every case.")
    P(f"logit = bias ({df.bias.iloc[0]:+.3f}) + sum of the five stage contributions.  Contributions are relative to the average training case (they can be negative).\n")
    P("=== Stage contributions to the logit (mean / SD; positives vs negatives) ===")
    P(f"{'stage':6s} {'all mean':>9s} {'pos mean':>9s} {'neg mean':>9s} {'pos-neg':>9s} {'SD (all)':>9s} {'mean |c|':>9s}")
    pos, neg = df[df.label == 1], df[df.label == 0]
    for s in S + ["c012", "c34"]:
        P(f"{s:6s} {df[s].mean():9.3f} {pos[s].mean():9.3f} {neg[s].mean():9.3f} {pos[s].mean() - neg[s].mean():9.3f} {df[s].std():9.3f} {df[s].abs().mean():9.3f}")
    sep = {s: pos[s].mean() - neg[s].mean() for s in S}
    tot = sum(sep.values())
    P("\n=== Share of the positive-vs-negative separation (mean logit difference) carried by each stage ===")
    P("   " + "  ".join(f"{s}: {sep[s] / tot * 100:5.1f} %" for s in S) + f"     [total separation {tot:.2f} logit]")
    P(f"   stages 0-2 together: {(sep['c0'] + sep['c1'] + sep['c2']) / tot * 100:.1f} %;  stages 3-4 together: {(sep['c3'] + sep['c4']) / tot * 100:.1f} %")

    P("\n=== What a Grad-CAM can see ===")
    absall = df[S].abs().sum(1)
    for tag, vis, miss in (("Grad-CAM stage 3 (sees c3 + c4)", df.c34, df.c012), ("Grad-CAM stage 4 (sees c4 only)", df.c4, df.c0 + df.c1 + df.c2 + df.c3)):
        r = np.corrcoef(vis, df.gap_total)[0, 1]
        P(f"{tag}")
        P(f"   missed part: median |missed| = {miss.abs().median():.3f} logit;  median share of sum|c_s| = {np.median(miss.abs() / absall) * 100:.1f} %")
        P(f"   case-to-case: corr(visible part, logit - bias) = {r:.3f}  (R^2 = {r ** 2:.3f}); visible part alone explains {r ** 2 * 100:.0f} % of the variation of the logit across cases")
        lab = pos if len(pos) else df
        P(f"   positives only: median |missed| = {miss[df.label == 1].abs().median():.3f}, median |visible| = {vis[df.label == 1].abs().median():.3f}")
    P("\nReading: if stages 0-2 carry little of the separation and the visible part tracks the logit (R^2 close to 1), a stage-3 Grad-CAM explains nearly the whole decision; "
      "if not, say in the report that Grad-CAM explains the stage 3-4 part only (IG covers everything).")
    (out / "stage_decomp.txt").write_text("\n".join(log))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    rng = np.random.default_rng(0)
    for k, (d, col, name) in enumerate(((neg, "#0072B2", "negative"), (pos, "#D55E00", "positive"))):
        for i, s in enumerate(S):
            v = d[s].values; xs = i + (k - .5) * 0.34 + rng.uniform(-.06, .06, len(v))
            ax[0].scatter(xs, v, s=9, color=col, alpha=.5, edgecolor="none", label=name if i == 0 else None)
            ax[0].plot([i + (k - .5) * 0.34 - .1, i + (k - .5) * 0.34 + .1], [v.mean()] * 2, color="k", lw=1.5)
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"stage {s[1]}" for s in S])
    ax[0].set_ylabel("contribution to logit"); ax[0].set_title("Per-stage contribution (bar = mean)", fontsize=9); ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar(range(5), [sep[s] for s in S], color=["#999999"] * 3 + ["#56B4E9", "#0072B2"], width=.6)
    ax[1].axhline(0, color="k", lw=.6); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f"stage {s[1]}" for s in S])
    ax[1].set_ylabel("mean logit difference, positive - negative"); ax[1].set_title("Who carries the separation (grey = invisible to a stage-3 Grad-CAM)", fontsize=9)
    for q in ax:
        q.spines[["top", "right"]].set_visible(False); q.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(out / "stage_decomp.png"); plt.close(fig)
    print(f"\nwrote {out}/stage_contrib.csv, stage_decomp.txt, stage_decomp.png")


if __name__ == "__main__":
    main()
