"""Does the classifier read global noise / smoothness / contrast?  Do the XAI maps just follow contrast?

Put next to head_xai_agreement.py in head_ct_ich_ctfm/src and run
    python head_noise_diag.py              # all three sections
    python head_noise_diag.py feat block   # or a subset: feat perturb block

feat    Per-case image statistics of the MODEL INPUT inside an eroded brain mask (so mask edges do not count):
          noise_std  = std of (x - gaussian(x, 1 voxel))      high-frequency noise
          grad_mean  = mean |grad x|                          sharpness
          mean_in, sd_in, frac_zero (<= -20 HU), frac_one (>= 100 HU)
        AUC of each statistic for the label (whole cohort, and the test set), Spearman with the model logit
        (test set, within each class), and a patient-grouped CV logistic regression:  image stats, metadata,  both.
perturb Whole-image perturbations of the test cases and the change in logit by label:
          blurS_halo    gaussian blur as in xai_faith.baseline_volumes (the halo spills outside the brain, where the
                        model was always shown exactly 0)
          blurS_masked  blur restricted to the brain (normalised convolution), outside stays 0
          noiseSD       Gaussian noise inside the brain only
        If 'halo' raises the logit a lot and 'masked' does not, the +8 seen for 'blur2' in head_curves.py comes from the
        halo, not from smoothing.  If both raise it, the model likes smooth images.
block   For the cached attribution cases: per case, Spearman between each method's block score and block features
        (mean intensity, |x-0.5| = contrast against the uniform baseline, sd, gradient, distance to the head edge,
        z index).  Tests whether a map follows contrast / edges / position.

Outputs go to results/diag_head/ .
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import ndimage as ndi
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

import config as C
import xai_common as xc
import xai_faith as xf
import head_xai_agreement as ha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "diag_head"
MAX_CASES = None
ERODE = 3
BLURS = (1.0, 2.0)
NOISES = (0.02, 0.05)
METHODS = ["GC4", "GC3", "IG_mn", "IG_u"]


def auc_safe(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else np.nan


# ----------------------------------------------------------------------------- feat
def case_features(x, head):
    xa, h = x[0].numpy(), head.numpy()
    er = ndi.binary_erosion(h, iterations=ERODE)
    if er.sum() < 1000:
        er = h
    res = xa - ndi.gaussian_filter(xa, 1.0)
    g = np.sqrt(sum(np.square(np.gradient(xa, axis=a)) for a in range(3)))
    v = xa[er]
    return dict(noise_std=float(res[er].std()), grad_mean=float(g[er].mean()), mean_in=float(v.mean()),
                sd_in=float(v.std()), frac_zero=float((v <= 0.001).mean()), frac_one=float((v >= 0.999).mean()),
                head_vox=int(h.sum()))


def section_feat():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    coh = pd.read_csv(C.DATA_DIR / "cohort_masked.csv")
    if MAX_CASES:
        coh = coh.head(MAX_CASES)
    rows, miss = [], 0
    for n, (_, r) in enumerate(coh.iterrows()):
        try:
            x = xc.load_tensor(r["id"])
        except Exception:
            miss += 1
            continue
        rows.append(dict(id=r["id"], **case_features(x, ha.head_of(x))))
        if (n + 1) % 200 == 0:
            print(f"  features {n + 1}/{len(coh)}", flush=True)
    f = pd.DataFrame(rows).merge(coh, on="id")
    f["z_extent_mm"] = f["n_slices"] * f["z_spacing"]
    f.to_csv(OUT / "image_features.csv", index=False)
    feats = ["noise_std", "grad_mean", "mean_in", "sd_in", "frac_zero", "frac_one", "head_vox"]
    meta = [c for c in ["n_slices", "z_spacing", "xy_spacing", "z_extent_mm"] if c in f]
    print(f"\n=== FEAT: image statistics of the model input ({len(f)} cases, {miss} without cached tensor) ===")
    print(f.groupby("label")[feats].mean().T.round(4).to_string())
    te = f[f["split"] == "test"].drop(columns=["score"], errors="ignore").merge(xc.load_cases()[["id", "score"]], on="id")
    print(f"\nAUC of each statistic for the label (0.5 = none):  whole cohort (n={len(f)}),  test set (n={len(te)})")
    for c in feats:
        print(f"   {c:10s}  cohort {auc_safe(f['label'], f[c]):.3f}   test {auc_safe(te['label'], te[c]):.3f}")
    print("\nSpearman(statistic, model logit) on the test set:   all    label=0   label=1")
    for c in feats:
        print(f"   {c:10s}  {spearmanr(te[c], te['score'])[0]:+.2f}   {spearmanr(te[te.label == 0][c], te[te.label == 0]['score'])[0]:+.2f}"
              f"   {spearmanr(te[te.label == 1][c], te[te.label == 1]['score'])[0]:+.2f}")
    print("\nPatient-grouped 5-fold logistic regression for the label, whole cohort:")
    g = f["patient"].values
    for name, cols in (("metadata only", meta), ("image statistics only", feats), ("both", meta + feats)):
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                              f[cols].fillna(0).values, f["label"].values, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
        print(f"   {name:22s} AUC = {auc_safe(f['label'], p):.3f}")
    print("   (image model on the test set: 0.80; metadata only from head_cheap_diag: 0.64)")


# ----------------------------------------------------------------------------- perturb
def variants(x, head):
    xa, h = x[0].numpy(), head.numpy().astype(np.float32)
    out = {}
    for s in BLURS:
        out[f"blur{s:g}_halo"] = torch.from_numpy(ndi.gaussian_filter(xa, s)).float()[None]
        num, den = ndi.gaussian_filter(xa * h, s), ndi.gaussian_filter(h, s)
        out[f"blur{s:g}_masked"] = torch.from_numpy(np.where(h > 0, num / np.maximum(den, 1e-6), 0.0)).float()[None]
    rng = np.random.default_rng(0)
    for sd in NOISES:
        out[f"noise{sd:g}"] = torch.from_numpy(np.where(h > 0, np.clip(xa + rng.normal(0, sd, xa.shape), 0, 1), 0.0)).float()[None]
    return out


def section_perturb():
    device = xc.pick_device()
    model = xc.load_model(device)
    cases = xc.load_cases()
    if MAX_CASES:
        cases = cases.head(MAX_CASES)
    rows = []
    for n, (_, r) in enumerate(cases.iterrows()):
        x = xc.load_tensor(r["id"])
        vs = variants(x, ha.head_of(x))
        l = xf._logits(model, [x] + list(vs.values()), device).tolist()
        rows.append(dict(id=r["id"], label=int(r["label"]), logit=l[0], **{k: v - l[0] for k, v in zip(vs, l[1:])}))
        if (n + 1) % 40 == 0:
            print(f"  perturb {n + 1}/{len(cases)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "perturbation_dlogit.csv", index=False)
    cols = [c for c in d if c not in ("id", "label", "logit")]
    print("\n=== PERTURB: change in logit after whole-image perturbation (test set) ===")
    for lab in (0, 1):
        g = d[d.label == lab]
        t = pd.DataFrame({"mean": g[cols].mean(), "median": g[cols].median(), "sd": g[cols].std(), "frac_up": (g[cols] > 0).mean()})
        print(f"\n-- label={lab} (n={len(g)})")
        print(t.round(2).to_string())
    print("\nReading: compare *_halo with *_masked.  halo >> masked  -> the jump of 'blur2' in head_curves.py is the halo outside "
          "the brain.  Both up -> the model rates smoother images as more hemorrhage-like.  noise up/down shows the same for noise.")


# ----------------------------------------------------------------------------- block
def block_features(c):
    xa, head, bid, nb = c["x"][0], c["head"], c["bid"], c["nb"]
    cnt, hf = c["cnt"].clamp(min=1), c["head"].float()

    def bmean(v):
        return torch.bincount(bid.flatten(), weights=(v * hf).flatten().float(), minlength=nb) / cnt
    mean = bmean(xa)
    grad = sum(torch.gradient(xa, dim=a)[0] ** 2 for a in range(3)).sqrt()
    dist = torch.from_numpy(ndi.distance_transform_edt(head.numpy())).float()
    zc, _ = ha.block_centres(bid, nb, head.shape)
    return {"mean_x": mean, "contrast_vs_0.5": bmean((xa - 0.5).abs()), "sd_x": (bmean(xa ** 2) - mean ** 2).clamp(min=0).sqrt(),
            "gradient": bmean(grad), "dist_to_edge": bmean(dist), "z_index": zc}


def section_block():
    cdf = xc.load_cases().set_index("id")
    ids = [i for i in cdf.index if (Path(xc.XAI_DIR) / "attributions" / f"{i}.pt").exists()]
    rows = []
    for cid in ids:
        c = ha.load_case(cid)
        if c is None:
            continue
        bf = block_features(c)
        for m in METHODS:
            for fn, fv in bf.items():
                a, b = c["sc"][m][c["elig"]].numpy(), fv[c["elig"]].numpy()
                if a.std() > 0 and b.std() > 0:
                    rows.append(dict(id=cid, label=int(cdf.loc[cid, "label"]), method=m, feature=fn, rho=float(spearmanr(a, b)[0])))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "block_score_vs_features.csv", index=False)
    print(f"\n=== BLOCK: per-case Spearman between block score and block feature, mean over cases ({d['id'].nunique()} cases) ===")
    for lab, name in ((1, "positives"), (None, "all cases")):
        g = d if lab is None else d[d.label == lab]
        t = g.pivot_table(index="method", columns="feature", values="rho", aggfunc="mean").round(2)
        print(f"\n-- {name}: mean rho")
        print(t.to_string())
        if lab == 1:
            p = g.groupby(["method", "feature"])["rho"].apply(lambda s: wilcoxon(s)[1] if len(s) >= 6 and (s != 0).any() else np.nan)
            print("\n-- p-values of rho != 0 (Wilcoxon) among positives")
            print(p.unstack().round(4).to_string())
    print("\nReading: a large |rho| with 'contrast_vs_0.5' means the map mostly marks blocks that differ from a flat 40 HU brain; "
          "with 'dist_to_edge' or 'z_index' it follows position (skull base, head rim).")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = set(sys.argv[1:]) or {"feat", "perturb", "block"}
    if "feat" in todo:
        section_feat()
    if "perturb" in todo:
        section_perturb()
    if "block" in todo:
        section_block()
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
