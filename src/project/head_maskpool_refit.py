"""Refit the linear head with MASKED global-average pooling (pool only over head voxels) on the same frozen CT-FM encoder.

Put next to head_xai_agreement.py / head_noise_diag.py in head_ct_ich_ctfm/src and run   python head_maskpool_refit.py

Why: the head occupies ~5 % of the (96,128,128) volume.  The current model averages every stage over ALL voxels, so ~95 %
of each pooled feature is background and any non-zero value outside the brain moves the logit by several units
(head_noise_diag.py: blur with halo +5..+8, blur inside the mask only +1).  Here each stage is averaged with weights =
fraction of head voxels in that feature cell (head = xc.head_mask(x), the project's own definition).

What it does
 1. Extracts, for every cohort case, the 992-d pooled features twice (plain GAP, masked pooling); cached in
    results/maskpool/features.npz  (resumable).
 2. Sanity check: plain-GAP features + the SAVED checkpoint must reproduce the saved test scores.
 3. Fits the same recipe for both poolings: standardise on train, L2 logistic regression, class_weight='balanced',
    C chosen by patient-grouped 5-fold CV on the TRAIN split only (AUC).  Reports test AUC, AP, sens@spec90, and the
    paired bootstrap difference (masked - GAP).  NOTE: the recipe is my assumption; if your original training used a
    different head recipe, compare the 'GAP refit' row, not the old number, to the 'masked' row.
 4. Saves the masked model as checkpoints/ich_head_masked_pool_stages_0_1_2_3_4.pt (same keys as the old checkpoint) and
    results/test_predictions_maskpool.csv (same columns as test_predictions_*.csv).
 5. Robustness on the test set for old vs new model: change in logit under whole-image blur (halo / inside the brain),
    noise, and the AUC of the logit of the 'uniform brain' input (mask shape only).

To use the new model elsewhere:   from head_maskpool_refit import load_masked_model;  model = load_masked_model(device)
The masked head is derived from the input itself (x > 0.01), the same rule as xc.head_mask.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import config as C
import xai_common as xc
import xai_faith as xf
import head_xai_agreement as ha
import head_noise_diag as hn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "maskpool"
NEW_CKPT = Path(xc.CKPT_PATH).parent / "ich_head_masked_pool_stages_0_1_2_3_4.pt"
PRED_CSV = ROOT / "results" / "test_predictions_maskpool.csv"
CS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
MAX_CASES = None
N_BOOT = 2000


class ICHMasked(xc.ICHModel):
    """Same frozen encoder and linear head; every stage is pooled over head voxels only."""

    def forward(self, x, return_feats=False):
        fs = self.backbone(x)
        head = xc.head_mask(x).to(x.dtype).reshape(x.shape[0], 1, *x.shape[2:])
        zs = []
        for s in self.stages:
            f = fs[s]
            w = F.adaptive_avg_pool3d(head, f.shape[2:])
            zs.append((f * w).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3, 4)).clamp(min=1e-6))
        logit = ((torch.cat(zs, 1) - self.mean) / self.std) @ self.w + self.b
        return (logit, fs) if return_feats else logit


def load_masked_model(device=None):
    return ICHMasked(ckpt_path=NEW_CKPT).to(device or xc.pick_device())


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def both_poolings(model, x):
    fs = model.backbone(x)
    head = xc.head_mask(x).to(x.dtype).reshape(x.shape[0], 1, *x.shape[2:])
    zg, zm = [], []
    for s in model.stages:
        f = fs[s]
        w = F.adaptive_avg_pool3d(head, f.shape[2:])
        zg.append(f.mean(dim=(2, 3, 4)))
        zm.append((f * w).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3, 4)).clamp(min=1e-6))
    return torch.cat(zg, 1)[0].cpu().numpy(), torch.cat(zm, 1)[0].cpu().numpy()


def get_features(model, coh, device):
    path = OUT / "features.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if list(d["ids"]) == list(coh["id"]):
            print(f"features loaded from {path}")
            return d["gap"], d["masked"]
    G, M = [], []
    for n, cid in enumerate(coh["id"]):
        g, m = both_poolings(model, xc.load_tensor(cid)[None].to(device))
        G.append(g); M.append(m)
        if (n + 1) % 100 == 0:
            print(f"  features {n + 1}/{len(coh)}", flush=True)
    G, M = np.stack(G), np.stack(M)
    np.savez(path, ids=np.array(coh["id"]), gap=G, masked=M)
    return G, M


# ----------------------------------------------------------------------------- fitting / evaluation
def fit_head(Z, y, groups, tr):
    sc = StandardScaler().fit(Z[tr])
    Zs = sc.transform(Z)
    idx = np.where(tr)[0]
    best = None
    for c in CS:
        oof = np.zeros(len(idx))
        for a, b in GroupKFold(5).split(idx, y[idx], groups[idx]):
            m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(Zs[idx[a]], y[idx[a]])
            oof[b] = m.decision_function(Zs[idx[b]])
        auc = roc_auc_score(y[idx], oof)
        if best is None or auc > best[0]:
            best = (auc, c, oof)
    m = LogisticRegression(C=best[1], max_iter=5000, class_weight="balanced").fit(Zs[idx], y[idx])
    fpr, tpr, thr = roc_curve(y[idx], best[2])
    return sc, m, dict(cv_auc=best[0], C=best[1], threshold=float(thr[np.argmax(tpr - fpr)]))


def sens_at_spec90(y, s):
    t = np.quantile(s[y == 0], 0.90)
    return float((s[y == 1] > t).mean())


def evaluate(name, y, s, thr):
    pred = s >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    return dict(model=name, auc=roc_auc_score(y, s), ap=average_precision_score(y, s), sens_spec90=sens_at_spec90(y, s),
                sens_at_thr=tp / max(1, tp + fn), spec_at_thr=tn / max(1, tn + fp), TP=tp, FN=fn, TN=tn, FP=fp)


def boot_diff(y, s_new, s_old, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(roc_auc_score(y[i], s_new[i]) - roc_auc_score(y[i], s_old[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ----------------------------------------------------------------------------- robustness old vs new
def robustness(models, device):
    cases = xc.load_cases()
    if MAX_CASES:
        cases = cases.head(MAX_CASES)
    rows = []
    for n, (_, r) in enumerate(cases.iterrows()):
        x = xc.load_tensor(r["id"])
        head = ha.head_of(x)
        vs = hn.variants(x, head)
        vs["const40"] = torch.where(head[None], torch.tensor(0.5), torch.tensor(0.0))
        for name, mdl in models.items():
            l = xf._logits(mdl, [x] + list(vs.values()), device).tolist()
            row = dict(id=r["id"], label=int(r["label"]), model=name, logit=l[0], const40=l[-1])
            row.update({k: v - l[0] for k, v in zip(list(vs)[:-1], l[1:-1])})
            rows.append(row)
        if (n + 1) % 40 == 0:
            print(f"  robustness {n + 1}/{len(cases)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "robustness.csv", index=False)
    cols = [c for c in d if c not in ("id", "label", "model", "logit", "const40")]
    print("\n=== ROBUSTNESS (test set): mean change in logit under whole-image perturbation ===")
    for lab in (0, 1):
        t = d[d.label == lab].groupby("model")[cols].mean().T.round(2)
        print(f"\n-- label={lab} (n={int((d[(d.model == list(models)[0]) & (d.label == lab)]).shape[0])})")
        print(t.to_string())
    print("\n'uniform brain' input (0.5 inside the head, 0 outside) = mask shape / size only:")
    for name in models:
        g = d[d.model == name]
        print(f"   {name:8s} AUC of its logit for the label = {roc_auc_score(g['label'], g['const40']):.3f}   "
              f"Spearman with the model's own logit = {spearmanr(g['const40'], g['logit'])[0]:+.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = xc.pick_device()
    old_model = xc.load_model(device)
    old_ck = torch.load(xc.CKPT_PATH, weights_only=True)
    coh = pd.read_csv(C.DATA_DIR / "cohort_masked.csv")
    if MAX_CASES:
        coh = coh.head(MAX_CASES)
    G, M = get_features(old_model, coh, device)
    y, groups = coh["label"].values.astype(int), coh["patient"].values
    tr, te = (coh["split"] == "train").values, (coh["split"] == "test").values
    print(f"train {tr.sum()} (pos {int(y[tr].sum())}),  test {te.sum()} (pos {int(y[te].sum())})")

    # 2. sanity check of the extraction
    s_old = (((G - old_ck["mean"].numpy()) / old_ck["std"].numpy()) @ old_ck["w"].numpy() + float(old_ck["b"]))
    saved = xc.load_cases()[["id", "score"]].merge(coh[["id"]].assign(i=np.arange(len(coh))), on="id")
    print(f"check: GAP features + saved checkpoint vs saved test scores: max |diff| = "
          f"{np.abs(s_old[saved['i'].values] - saved['score'].values).max():.2e}  (should be ~1e-5)")

    # 3. refit both poolings with the same recipe
    res, preds = [], {}
    fitted = {}
    for name, Z in (("GAP refit", G), ("masked", M)):
        sc, m, info = fit_head(Z, y, groups, tr)
        s = m.decision_function(sc.transform(Z))
        fitted[name] = (sc, m, info)
        preds[name] = s
        print(f"{name}: chosen C = {info['C']}, train CV AUC = {info['cv_auc']:.3f}, Youden threshold = {info['threshold']:.3f}")
        res.append(evaluate(name, y[te], s[te], info["threshold"]))
    res.insert(0, evaluate("old checkpoint", y[te], s_old[te], float(old_ck["threshold"])))
    print("\n=== TEST SET ===")
    print(pd.DataFrame(res).round(3).to_string(index=False))
    d, lo, hi = boot_diff(y[te], preds["masked"][te], preds["GAP refit"][te])
    print(f"\nAUC(masked) - AUC(GAP refit) = {d:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   (n_pos = {int(y[te].sum())})")

    # 4. save the masked model
    sc, m, info = fitted["masked"]
    ck = {k: v for k, v in old_ck.items()}
    ck.update(mean=torch.tensor(sc.mean_, dtype=torch.float32), std=torch.tensor(sc.scale_, dtype=torch.float32),
              w=torch.tensor(m.coef_[0], dtype=torch.float32), b=float(m.intercept_[0]), threshold=info["threshold"],
              pooling="masked")
    NEW_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, NEW_CKPT)
    out = coh[te][["id", "patient", "label"] + (["z_spacing"] if "z_spacing" in coh else [])].copy()
    out["score"] = preds["masked"][te]
    out["pred"] = (out["score"] >= info["threshold"]).astype(int)
    out.to_csv(PRED_CSV, index=False)
    print(f"saved {NEW_CKPT}\nsaved {PRED_CSV}")

    # 5. robustness
    new_model = load_masked_model(device)
    with torch.no_grad():
        chk = [float(new_model(xc.load_tensor(i)[None].to(device))) for i in out["id"][:5]]
    print(f"check: ICHMasked forward vs refit scores (first 5 test cases): max |diff| = "
          f"{np.abs(np.array(chk) - out['score'].values[:5]).max():.2e}")
    robustness({"old": old_model, "masked": new_model}, device)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
