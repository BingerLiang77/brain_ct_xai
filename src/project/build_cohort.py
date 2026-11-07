"""Build the cohort table: data/cohort.csv

  train: 192 positives (label_192)      + 5 x 192 = 960 negatives (nobleed)
  test : 50 positives (FirstStage_test) + 5 x  50 = 250 negatives (nobleed)

Design choices (all printed so they can be checked):
  * Split unit = patient (the first ID in ID_<patient>_ID_<series>); one scan per negative patient, and
    train/test negatives never share a patient.
  * Negatives are drawn so that their slice-thickness mix (thin < 4 mm vs thick) matches the positives of
    the same split. Otherwise "thin slices -> negative" is a shortcut the classifier can learn.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def scan_dir(d, label, split):
    rows = []
    for p in sorted(Path(d).glob("*.nii.gz")):
        img = nib.load(str(p))
        # voxel size from the affine, not the header: nibabel silently replaces pixdim=0 with 1 in
        # header.get_zooms(), which hides scans whose affine is degenerate (preprocessing would crash/NaN)
        vs = nib.affines.voxel_sizes(img.affine)
        rows.append(dict(id=p.name[:-7], patient=p.name.split("_")[1], path=str(p), label=label, split=split,
                         n_slices=int(img.shape[2]), z_spacing=float(vs[2]), xy_spacing=float(vs[0])))
    return pd.DataFrame(rows)


def add_bin(df):
    df["thick"] = np.where(df["z_spacing"] < 4.0, "thin(<4mm)", "thick(>=4mm)")
    return df


def draw_negatives(pool, pos, rng):
    """Draw NEG_PER_POS * len(pos) negatives whose thickness mix matches `pos` (fall back to random if short)."""
    n_total = C.NEG_PER_POS * len(pos)
    chosen = []
    for b, n_pos in pos["thick"].value_counts().items():
        cand = pool[pool["thick"] == b]
        n = min(C.NEG_PER_POS * n_pos, len(cand))
        if n < C.NEG_PER_POS * n_pos:
            print(f"  WARNING: only {len(cand)} '{b}' negatives available, wanted {C.NEG_PER_POS * n_pos}")
        chosen.append(cand.sample(n, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(chosen)
    if len(out) < n_total:  # top up from whatever is left
        rest = pool[~pool["id"].isin(out["id"])]
        out = pd.concat([out, rest.sample(n_total - len(out), random_state=int(rng.integers(1 << 31)))])
    return out


def main():
    rng = np.random.default_rng(C.SEED)
    pos_tr = add_bin(scan_dir(C.POS_TRAIN_DIR, 1, "train"))
    pos_te = add_bin(scan_dir(C.POS_TEST_DIR, 1, "test"))
    neg = add_bin(scan_dir(C.NEG_POOL_DIR, 0, "pool"))
    print(f"positives: train {len(pos_tr)}, test {len(pos_te)} | negative pool {len(neg)} scans")

    # voxel spacing outside this range means a broken header (e.g. z = 0.03 mm or 90 mm): resampling would be wrong
    plausible = lambda d: d["z_spacing"].between(1.0, 10.0) & d["xy_spacing"].between(0.3, 1.2)
    for name, d in (("train positives", pos_tr), ("test positives", pos_te)):
        if (~plausible(d)).any():
            print(f"  WARNING: {(~plausible(d)).sum()} {name} have odd spacing (kept):",
                  d[~plausible(d)][["id", "xy_spacing", "z_spacing"]].values.tolist())
    n0 = len(neg)
    neg = neg[plausible(neg)]
    print(f"negatives dropped for implausible spacing: {n0 - len(neg)}")

    # one scan per negative patient, and no patient shared with any positive
    pos_patients = set(pos_tr["patient"]) | set(pos_te["patient"])
    neg = neg[~neg["patient"].isin(pos_patients)]
    neg = neg.sample(frac=1, random_state=C.SEED).drop_duplicates("patient")
    print(f"negative pool after patient de-duplication: {len(neg)}")

    neg_tr = draw_negatives(neg, pos_tr, rng).assign(split="train")
    neg = neg[~neg["id"].isin(neg_tr["id"])]
    neg_te = draw_negatives(neg, pos_te, rng).assign(split="test")

    cohort = pd.concat([pos_tr, neg_tr, pos_te, neg_te], ignore_index=True)
    assert cohort["id"].is_unique
    assert not (set(cohort[cohort.split == "train"]["patient"]) & set(cohort[cohort.split == "test"]["patient"]))

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = C.DATA_DIR / "cohort.csv"
    cohort.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(cohort)} scans)")
    print("\nsplit x label:\n", pd.crosstab(cohort["split"], cohort["label"]))
    print("\nthickness x (split, label):\n", pd.crosstab(cohort["thick"], [cohort["split"], cohort["label"]]))
    print("\nz-spacing by (split, label):\n", cohort.groupby(["split", "label"])["z_spacing"].describe().round(2))


if __name__ == "__main__":
    main()
