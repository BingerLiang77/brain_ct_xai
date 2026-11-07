"""Build the annotated cohort table: data/cohort_annotated.csv

Train and test positives both come from label_192 (192 volumes, 100% verified ground truth).
Split: patient-level 80% train / 20% test (stratified by thickness: thin < 4mm vs thick >= 4mm).
Negatives: 1:5 ratio drawn from the nobleed pool already extracted in cohort.csv, matched on thickness.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def main():
    rng = np.random.default_rng(C.SEED)
    cohort_orig = pd.read_csv(C.DATA_DIR / "cohort.csv")

    # 1. Positives from label_192 (in original cohort they were marked split=='train')
    pos = cohort_orig[(cohort_orig["label"] == 1) & (cohort_orig["split"] == "train")].copy()
    assert len(pos) == 192

    # Patient-level 80/20 split
    # Handle patient with 2 scans by collapsing to patient level for split
    pat_df = pos.drop_duplicates("patient").copy()
    pat_df["thick_bin"] = pat_df["thick"]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=C.SEED)
    tr_pat_idx, te_pat_idx = next(splitter.split(pat_df, pat_df["thick_bin"]))
    tr_patients = set(pat_df.iloc[tr_pat_idx]["patient"])
    te_patients = set(pat_df.iloc[te_pat_idx]["patient"])

    pos["split"] = np.where(pos["patient"].isin(tr_patients), "train", "test")
    pos_tr = pos[pos["split"] == "train"]
    pos_te = pos[pos["split"] == "test"]

    print(f"Annotated positives (label_192): train={len(pos_tr)} (patients={len(tr_patients)}), "
          f"test={len(pos_te)} (patients={len(te_patients)})")
    print("Positives thickness mix:")
    print(pd.crosstab(pos["thick"], pos["split"]))

    # 2. Negatives: select from the nobleed pool already in cohort.csv
    # Original cohort has 1210 negatives (960+250) from nobleed
    neg_pool = cohort_orig[cohort_orig["label"] == 0].copy()
    # Ensure no patient overlap with positives
    pos_pats = set(pos["patient"])
    neg_pool = neg_pool[~neg_pool["patient"].isin(pos_pats)].drop_duplicates("patient")

    n_tr_neg = C.NEG_PER_POS * len(pos_tr)
    n_te_neg = C.NEG_PER_POS * len(pos_te)

    # Thickness matching helper
    def draw_matched_negs(pool, target_pos, n_wanted, rng):
        chosen = []
        for b, n_p in target_pos["thick"].value_counts().items():
            wanted_b = int(round(n_wanted * (n_p / len(target_pos))))
            cand = pool[pool["thick"] == b]
            n_take = min(wanted_b, len(cand))
            chosen.append(cand.sample(n_take, random_state=int(rng.integers(1 << 31))))
        out = pd.concat(chosen)
        if len(out) < n_wanted:
            rest = pool[~pool["id"].isin(out["id"])]
            out = pd.concat([out, rest.sample(n_wanted - len(out), random_state=int(rng.integers(1 << 31)))])
        return out

    neg_tr = draw_matched_negs(neg_pool, pos_tr, n_tr_neg, rng).assign(split="train")
    neg_pool_rem = neg_pool[~neg_pool["id"].isin(neg_tr["id"])]
    neg_te = draw_matched_negs(neg_pool_rem, pos_te, n_te_neg, rng).assign(split="test")

    cohort_ann = pd.concat([pos_tr, neg_tr, pos_te, neg_te], ignore_index=True)
    assert cohort_ann["id"].is_unique
    assert not (set(cohort_ann[cohort_ann.split == "train"]["patient"]) &
                set(cohort_ann[cohort_ann.split == "test"]["patient"]))

    out_file = C.DATA_DIR / "cohort_annotated.csv"
    cohort_ann.to_csv(out_file, index=False)
    print(f"\nwrote {out_file} ({len(cohort_ann)} scans)")
    print("\nSplit x Label:")
    print(pd.crosstab(cohort_ann["split"], cohort_ann["label"]))
    print("\nThickness x (Split, Label):")
    print(pd.crosstab(cohort_ann["thick"], [cohort_ann["split"], cohort_ann["label"]]))


if __name__ == "__main__":
    main()
