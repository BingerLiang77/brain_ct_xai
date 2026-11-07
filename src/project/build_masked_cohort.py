import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

def main():
    cohort_ann = pd.read_csv(C.DATA_DIR / "cohort_annotated.csv")
    pos_qc = pd.read_csv(C.DATA_DIR / "brain_mask" / "positives_192" / "masks" / "brainmask_qc.csv")
    neg_qc = pd.read_csv(C.DATA_DIR / "brain_mask" / "negatives_960" / "masks" / "brainmask_qc.csv")

    # Exclusions
    # Positives with coverage < 0.50
    exclude_pos = set(pos_qc[pos_qc["lesion_in_mask"] < 0.50]["case"])
    # Negatives with mask_ml < 500 mL or status != ok
    exclude_neg = set(neg_qc[(neg_qc["mask_ml"] < 500) | (neg_qc["status"] != "ok")]["case"])

    print(f"Excluding {len(exclude_pos)} positive cases (< 50% coverage)")
    print(f"Excluding {len(exclude_neg)} negative cases (< 500 mL volume or error)")

    # Filter cohort
    kept = cohort_ann[~cohort_ann["id"].isin(exclude_pos | exclude_neg)].copy()
    
    # Add mask_path column
    pos_mask_dir = C.DATA_DIR / "brain_mask" / "positives_192" / "masks"
    neg_mask_dir = C.DATA_DIR / "brain_mask" / "negatives_960" / "masks"

    mask_paths = []
    for _, r in kept.iterrows():
        cid = r["id"]
        if r["label"] == 1:
            mp = pos_mask_dir / f"{cid}_brainmask.nii.gz"
        else:
            mp = neg_mask_dir / f"{cid}_brainmask.nii.gz"
        assert mp.exists(), f"Mask not found: {mp}"
        mask_paths.append(str(mp))
        
    kept["mask_path"] = mask_paths

    # Verify counts
    n_pos = (kept["label"] == 1).sum()
    n_neg = (kept["label"] == 0).sum()
    print(f"\nFinal Cohort: Total={len(kept)} | Positives={n_pos} | Negatives={n_neg}")
    assert n_pos == 174, f"Expected 174 positives, got {n_pos}"
    assert n_neg == 958, f"Expected 958 negatives, got {n_neg}"

    # Verify patient independence
    tr_pats = set(kept[kept["split"] == "train"]["patient"])
    te_pats = set(kept[kept["split"] == "test"]["patient"])
    overlap = tr_pats & te_pats
    print(f"Patient overlap between train and test: {len(overlap)}")
    assert len(overlap) == 0, f"Patient overlap found: {overlap}"

    print("\nSplit x Label distribution:")
    print(pd.crosstab(kept["split"], kept["label"], margins=True))

    out_file = C.DATA_DIR / "cohort_masked.csv"
    kept.to_csv(out_file, index=False)
    print(f"\nSaved clean cohort to {out_file}")

if __name__ == "__main__":
    main()
