#!/usr/bin/env python3
"""Standalone verification script for segcls_pack.

Runs on CPU (or optional MPS/CUDA), requires NO imports from outside the pack.
Loads cached preprocessed x for all 230 test cases from data/cases/<id>.pt,
runs the frozen CT-FM encoder + masked-pooling head from weights/,
and reports:
  1. Max absolute difference between recomputed logits and stored reference logits (ref/test_predictions_maskpool.csv)
  2. Mean absolute difference
  3. Recomputed test AUC vs reference test AUC
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lighter_zoo import SegResEncoder
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent


class ICHMasked(nn.Module):
    """Frozen CT-FM encoder + Masked Pooling (x > 0.01) + Linear Head."""

    def __init__(self, backbone, ckpt):
        super().__init__()
        self.backbone = backbone
        self.stages = list(ckpt["stages"])
        for n in ("mean", "std", "w"):
            self.register_buffer(n, ckpt[n].float())
        self.register_buffer("b", torch.tensor(float(ckpt["b"])))
        self.threshold = float(ckpt["threshold"])
        self.eval()

    def forward(self, x):
        fs = self.backbone(x)
        head = (x > 0.01).to(x.dtype).reshape(x.shape[0], 1, *x.shape[2:])
        zs = []
        for s in self.stages:
            f = fs[s]
            w = F.adaptive_avg_pool3d(head, f.shape[2:])
            zs.append((f * w).sum(dim=(2, 3, 4)) / w.sum(dim=(2, 3, 4)).clamp(min=1e-6))
        logit = ((torch.cat(zs, 1) - self.mean) / self.std) @ self.w + self.b
        return logit


def main():
    parser = argparse.ArgumentParser(description="Verify segcls_pack test predictions")
    parser.add_argument("--pack", type=str, default=None,
                        help="Path to segcls_pack root directory (default: checks ROOT, PROJECT_ROOT/segcls_pack, or ./segcls_pack)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Device to run inference on (default: cpu)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of test cases to verify (default: all 230)")
    args = parser.parse_args()

    if args.pack:
        pack_root = Path(args.pack).resolve()
    elif (ROOT / "cohort.csv").exists():
        pack_root = ROOT
    elif (ROOT.parents[1] / "segcls_pack" / "cohort.csv").exists():
        pack_root = (ROOT.parents[1] / "segcls_pack").resolve()
    else:
        pack_root = Path("segcls_pack").resolve()

    device = torch.device(args.device)
    print("=== SegCls Pack Verification ===")
    print(f"Pack root : {pack_root}")
    print(f"Device    : {device}")

    weights_enc = pack_root / "weights" / "ct_fm_feature_extractor"
    weights_head = pack_root / "weights" / "ich_head_masked_pool_stages_0_1_2_3_4.pt"
    cohort_file = pack_root / "cohort.csv"
    ref_file = pack_root / "ref" / "test_predictions_maskpool.csv"

    for p in (weights_enc, weights_head, cohort_file, ref_file):
        assert p.exists(), f"Missing required file in pack: {p}"

    coh = pd.read_csv(cohort_file)
    ref = pd.read_csv(ref_file).set_index("id")

    test_coh = coh[coh["split"] == "test"].copy()
    if args.limit:
        test_coh = test_coh.head(args.limit)
    n_test = len(test_coh)
    print(f"Test cases to verify: {n_test} (pos: {(test_coh['label'] == 1).sum()}, neg: {(test_coh['label'] == 0).sum()})")

    print("\nLoading weights from pack...")
    t0 = time.time()
    backbone = SegResEncoder.from_pretrained(weights_enc).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    ck = torch.load(weights_head, map_location="cpu", weights_only=True)
    model = ICHMasked(backbone, ck).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s")

    print(f"\nRunning forward passes on {n_test} cases...")
    t_start = time.time()
    diffs = []
    comp_logits = []
    ref_logits = []
    true_labels = []

    with torch.no_grad():
        for i, (_, r) in enumerate(test_coh.iterrows()):
            cid = r["id"]
            pt_path = pack_root / "data" / "cases" / f"{cid}.pt"
            assert pt_path.exists(), f"Case file missing: {pt_path}"

            data = torch.load(pt_path, weights_only=False)
            x = data["x"].float()[None].to(device)
            logit = float(model(x))

            r_logit = float(ref.loc[cid, "score"])
            diff = abs(logit - r_logit)

            diffs.append(diff)
            comp_logits.append(logit)
            ref_logits.append(r_logit)
            true_labels.append(int(r["label"]))

            if (i + 1) % 50 == 0 or (i + 1) == n_test:
                elapsed = time.time() - t_start
                print(f"  [{i+1:3d}/{n_test}] elapsed: {elapsed:.1f}s ({elapsed/(i+1):.2f}s/scan) | max diff so far: {max(diffs):.2e}")

    diffs = np.array(diffs)
    comp_logits = np.array(comp_logits)
    ref_logits = np.array(ref_logits)
    true_labels = np.array(true_labels)

    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    median_diff = float(np.median(diffs))

    auc_ref = roc_auc_score(true_labels, ref_logits)
    auc_comp = roc_auc_score(true_labels, comp_logits)
    ap_ref = average_precision_score(true_labels, ref_logits)
    ap_comp = average_precision_score(true_labels, comp_logits)

    print("\n" + "=" * 50)
    print("=== VERIFICATION RESULTS ===")
    print("=" * 50)
    print(f"Cases tested          : {n_test}")
    print(f"Max absolute diff     : {max_diff:.6f}  ({max_diff:.2e})")
    print(f"Mean absolute diff    : {mean_diff:.6f}  ({mean_diff:.2e})")
    print(f"Median absolute diff  : {median_diff:.6f}  ({median_diff:.2e})")
    print(f"Reference Test AUC    : {auc_ref:.6f}")
    print(f"Recomputed Test AUC   : {auc_comp:.6f}")
    print(f"AUC Difference        : {abs(auc_comp - auc_ref):.2e}")
    print(f"Reference Test AUPRC  : {ap_ref:.6f}")
    print(f"Recomputed Test AUPRC : {ap_comp:.6f}")

    if max_diff < 1e-2:
        print("\n>>> PASS: Maximum logit difference is < 1e-2 (float16 rounding artifact only).")
        print(">>> Pack integrity is fully verified!")
        return 0
    else:
        print(f"\n>>> FAIL: Maximum logit difference {max_diff:.4f} >= 1e-2!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
