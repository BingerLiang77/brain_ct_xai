# Technical Caveats & Operational Assumptions

This document lists important environmental constraints, operational assumptions, and methodological caveats associated with the code and experimental results in this repository.

---

## 1. Execution Environment & Hardware

- **Execution Environment**:
  The Colab explainability analyses ran on a standard, free Google Colab instance with an NVIDIA T4 GPU.
- **Standalone Script Structure**:
  The codebase in `src/colab/` and `src/project/` consists of standalone research scripts rather than an installable Python package (`setup.py` / `pyproject.toml`). Scripts are designed to be executed via command-line flags.
- **Colab Drive Mounting Assumptions**:
  The execution workflows recorded in `notebooks/colab_cells.txt` assume a standard Google Colab environment with Google Drive mounted at `/content/drive`. Example CLI invocations in docstrings (e.g. `segcls_train.py:4`, `segcls_eval.py:2`) reference `/content/segcls_pack` and `/content/drive/MyDrive/segcls_run1`. When running locally, users must adjust `--pack`, `--ckpt`, and `--out` CLI arguments accordingly.
- **Colab Dependencies**:
  Google Colab provides PyTorch pre-installed. Users should not run `pip install -r requirements.txt` on Colab (which could overwrite Colab's default PyTorch build). Only supplementary libraries should be installed (`pip -q install monai lighter_zoo safetensors scikit-learn pandas`).

---

## 2. External Weights & Case Format

- **Foundation Model Weights**:
  In accordance with research repository best practices, large model checkpoints and pre-trained CT-FM foundation model weights (`ct_fm_feature_extractor`) are not tracked in git.
  - Users must supply the foundation model weights under `weights/ct_fm_feature_extractor`.
  - The script `segcls_model.py` provides primary loading via `lighter_zoo.SegResEncoder.from_pretrained()`, with a secondary fallback to `monai.networks.nets.segresnet_ds.SegResEncoder` using `safetensors`.
- **Preprocessed Case Format (`cases/<id>.pt`)**:
  Each `.pt` file is a dictionary containing:
  - `x`: `torch.float16`, $(1, 96, 128, 128)$. Brain window $[-20, 100]\text{ HU}$ mapped to $[0, 1]$; outside brain mask $= 0.0$.
  - `head`: `torch.uint8`, $(96, 128, 128)$. Binary mask for block eligibility ($x > 0.01$).
  - `brain`: `torch.uint8`, $(96, 128, 128)$. Skull-stripped brain mask from NIfTI `mask_path`.
  - `lesion`: `torch.uint8`, $(96, 128, 128)$. Binary ground truth haemorrhage mask.
  - `meta`: Metadata dictionary containing `id`, `patient`, `label` ($1$ for positive, $0$ for negative), `split`, `fold` ($0..4$ for training, $-1$ for test), `lesion_vox`, `lesion_ml`, and `lesion_in_mask`.

---

## 3. Cohort Derivation & Selection Biases (1,152 -> 1,132)

- **Initial Cohort (`build_cohort.py`)**: 1,152 scans (192 positive scans from `label_192` + 960 thickness-matched negatives from `unlabel_2000/nobleed`).
- **Quality Control Filtering (`build_masked_cohort.py`)**:
  - **18 positive cases removed**: Dropped due to lesion coverage inside the intracranial brain mask $< 50\%$ (`lesion_in_mask < 0.50`), leaving 174 verified positive scans.
  - **2 negative cases removed**: Dropped due to intracranial brain mask volume $< 500\text{ mL}$ or segmentation errors (`status != 'ok'`), leaving 958 verified negative scans.
- **Selection Bias**:
  Removing the 18 positive cases with lesions mostly outside the classical brain mask introduces an upstream anatomical selection bias toward intracerebral/intraventricular haemorrhages and under-represents skull-adjacent extra-axial lesions such as subdural (SDH) and epidural haematomas (EDH).
- **Final Cohort**: **1,132 scans** (174 positives / 958 negatives).
  - **Training Split**: 902 scans (139 positive / 763 negative across 901 unique patients).
  - **Held-Out Test Split**: 230 scans (35 positive / 195 negative).
  - Strict patient isolation: 0 patient overlap between training and testing sets.

---

## 4. Checkpoint Metadata Keys vs. Recomputed Metrics

- In the final saved model checkpoint `checkpoints/ich_head_masked_pool_stages_0_1_2_3_4.pt`, the stored dictionary keys for performance are:
  - `test_auc: 0.800586`
  - `train_cv_auc: 0.834410`
- **Important Note on Stale Metadata**:
  These two specific scalar metadata fields are **stale** because `head_maskpool_refit.py:218` cloned the predecessor GAP checkpoint dictionary (`old_ck`) and updated `mean`, `std`, `w`, `b`, `threshold = -0.782635`, and `pooling = 'masked'`, without updating the string metric tags `test_auc` and `train_cv_auc`.
- **True Recomputed Metrics (Masked-Pooling Head)**:
  - **Test AUC**: **0.849231** (~0.849)
  - **Test AUPRC**: **0.575633** (~0.576)
  - **Training Out-Of-Fold CV AUC**: **0.885939** (~0.886, saved in `ref/train_oof_maskpool.csv`)
  - Verified across all 230 test scans via `verify_pack.py` (max logit diff $< 4.66 \times 10^{-3}$, attributable solely to float16 tensor precision).

---

## 5. Cross-Validation Fold Reproducibility

- The 5 cross-validation folds on the 902 training scans were re-derived using deterministic scikit-learn `GroupKFold(n_splits=5)` grouped strictly by patient ID (`head_maskpool_refit.py:114`):
  ```python
  GroupKFold(5).split(idx_tr, y[idx_tr], groups[idx_tr])
  ```
- **Fold Verification**:
  Regenerating the folds produces **exact 100% equivalence (0 mismatches across all 1,132 cases)** with `segcls_pack/cohort.csv` and `ref/train_oof_maskpool.csv`.
- Fold distribution across the 902 training cases:
  - Fold 0: 181 cases (34 pos / 147 neg)
  - Fold 1: 181 cases (29 pos / 152 neg)
  - Fold 2: 180 cases (24 pos / 156 neg)
  - Fold 3: 180 cases (26 pos / 154 neg)
  - Fold 4: 180 cases (26 pos / 154 neg)

---

## 6. Predecessor GAP Model Pipeline (Earlier Iteration)

- An earlier iteration of this project extracted unmasked Global Average Pooling (GAP) features across all 5 encoder stages using `extract_features.py` and trained a linear classification head using `train_eval.py`.
- **Performance**:
  - Achieved Test AUC of **0.8006** (CV AUC 0.8344).
  - Superseded by head-masked average pooling in `head_maskpool_refit.py` (Test AUC **0.8492**).
- `extract_features.py` and `train_eval.py` are deliberately excluded from `src/project/` as they represent superseded architecture.

---

## 7. Methodological Caveats in XAI Evaluation

- **Small Sample Size (35 Positives) & Unadjusted p-values**:
  Localisation and faithfulness evaluations are conducted on the 35 positive test cases. All reported p-values are unadjusted paired Wilcoxon signed-rank tests against population priors or random baselines.
- **Grad-CAM Stage Blindness**:
  Grad-CAM computed at stage 3 and stage 4 does not see encoder stages 0, 1, and 2. As demonstrated in `results/stage_decomposition/stage_decomp.txt`, stages 0–2 carry 36.2% of the positive-vs-negative logit separation. Thus, Grad-CAM inherently captures only the deeper semantic portion of the model's decision.
- **Replacement Non-Neutrality & Favourability by Construction**:
  As shown in `results/faithfulness/faith_summary.txt`, baseline replacement volumes (zero, constant 40 HU, Gaussian blur, negative mean) are not neutral:
  - Replacing all eligible blocks shifts the mean logit to $+1.77$ (zero), $-1.56$ (const), $+2.15$ (blur), and $-1.38$ (negmean), compared to $+1.05$ for the original scan.
  - Evaluation cells where the perturbation replacement volume matches the Integrated Gradients baseline (e.g. `IG_zero` under zero replacement, `IG_bl` under blur replacement) are partly favourable by construction.
- **Integrated Gradients Completeness Approximation Errors**:
  Across the 35 test cases (`results/localization/loc_summary.txt`), the median relative completeness integration error $| \sum \text{IG} - (F(x) - F(x')) | / | F(x) - F(x') |$ is:
  - `IG_bl`: ~0.7% (median error 0.007)
  - `IG_neg`: ~0.7% (median error 0.007)
  - `IG_zero`: 1.4%–1.8% (with power path warp $\alpha = t^4$ and 128 steps)
  - `IG_u`: 7.9%–11.0% (median error 0.079)
- **Top-N Dice Dependence on Lesion Size**:
  Because $N$ is set to the exact number of lesion voxels, Top-N Dice is mathematically constrained by lesion volume and spatial dispersion across small punctate haemorrhages.
