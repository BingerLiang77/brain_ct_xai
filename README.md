# Brain CT Haemorrhage Classification & Explainability (XAI)

This repository contains the complete research codebase, local preprocessing and training pipelines, Colab execution workflows, and explainability evaluation results for an intracranial haemorrhage classifier evaluated on head CT scans.

## Project Overview

Frozen CT-FM head-CT haemorrhage classifier on BHSD (174 positive / 958 negative after QC; train 902 / test 230 = 35 positive / 195 negative). Model = frozen CT-FM SegResEncoder, head-masked average pooling of all 5 encoder stages (992 features), standardisation, linear head. The repo covers the explainability analyses: Grad-CAM (stages 3 and 4) and Integrated Gradients with four baselines (zero, constant 40 HU, blur, negative mean), localisation, deletion/insertion faithfulness, per-stage logit decomposition, and a mean-vs-max block-ranking comparison.

## Repository Layout

```text
.
├── .gitignore               # Strict exclusion of weights, large data, caches, and tensors
├── LICENSE                  # Repository licence notice (Unlicensed)
├── README.md                # Project overview, layout, and quick-start
├── requirements.txt         # Pinned local training environment dependencies
├── notebooks/
│   └── colab_cells.txt      # Execution trace and cell commands for Google Colab
├── src/
│   ├── colab/               # 11 frozen standalone Colab research scripts
│   │   ├── segcls_model.py          # Model architecture (frozen CT-FM encoder + head / decoder)
│   │   ├── segcls_data.py           # Dataset loader for preprocessed .pt cases
│   │   ├── segcls_train.py          # Training workflow (cross-validation folds)
│   │   ├── segcls_eval.py           # Evaluation across pooling strategies
│   │   ├── segcls_diag.py           # Diagnostic evaluation script
│   │   ├── segcls_xai.py            # Core attribution generation
│   │   ├── maskpool_xai.py          # Head-masked feature extraction & attribution
│   │   ├── mpool_loc.py             # Localisation evaluation (Pointing game, AUC, concentration)
│   │   ├── mpool_faith.py           # Faithfulness curves (deletion/insertion under max/mean rank)
│   │   ├── mpool_stage_decomp.py    # Per-stage logit decomposition (stages 0–4)
│   │   └── mpool_rank_compare.py    # Mean-vs-max block-ranking comparison
│   └── project/             # Local data preparation, preprocessing, and training pipeline
│       ├── config.py                # Central project configuration and default paths
│       ├── brain_mask_classical.py  # Classical morphological skull-stripping & brain masking
│       ├── build_cohort.py          # Initial raw BHSD cohort scanning & thickness matching
│       ├── build_annotated_cohort.py# Cohort restricted to verified label_192 positives
│       ├── build_masked_cohort.py   # Mask QC filtering (produces final 1,132 cohort)
│       ├── dataset.py               # Deterministic volume resampling, SPL orientation, windowing
│       ├── build_cache.py           # Multiprocess batch preprocessed tensor caching
│       ├── head_maskpool_refit.py   # Masked-pooling feature extraction, 5-fold CV & head training
│       ├── build_cases.py           # Self-contained segcls_pack case packaging
│       ├── verify_pack.py           # Inference verification against reference predictions
│       ├── xai_common.py            # Shared dataset and model helper functions
│       ├── xai_faith.py             # Perturbation and faithfulness helpers
│       ├── head_xai_agreement.py    # Spatial agreement utilities
│       └── head_noise_diag.py       # Noise perturbation diagnostics
├── docs/
│   ├── methods.md           # Formal experimental methodology and XAI formulation
│   ├── caveats.md           # Hardware requirements, data format, assumptions, and caveats
│   └── results.md           # Quantitative evaluation results and summaries
└── results/
    ├── localization/        # Pointing game, AUC-ROC, and lesion concentration results
    ├── faithfulness/        # Deletion/insertion AUC and logit curve summaries
    ├── stage_decomposition/ # Per-stage logit contribution metrics
    ├── rank_compare/        # Block overlap and negative attribution cancellation
    ├── ig_completeness/     # Axiom verification (sum of attributions vs logit shift)
    └── segcls/              # Training logs, configurations, and evaluation metrics
```

## Pre-processing and Training Pipeline

The local data preparation and model training workflow under `src/project/` runs in the following sequential order:

```text
1. brain_mask_classical.py
   ├── Inputs : Raw head CT NIfTI scans (--in-dir), optional GT masks (--lesion-dir)
   └── Outputs: Intracranial masks (<id>_brainmask.nii.gz), brainmask_qc.csv, brainmask_qc_flagged.csv

2. build_cohort.py
   ├── Inputs : POS_TRAIN_DIR (label_192), POS_TEST_DIR (FirstStage_test), NEG_POOL_DIR (nobleed)
   └── Outputs: data/cohort.csv (1,152 scans, 1:5 thickness-matched negative sampling)

3. build_annotated_cohort.py
   ├── Inputs : data/cohort.csv
   └── Outputs: data/cohort_annotated.csv (splits only verified label_192 positives 80/20 train/test)

4. build_masked_cohort.py
   ├── Inputs : data/cohort_annotated.csv, positives & negatives brainmask_qc.csv
   └── Outputs: data/cohort_masked.csv (1,132 scans: excludes 18 pos with lesion coverage < 50%,
                2 neg with volume < 500 mL or errors; attaches mask_path; 0 patient leakage)

5. dataset.py / build_cache.py
   ├── Inputs : data/cohort_masked.csv, raw CT scans, brain masks
   └── Outputs: data/cache_roi_pt/<hash>/<id>.pt (in-plane bounding box margin 10mm, outside brain
                set to -1000 HU, resampled to 2.5mm isotropic, SPL orientation, brain window
                [-20, 100] HU -> [0, 1], padded/cropped to (96, 128, 128))

6. head_maskpool_refit.py
   ├── Inputs : data/cohort_masked.csv, cached tensors, frozen CT-FM encoder
   └── Outputs: checkpoints/ich_head_masked_pool_stages_0_1_2_3_4.pt (linear head with masked
                average pooling across stages 0–4, StandardScaler, balanced LogisticRegression),
                results/test_predictions_maskpool.csv, results/maskpool/features.npz

7. build_cases.py
   ├── Inputs : cohort.csv, cached preprocessed tensors, raw CTs, brain masks, lesion masks, QC CSVs
   └── Outputs: segcls_pack/data/cases/<id>.pt (packaged case tensors: x, head, brain, lesion, meta)

8. verify_pack.py
   ├── Inputs : segcls_pack/weights/, segcls_pack/cohort.csv, segcls_pack/data/cases/<id>.pt
   └── Outputs: Verifies test forward passes match reference predictions (max |diff| < 0.005)
```

## Quick-Start: Reproducing in Google Colab

The scripts in `src/colab/` are frozen research artefacts designed to run in a standard Google Colab GPU environment (analyses ran on a free Colab T4).

Refer to [`notebooks/colab_cells.txt`](notebooks/colab_cells.txt) for the exact end-to-end execution cells. Note that Colab provides PyTorch pre-installed; do **not** run `pip install -r requirements.txt` on Colab (which would overwrite Colab's default torch build). Instead, install only the supplementary packages via:
```bash
pip -q install monai lighter_zoo safetensors scikit-learn pandas
```

For technical details on input volume preprocessing and parameters, see [`docs/methods.md`](docs/methods.md).

## Summary of Findings

- Localisation (35 positives) [results/localization/loc_summary.txt]: all methods are close to chance. Top-N Dice 0.012-0.046 against a chance level of 0.019; exact-peak pointing 0/35 (GC3) to 5/35 (IG_neg) against about 0.7 expected. IG_bl and IG_neg are modestly above chance (energy ratio 1.79 [1.40, 2.22] and 1.55 [1.18, 2.02]); the constant baselines are at or below chance (IG_u 0.57). P-values are unadjusted.
- Faithfulness [results/faithfulness/faith_summary.txt]: results depend strongly on the replacement volume, which is not neutral (random curves are humped). With mean-ranked blocks, IG_bl and IG_neg beat random under most replacements, IG_zero rarely does; cells where the replacement equals the IG baseline are partly favourable by construction; max ranking is inconsistent.
- Stage decomposition [results/stage_decomposition/stage_decomp.txt]: stages 0-2 carry 36.2 % of the positive-vs-negative logit separation. A stage-3 Grad-CAM sees stages 3+4 (R^2 0.88 across cases); a stage-4 Grad-CAM sees stage 4 only (R^2 0.59).
- Ranking comparison [results/rank_compare/rank_compare.txt]: top-10 % blocks of the mean and max rankings overlap 0.15-0.33 (chance 0.08); Spearman 0.09-0.17. Max-ranked blocks are sign-balanced (net/gross 0.01-0.03, like an average block) but contain about 1.8-2.7 times more strong voxels.

## Codebase Status & Reproducibility Notice

All Python scripts in `src/colab/` are maintained in their frozen, original execution form to guarantee full auditability and descriptive reproducibility of the experiments. For operational assumptions, GPU memory requirements, and external weight paths, consult [`docs/caveats.md`](docs/caveats.md).

## Licence

All rights reserved. No open-source licence has been granted at this time. See [`LICENSE`](LICENSE) for details.
