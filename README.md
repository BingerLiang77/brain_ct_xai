# Brain CT Haemorrhage Classification & Explainability (XAI)

This repository contains the research codebase, Colab execution workflows, and explainability evaluation results for an intracranial haemorrhage classifier evaluated on head CT scans.

## Project Overview

Frozen CT-FM head-CT haemorrhage classifier on BHSD (174 positive / 958 negative after QC; train 902 / test 230 = 35 positive / 195 negative). Model = frozen CT-FM SegResEncoder, head-masked average pooling of all 5 encoder stages (992 features), standardisation, linear head. The repo covers the explainability analyses: Grad-CAM (stages 3 and 4) and Integrated Gradients with four baselines (zero, constant 40 HU, blur, negative mean), localisation, deletion/insertion faithfulness, per-stage logit decomposition, and a mean-vs-max block-ranking comparison.

## Repository Layout

```text
.
├── .gitignore               # Strict exclusion of weights, large data, caches, and tensors
├── LICENSE                  # Repository licence notice
├── README.md                # Project overview, layout, and quick-start
├── requirements.txt         # Core dependencies for training, inference, and XAI
├── notebooks/
│   └── colab_cells.txt      # Execution trace and cell commands for Google Colab
├── src/
│   └── colab/               # 11 frozen standalone Colab research scripts
│       ├── segcls_model.py          # Model architecture (frozen CT-FM encoder + head / decoder)
│       ├── segcls_data.py           # Dataset loader for preprocessed .pt cases
│       ├── segcls_train.py          # Training workflow (cross-validation folds)
│       ├── segcls_eval.py           # Evaluation across pooling strategies
│       ├── segcls_diag.py           # Diagnostic evaluation script
│       ├── segcls_xai.py            # Core attribution generation
│       ├── maskpool_xai.py          # Head-masked feature extraction & attribution
│       ├── mpool_loc.py             # Localisation evaluation (Pointing game, AUC, concentration)
│       ├── mpool_faith.py           # Faithfulness curves (deletion/insertion under max/mean rank)
│       ├── mpool_stage_decomp.py    # Per-stage logit decomposition (stages 0–4)
│       └── mpool_rank_compare.py    # Mean-vs-max block-ranking comparison
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

## Quick-Start: Reproducing in Google Colab

The scripts in `src/colab/` are frozen research artefacts designed to run in a GPU-accelerated Google Colab environment (e.g. NVIDIA A100 or V100 GPU).

1. **Environment Setup**:
   Mount Google Drive and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execution Steps**:
   Refer to [`notebooks/colab_cells.txt`](notebooks/colab_cells.txt) for the exact end-to-end execution cells, including:
   - Extracting `segcls_pack.zip` to `/content/segcls_pack`
   - Training or evaluating the model checkpoint (`segcls_train.py`, `segcls_eval.py`)
   - Running attribution generation with Grad-CAM and Integrated Gradients (`maskpool_xai.py`)
   - Running localisation benchmarks (`mpool_loc.py`)
   - Generating faithfulness deletion/insertion curves (`mpool_faith.py`)
   - Running stage logit decomposition (`mpool_stage_decomp.py`) and block-rank comparisons (`mpool_rank_compare.py`)

For technical details on input volume preprocessing and parameters, see [`docs/methods.md`](docs/methods.md).

## Summary of Findings

- **Attribution Localisation**: The negative-mean baseline (`IG_neg`) achieves the highest pointing game accuracy and lesion concentration among all IG baselines and Grad-CAM variants ($p < 0.001$).
- **Faithfulness**: Removing high-attribution blocks under max-ranking significantly degrades predictive logit, confirming explanation faithfulness, but also reveals baseline-dependent replacement artefacts.
- **Stage Contribution**: Deep encoder stages (stage 3 and stage 4) dominate case-to-case logit variation (>80%), explaining why Grad-CAM stage 3/4 captures primary discriminative signals.
- Complete metric tables and charts are compiled in [`docs/results.md`](docs/results.md).

## Codebase Status & Reproducibility Notice

All Python scripts in `src/colab/` are maintained in their frozen, original execution form to guarantee full auditability and descriptive reproducibility of the published experiments. For operational assumptions, GPU memory requirements, and external weight paths, consult [`docs/caveats.md`](docs/caveats.md).

## Licence

All rights reserved. No open-source licence has been granted at this time. See [`LICENSE`](LICENSE) for details.
