# Quantitative Experimental Results

This document synthesizes the empirical results obtained across classification evaluation, attribution localisation, deletion/insertion faithfulness, per-stage logit decomposition, and block-ranking comparisons.

---

## 1. Classification Performance

Evaluated on the held-out test cohort of 230 head CT scans (35 positive ICH cases, 195 negative cases) from the BHSD dataset:

| Model Architecture / Pooling Head | Val AUC | Test AUC | Test AUPRC |
| :--- | :---: | :---: | :---: |
| **CT-FM + Head-Masked Average Pooling (Frozen Head)** | **—** | **0.849** | **0.576** |
| Segmentation Decoder Diagnostic Head (LSE-2 pooling) | 0.811 | 0.790 | 0.406 |
| Segmentation Decoder Diagnostic Head (Top-200 pooling) | 0.814 | 0.784 | 0.398 |
| Segmentation Decoder Diagnostic Head (Soft-Volume pooling) | 0.796 | 0.802 | 0.496 |

*Note*: The primary classification model evaluated throughout the explainability benchmarks is the frozen CT-FM encoder with head-masked average pooling across all 5 stages.

---

## 2. Attribution Localisation Benchmark

Evaluated on the 35 positive test cases against radiologist-annotated lesion masks (`results/localization/loc_summary.txt`). All p-values are unadjusted paired Wilcoxon tests against population priors:

### 2.1 Summary Metrics

- **Overall Trend**: All methods are close to chance.
- **Top-N Dice** ($N = \text{lesion voxels}$): Ranges from **0.012 to 0.046** against an analytic chance level of **0.019**:
  - `GC3`: 0.012 [0.002, 0.025]
  - `GC4`: 0.020 [0.004, 0.043]
  - `IG_zero`: 0.013 [0.003, 0.025]
  - `IG_u`: 0.014 [0.002, 0.029]
  - `IG_bl`: 0.033 [0.019, 0.049]
  - `IG_neg`: 0.046 [0.020, 0.076]
- **Exact-Peak Pointing ($r=0$)**:
  - Ranges from **0/35** (`GC3`, 0.000) to **5/35** (`IG_neg`, 0.143) against about **0.7 cases expected by chance** ($0.019 \times 35 \approx 0.67$).
  - Tolerances: at $r=2$ (5 mm), `IG_bl` and `IG_neg` achieve 6/35 (0.171); at $r=4$ (10 mm), `IG_neg` achieves 7/35 (0.200).
- **Lesion Energy Ratio** (1.0 = chance):
  - `IG_bl` and `IG_neg` are modestly above chance: **1.79 [1.40, 2.22]** and **1.55 [1.18, 2.02]**.
  - Constant baseline `IG_u` is at or below chance: **0.57 [0.42, 0.76]**.
  - `GC3`: 1.03 [0.74, 1.38]; `GC4`: 0.90 [0.70, 1.10]; `IG_zero`: 0.92 [0.75, 1.09].

---

## 3. Faithfulness: Deletion & Insertion Curves

Evaluated across a 21-point perturbation grid ($k = 0\%$ to $100\%$ in 5% increments) under max-ranking and mean-ranking block selections (`results/faithfulness/faith_summary.txt`):

- **Replacement Volume Non-Neutrality**:
  Results depend strongly on the replacement volume, which is not neutral (random curves are humped):
  - Replacing all eligible blocks gives a mean logit of **+1.77** (zero), **-1.56** (const), **+2.15** (blur), **-1.38** (negmean), against **+1.05** for the original scan [`results/faithfulness/faith_summary.txt`].
- **Mean Ranking vs. Max Ranking**:
  - With **mean-ranked blocks**, `IG_bl` and `IG_neg` beat random under most replacements; `IG_zero` rarely does.
  - Cells where the replacement volume equals the Integrated Gradients baseline are partly favourable by construction.
  - **Max ranking** is inconsistent across replacements due to attribution cancellation artefacts.

---

## 4. Per-Stage Logit Decomposition

Linear decomposition of the model decision logit $\text{logit} = b + \sum_{s=0}^{4} c_s$ across the 5 encoder stages (`results/stage_decomposition/stage_decomp.txt`):

- **Separation Share**:
  - **Stages 0–2 carry 36.2%** of the positive-vs-negative logit separation (Stage 0: 3.6%, Stage 1: 11.2%, Stage 2: 21.3%).
  - **Stages 3–4 carry 63.8%** of the logit separation (Stage 3: 37.7%, Stage 4: 26.2%).
- **What Grad-CAM Sees**:
  - A **stage-3 Grad-CAM** sees stages 3+4 ($R^2 = 0.88$ across cases; $r = 0.938$).
  - A **stage-4 Grad-CAM** sees stage 4 only ($R^2 = 0.59$ across cases; $r = 0.768$).
  - Because stages 0–2 carry 36.2% of the decision separation, Grad-CAM at stage 3/4 does not capture the full decision mechanism.

---

## 5. Mean-vs-Max Block-Ranking Comparison

Analysis of spatial block selection across top-$k\%$ blocks (`results/rank_compare/rank_compare.txt`):

- **Ranking Divergence**:
  - Top-10% blocks of the mean and max rankings overlap **0.15–0.33** (chance level = **0.08**).
  - Spearman rank correlation between mean-scored and max-scored blocks across cases is low: **0.09–0.17**.
- **Sign Cancellation**:
  - Max-ranked blocks are sign-balanced (net/gross attribution ratio: **0.01–0.03**, comparable to an average block).
  - However, max-ranked blocks contain about **1.8–2.7 times more strong voxels** (both strongly positive and strongly negative voxels that cancel each other out under spatial pooling).

---

## 6. IG Completeness Axiom Verification

Axiom verification across all 35 test cases (`results/ig_completeness/ig_completeness.csv` and `results/localization/loc_summary.txt`):
- Relative completeness integration error $| \sum \text{IG} - (F(x) - F(x')) | / | F(x) - F(x') |$:
  - `IG_bl`: ~0.7% (median error 0.007)
  - `IG_neg`: ~1% (median error 0.007)
  - `IG_zero`: 1.4%–1.8% (median error 0.014 with power path warp $\alpha = t^4$ and 128 steps)
  - `IG_u`: 7.9%–11.0% (median error 0.079)
