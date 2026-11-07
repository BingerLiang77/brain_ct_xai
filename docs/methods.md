# Experimental Methodology & Explainability Formulation

This document details the dataset characteristics, model architecture, and explainability (XAI) methodology implemented in this repository.

---

## 1. Dataset & Cohort Split

The project evaluates intracranial haemorrhage (ICH) classification on non-contrast head CT scans from the Brain Haemorrhage Segmentation Dataset (BHSD).

- **Cohort after Quality Control (QC)**: 1,132 total scans
  - **174 positive cases** (confirmed haemorrhage)
  - **958 negative cases** (no haemorrhage)
- **Data Partitioning**:
  - **Training Cohort**: 902 scans (split across cross-validation folds)
  - **Held-out Test Cohort**: 230 scans (35 positive, 195 negative)

### Preprocessing & Input Representation
Each head CT scan is resampled and preprocessed into a fixed 3D tensor:
- **Dimensions**: $(B, 1, 96, 128, 128)$
- **Windowing**: Brain window (typically 0–80 HU or 40 HU center / 80 HU width), normalized to $[0, 1]$.
- **Anatomical Mask**: A binary brain/head mask is computed to isolate intracranial voxels and exclude skull/background air from feature pooling.
- **Reference Ground Truth**: 3D binary lesion segmentation masks (`lesion`) annotated by radiologists for all positive test cases.

---

## 2. Model Architecture

The classifier leverages a frozen pre-trained 3D foundation model backbone with an anatomically-guided classification head:

```text
3D Head CT (1x96x128x128)
       │
       ▼
┌──────────────────────────────────────────────┐
│  Frozen CT-FM SegResEncoder (5 Stages)       │
│  Stage 0: 32 ch,  (48, 64, 64)               │
│  Stage 1: 64 ch,  (24, 32, 32)               │
│  Stage 2: 128 ch, (12, 16, 16)               │
│  Stage 3: 256 ch, (6, 8, 8)                  │
│  Stage 4: 512 ch, (3, 4, 4)                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Head-Masked Average Pooling                 │
│  - Downsampled head mask per stage           │
│  - Average features within head mask         │
│  - Concatenate all 5 stages -> 992 features  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Standardisation + Linear Head               │
│  - Differentiable scalar decision logit      │
└──────────────────────────────────────────────┘
```

- **Backbone**: Frozen CT-FM SegResEncoder (`lighter_zoo` or MONAI `SegResEncoder`), with 5 progressive downsampling stages totaling 992 feature channels:
  - Stage 0: 32 channels
  - Stage 1: 64 channels
  - Stage 2: 128 channels
  - Stage 3: 256 channels
  - Stage 4: 512 channels
- **Feature Pooling**: Head-masked average pooling across all 5 encoder stages. Downsampled binary head masks ensure that only intracranial feature representations contribute to the feature vector, eliminating confounding from cranial bones, extracranial soft tissue, or background noise.
- **Classification Head**: Feature standardisation followed by a linear classification head outputting a differentiable scalar logit.

---

## 3. Explainability (XAI) Suite

The repository implements a comprehensive explainability benchmark evaluating attribution localisation, faithfulness, structural decomposition, and baseline inductive biases.

### 3.1 Attribution Methods & Baselines

1. **Grad-CAM (Stages 3 & 4)** (`GC3`, `GC4`):
   - Computes gradient-weighted activation maps at the deepest encoder feature representations (Stage 3: 256 channels, Stage 4: 512 channels) followed by trilinear upsampling back to input space $(96, 128, 128)$.
2. **Integrated Gradients (IG)**:
   - Computes path integrals of input gradients along the linear path between a neutral baseline $x'$ and the input volume $x$:
     $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$
   - Approximated via a 16-step Riemann summation.
   - Evaluated across four distinct baselines:
     - **Zero Baseline (`IG_zero`)**: $x' = 0$ (attenuation baseline).
     - **Constant 40 HU Baseline (`IG_u`)**: $x' = \text{const}_{40\text{HU}}$ (healthy brain parenchyma density).
     - **Blur Baseline (`IG_bl`)**: Gaussian blurred version of the input scan ($x' = \text{GaussianFilter}(x, \sigma=4)$).
     - **Negative-Mean Baseline (`IG_neg`)**: Empirical voxel-wise mean volume computed across all verified negative training scans.

---

### 3.2 Evaluation Metrics

#### A. Localisation Metrics (`src/colab/mpool_loc.py`)
Evaluated across the 35 positive test cases against expert radiologist lesion ground-truth:
- **Pointing Game**: Evaluates whether the maximum attribution peak falls within the true lesion boundary:
  - Exact hitting ($r=0$).
  - Tolerance radii: $r=2$ voxels (5 mm tolerance), $r=4$ voxels (10 mm tolerance).
- **Attribution AUC-ROC**: Treating positive attributions inside the head mask as a binary classifier for lesion voxels.
- **Lesion Concentration**: Ratio of mean attribution density inside the true lesion mask relative to the mean attribution density across the whole brain parenchyma.

#### B. Faithfulness Evaluation (`src/colab/mpool_faith.py`)
Measures whether features identified as important genuinely drive the model's prediction:
- **Deletion Paradigm**: Progressively replacing the top-$k\%$ attributed blocks ($k \in [0, 100]$ in 5% steps) with neutral replacement values; lower Area Under Curve (AUC) indicates greater faithfulness (faster degradation of positive logit).
- **Insertion Paradigm**: Progressively restoring top-$k\%$ attributed blocks starting from a replaced baseline; higher AUC indicates greater faithfulness.
- **Block-level Ranking Aggregations**:
  - **Max-ranking**: Blocks ranked by maximum attribution voxel within each block.
  - **Mean-ranking**: Blocks ranked by mean attribution across voxels in each block.

#### C. Per-Stage Logit Decomposition (`src/colab/mpool_stage_decomp.py`)
Because the linear head linearly combines pooled features from all 5 encoder stages:
$$\text{logit}(x) = b + \sum_{s=0}^{4} c_s(x)$$
where $c_s(x) = w_s^T z_s(x)$ represents the exact contribution of encoder stage $s$. This allows quantifying how much decision mass is captured by deep semantic stages (3 & 4) versus shallow textural stages (0, 1 & 2).

#### D. Mean-vs-Max Block-Ranking Comparison (`src/colab/mpool_rank_compare.py`)
Investigates attribution cancellation artefacts:
- Compares overlap and Jaccard similarity of selected top-$k\%$ blocks between max-ranking and mean-ranking.
- Measures the presence of strongly negative voxels alongside positive peaks inside max-ranked blocks, which cancel out under spatial pooling.

#### E. IG Completeness Axiom Verification (`results/ig_completeness/`)
Validates that the sum of Integrated Gradients attributions across all voxels strictly equals the difference between the model's output logit and the baseline logit:
$$\sum_i \text{IG}_i(x) \approx F(x) - F(x')$$
