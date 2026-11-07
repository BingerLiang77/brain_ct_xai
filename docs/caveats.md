# Technical Caveats & Operational Assumptions

This document lists important environmental constraints, operational assumptions, and methodological caveats associated with the code and experimental results in this repository.

---

## 1. Execution Environment & Dependencies

- **Standalone Script Structure**:
  The codebase in `src/colab/` consists of standalone research scripts rather than a modularized, packaged Python library (`setup.py` / `pyproject.toml`). Scripts are designed to be executed linearly via command-line flags.
- **Colab Drive Mounting Assumptions**:
  The execution workflows recorded in `notebooks/colab_cells.txt` assume a standard Google Colab environment with Google Drive mounted at `/content/drive`. Example CLI invocations in docstrings (e.g. `segcls_train.py:4`, `segcls_eval.py:2`) reference `/content/segcls_pack` and `/content/drive/MyDrive/segcls_run1`. When running locally or on an on-premise cluster, users must adjust `--pack`, `--ckpt`, and `--out` CLI arguments accordingly.
- **Hardware & VRAM Constraints**:
  - Processing full 3D head CT volumes ($96 \times 128 \times 128$) requires a CUDA-capable GPU with at least 16 GB VRAM (e.g., NVIDIA A100, V100, or RTX 3090/4090).
  - Computing Integrated Gradients involves a 16-step Riemann summation. Memory management scripts utilize `torch.cuda.empty_cache()` and mixed precision (`torch.cuda.amp.autocast`) to prevent Out-Of-Memory (OOM) errors during backpropagation.

---

## 2. External Weights & Data Prerequisites

- **Excluded Foundation Model Weights**:
  In accordance with research repository best practices, large model checkpoints (`best_auc.pt`, `best_dice.pt`, `last.pt`) and pre-trained CT-FM foundation model weights (`ct_fm_feature_extractor`) are not tracked in git.
  - Users must supply the foundation model weights under `weights/ct_fm_feature_extractor`.
  - The script `segcls_model.py` provides primary loading via `lighter_zoo.SegResEncoder.from_pretrained()`, with a secondary fallback to `monai.networks.nets.segresnet_ds.SegResEncoder` using `safetensors`.
- **Preprocessed Case Volumes**:
  The loader `segcls_data.py` expects individual case files stored as `.pt` dictionaries with keys:
  - `x`: $(1, 96, 128, 128)$ float tensor normalized to $[0, 1]$.
  - `head`: $(96, 128, 128)$ binary mask tensor indicating intracranial tissue.
  - `label`: Integer scalar (0 for negative, 1 for positive).
  - `lesion`: $(96, 128, 128)$ binary mask of radiologist-annotated haemorrhage (for positive cases).

---

## 3. Methodological Caveats in XAI Evaluation

- **Replacement Neutrality in Faithfulness Benchmarks**:
  As highlighted in `results/faithfulness/faith_summary.txt`, baseline replacement volumes (zero, constant 40 HU, Gaussian blur, negative mean) are **not neutral** to the neural network:
  - Completely replacing head voxels with zero produces an extreme out-of-distribution shift that sharply alters baseline logits ($+1.77$ vs $+1.05$ normal).
  - Random replacement curves do not remain flat across deletion steps. Consequently, faithfulness must always be interpreted relative to the paired random baseline curves rather than in isolation.
- **Attribution Cancellation under Spatial Pooling**:
  As revealed in `results/rank_compare/rank_compare_summary.txt`, max-ranking of spatial blocks preferentially selects blocks containing sharp positive peaks positioned directly adjacent to strong negative attribution voxels. Under subsequent spatial average pooling, these positive and negative components cancel each other out, artificially altering faithfulness dynamics.
- **Numerical Reproducibility Across GPUs**:
  Due to non-deterministic CUDA atomic operations in 3D trilinear interpolation (such as upsampling in Grad-CAM) and floating-point summation across large 3D tensors, exact voxel-level values may exhibit minor numerical tolerances ($< 10^{-5}$) across differing CUDA driver and GPU architectures.
