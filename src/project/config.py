"""Central configuration. Override the raw-data location without editing code:
    export BHSD_ROOT="/path/to/BHSD"
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache_roi_pt"      # preprocessed tensors
FEATURE_DIR = DATA_DIR / "features"        # frozen CT-FM features
RESULT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BHSD_ROOT = Path(os.environ.get(
    "BHSD_ROOT",
    "/path/to/BHSD",
))
POS_TRAIN_DIR = BHSD_ROOT / "label_192" / "images"        # 192 labelled ICH-positive volumes
POS_TEST_DIR = BHSD_ROOT / "FirstStage_test"              # 50 volumes (no labels)
NEG_POOL_DIR = BHSD_ROOT / "unlabel_2000" / "nobleed"     # 1418 ICH-negative volumes

# ---- cohort ----
NEG_PER_POS = 5          # negative : positive ratio, in both train and test
SEED = 0

# ---- preprocessing ----
MARGIN_MM = 10.0                      # in-plane margin around the head bounding box
HEAD_THRESHOLD_HU = -300.0            # soft tissue / bone vs. air
SPACING_MM = 2.5                      # isotropic resampling (CT-FM pretraining resolution)
SPATIAL_SIZE = (96, 128, 128)         # (S, P, L) after SPL orientation; multiples of 16
WINDOW_PRESETS = {
    "full": (-1024.0, 2048.0),   # CT-FM pretraining window
    "brain": (-20.0, 100.0),     # Acute hemorrhage & brain parenchyma window
}
WINDOW_MODE = os.environ.get("BHSD_WINDOW_MODE", "brain")
HU_WINDOW = WINDOW_PRESETS.get(WINDOW_MODE, (-20.0, 100.0))
EXP_TAG = f"_{WINDOW_MODE}"
BACKBONE_ID = "project-lighter/ct_fm_feature_extractor"
