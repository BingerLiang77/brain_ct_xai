"""Shared helpers for the XAI analysis of the ICH classifier (masked, brain, or full window).

Supports:
- "masked" / "masked_brain": brain-masked volumes, [-20, 100] HU window (default)
- "brain": unmasked brain window, [-20, 100] HU
- "full": unmasked full CT-FM window, [-1024, 2048] HU
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("BHSD_WINDOW_MODE", "masked")
WINDOW_MODE = os.environ.get("BHSD_WINDOW_MODE", "masked")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import monai.transforms as mt
from lighter_zoo import SegResEncoder
from monai.data import MetaTensor

import config as C
from dataset import cache_key, head_roi_slices

if WINDOW_MODE in ("masked", "masked_brain"):
    CKPT_PATH = C.CHECKPOINT_DIR / "ich_head_masked_brain_brain_stages_0_1_2_3_4.pt"
    PRED_CSV = C.RESULT_DIR / "test_predictions_masked_brain_brain.csv"
    COHORT_CSV = C.DATA_DIR / "cohort_masked.csv"
    XAI_DIR = C.DATA_DIR / "xai_masked"
    OUT_DIR = C.RESULT_DIR / "xai_masked"
    CACHE_KEY = "7930994d"
    HU_WINDOW = (-20.0, 100.0)
elif WINDOW_MODE == "brain":
    CKPT_PATH = C.CHECKPOINT_DIR / "ich_head_annotated_brain_stages_0_1_2_3_4.pt"
    PRED_CSV = C.RESULT_DIR / "test_predictions_annotated_brain.csv"
    COHORT_CSV = C.DATA_DIR / "cohort_annotated.csv"
    XAI_DIR = C.DATA_DIR / "xai_brain"
    OUT_DIR = C.RESULT_DIR / "xai_brain"
    CACHE_KEY = "6031f083"
    HU_WINDOW = (-20.0, 100.0)
else:
    CKPT_PATH = C.CHECKPOINT_DIR / "ich_head_annotated_full_stages_0_1_2_3_4.pt"
    PRED_CSV = C.RESULT_DIR / "test_predictions_annotated_full.csv"
    COHORT_CSV = C.DATA_DIR / "cohort_annotated.csv"
    XAI_DIR = C.DATA_DIR / "xai"
    OUT_DIR = C.RESULT_DIR / "xai"
    CACHE_KEY = "0b5b7fde"
    HU_WINDOW = (-1024.0, 2048.0)

GT_DIR = C.BHSD_ROOT / "label_192" / "ground truths"
HEAD_HU = C.HEAD_THRESHOLD_HU


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ICHModel(nn.Module):
    """Frozen CT-FM encoder -> global-average-pool of every stage -> standardise -> linear head -> logit."""

    def __init__(self, ckpt_path=CKPT_PATH):
        super().__init__()
        ck = torch.load(ckpt_path, weights_only=True)
        assert tuple(ck["pretrain_config"]["hu_window"]) == tuple(HU_WINDOW), (
            f"window mismatch: ckpt has {ck['pretrain_config']['hu_window']} vs expected {HU_WINDOW}"
        )
        self.backbone = SegResEncoder.from_pretrained(ck["pretrain_config"]["backbone_id"])
        self.stages = list(ck["stages"])
        for n in ("mean", "std", "w"):
            self.register_buffer(n, ck[n].float())
        self.register_buffer("b", torch.tensor(float(ck["b"])))
        self.threshold = float(ck["threshold"])
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)               # BatchNorm must keep its pretrained statistics

    def forward(self, x, return_feats=False):
        fs = self.backbone(x)
        z = torch.cat([fs[s].mean(dim=(2, 3, 4)) for s in self.stages], dim=1)
        logit = ((z - self.mean) / self.std) @ self.w + self.b
        return (logit, fs) if return_feats else logit


def load_model(device=None):
    return ICHModel().to(device or pick_device())


def tensor_path(cid):
    return C.CACHE_DIR / CACHE_KEY / f"{cid}.pt"


def load_tensor(cid):
    """Cached preprocessed CT, (1, 96, 128, 128) in [0, 1]."""
    return torch.load(tensor_path(cid), weights_only=True)


def head_mask(x):
    """Voxels that are brain tissue in a normalised tensor; air/extracranial outside is excluded."""
    if WINDOW_MODE in ("brain", "masked", "masked_brain"):
        return x > 0.01
    lo, hi = HU_WINDOW
    return x > (HEAD_HU - lo) / (hi - lo)


def to_hu(x):
    lo, hi = HU_WINDOW
    return x * (hi - lo) + lo


def load_cases():
    """Test split with stored scores and TP/FP/TN/FN outcome."""
    cohort = pd.read_csv(COHORT_CSV)
    test = cohort[cohort["split"] == "test"]
    pred = pd.read_csv(PRED_CSV)[["id", "score", "pred"]]
    df = test.merge(pred, on="id", validate="1:1").reset_index(drop=True)
    df["outcome"] = np.select(
        [(df.label == 1) & (df.pred == 1), (df.label == 0) & (df.pred == 1),
         (df.label == 0) & (df.pred == 0), (df.label == 1) & (df.pred == 0)], ["TP", "FP", "TN", "FN"], default="")
    assert (df["outcome"] != "").all()
    return df


def lesion_mask(cid, image_path, check=True):
    """Ground-truth (any-subtype) lesion mask in the model's input space (96, 128, 128), bool.

    The mask goes through exactly the same crop / orientation / resampling / pad-crop as the CT
    (bilinear on a float mask, then > 0.5). Returns (mask, info) where info has voxel counts before/after
    resampling so lost small lesions are visible.
    """
    img = nib.as_closest_canonical(nib.load(image_path))
    gt = nib.as_closest_canonical(nib.load(str(GT_DIR / f"{cid}.nii.gz")))
    if gt.shape != img.shape:
        raise ValueError(f"{cid}: ground truth shape {gt.shape} differs from image {img.shape}")
    if not np.allclose(gt.affine, img.affine, atol=1e-2):
        print(f"[Warning] {cid}: ground truth affine differs slightly from image, using image geometry.")
    arr = img.get_fdata(dtype=np.float32)
    sl = head_roi_slices(arr, img.header.get_zooms())
    m = (np.asanyarray(gt.dataobj) > 0)
    vox_mm3 = float(np.prod(nib.affines.voxel_sizes(img.affine)))
    n_native = int(m.sum())
    cropped = m[sl]
    vol = MetaTensor(torch.from_numpy(cropped.astype(np.float32))[None],
                     affine=torch.as_tensor(img.slicer[sl].affine, dtype=torch.float64))
    geometry = mt.Compose([mt.Orientation(axcodes="SPL"),
                           mt.Spacing(pixdim=(C.SPACING_MM,) * 3, mode="bilinear", padding_mode="zeros", dtype=np.float32)])
    out = mt.ResizeWithPadOrCrop(spatial_size=list(C.SPATIAL_SIZE))(geometry(vol)).as_tensor()[0] > 0.5
    info = dict(native_vox=n_native, native_ml=n_native * vox_mm3 / 1000.0, outside_roi_vox=n_native - int(cropped.sum()),
                model_vox=int(out.sum()), model_ml=float(out.sum()) * C.SPACING_MM ** 3 / 1000.0)
    return out, info


def mean_negative_baseline():
    """Mean of all training-set negative CTs in model space, (1, 96, 128, 128). Cached; uses no test data."""
    path = XAI_DIR / "baseline_meanneg.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    coh = pd.read_csv(COHORT_CSV)
    ids = coh[(coh["split"] == "train") & (coh["label"] == 0)]["id"].tolist()
    acc = torch.zeros(1, *C.SPATIAL_SIZE)
    for c in ids:
        acc += load_tensor(c)
    mn = acc / len(ids)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(mn, path)
    return mn
