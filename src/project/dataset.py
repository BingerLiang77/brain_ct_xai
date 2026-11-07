"""ROI extraction + deterministic preprocessing for non-contrast head CT, with a tensor cache.

Pipeline per scan:  load NIfTI -> head bounding box (largest soft-tissue/bone component) -> crop
                    -> orientation SPL -> 2.5 mm isotropic resampling -> HU window -> pad/crop to a fixed size.
No masking, no augmentation.
"""
import hashlib
import json
import warnings
from pathlib import Path

import monai.transforms as mt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from scipy import ndimage as ndi
from torch.utils.data import Dataset

import config as C

CACHE_VERSION = "v1"


def head_roi_slices(arr, zooms, margin_mm=C.MARGIN_MM, thr=C.HEAD_THRESHOLD_HU):
    """Bounding box of the head: largest 3D connected component of (HU > thr) after a per-slice opening
    (removes the thin scanner-table / noise connections without eating z slices).

    In-plane box is padded by `margin_mm`; the z axis keeps the full scan extent (a head scan is already
    z-cropped, and trimming z could cut off a bleed).
    """
    m = ndi.binary_opening(arr > thr, structure=np.ones((3, 3, 1), bool), iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    coords = np.argwhere(lab == (1 + int(np.argmax(sizes))))
    mx, my = np.ceil(margin_mm / np.asarray(zooms[:2], dtype=float)).astype(int)
    lo = [max(coords[:, 0].min() - mx, 0), max(coords[:, 1].min() - my, 0), 0]
    hi = [min(coords[:, 0].max() + mx + 1, arr.shape[0]), min(coords[:, 1].max() + my + 1, arr.shape[1]), arr.shape[2]]
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def load_roi_volume(path, mask_path=None):
    """Cropped head ROI as a MetaTensor (1, X, Y, Z) that keeps the correct affine.
    If mask_path is provided, voxels outside the brain mask are set to -1000 HU (air/background).
    """
    img = nib.as_closest_canonical(nib.load(path))
    arr = img.get_fdata(dtype=np.float32)
    sl = head_roi_slices(arr, img.header.get_zooms())
    if sl is None:
        raise ValueError(f"No head found (nothing above {C.HEAD_THRESHOLD_HU} HU): {path}")
    if mask_path is not None:
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))
        mask_arr = (mask_nii.get_fdata() > 0)
        assert arr.shape == mask_arr.shape, f"Shape mismatch: {arr.shape} vs {mask_arr.shape}"
        arr = np.where(mask_arr, arr, -1000.0).astype(np.float32)
    sub = img.slicer[sl]
    return MetaTensor(torch.from_numpy(arr[sl])[None], affine=torch.as_tensor(sub.affine, dtype=torch.float64))


def get_transforms():
    geometry = mt.Compose([
        mt.Orientation(axcodes="SPL"),
        mt.Spacing(pixdim=(C.SPACING_MM,) * 3, mode="bilinear", padding_mode="border", dtype=np.float32),
    ])
    finish = mt.Compose([
        mt.ScaleIntensityRange(a_min=C.HU_WINDOW[0], a_max=C.HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        mt.ResizeWithPadOrCrop(spatial_size=list(C.SPATIAL_SIZE)),
    ])
    return geometry, finish


def cache_key(masked=False):
    params = dict(version=CACHE_VERSION, margin_mm=C.MARGIN_MM, thr=C.HEAD_THRESHOLD_HU, spacing=C.SPACING_MM,
                  size=list(C.SPATIAL_SIZE), window=list(C.HU_WINDOW), masked=masked)
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


class HeadCTDataset(Dataset):
    """Yields (tensor (1, S, P, L) in [0, 1], label, id). Preprocessed tensors are cached on disk."""

    def __init__(self, cohort, cache_dir=C.CACHE_DIR, masked=False):
        self.df = cohort.reset_index(drop=True)
        self.masked = masked
        self.cache_dir = Path(cache_dir) / cache_key(masked=masked)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.geometry, self.finish = get_transforms()

    def __len__(self):
        return len(self.df)

    def _preprocess(self, path, mask_path=None):
        vol = self.geometry(load_roi_volume(path, mask_path if self.masked else None))
        too_big = [i for i, (s, t) in enumerate(zip(vol.shape[1:], C.SPATIAL_SIZE)) if s > t]
        if too_big:
            warnings.warn(f"{Path(path).name}: ROI {tuple(vol.shape[1:])} exceeds {C.SPATIAL_SIZE} on axes {too_big}; centre-cropped")
        return self.finish(vol).as_tensor().clone().float()

    def __getitem__(self, i):
        r = self.df.iloc[i]
        f = self.cache_dir / f"{r['id']}.pt"
        if f.exists():
            x = torch.load(f, weights_only=True)
        else:
            x = self._preprocess(r["path"], r.get("mask_path") if self.masked else None)
            torch.save(x, f)
        return x, torch.tensor(int(r["label"])), r["id"]


if __name__ == "__main__":
    df = pd.read_csv(C.DATA_DIR / "cohort.csv")
    x, y, i = HeadCTDataset(df)[0]
    print(i, y.item(), tuple(x.shape), float(x.min()), float(x.max()))

