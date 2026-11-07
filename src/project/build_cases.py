import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parents[1]
sys.path.insert(0, str(current_dir))

import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import nibabel as nib
import numpy as np
import monai.transforms as mt
from monai.data import MetaTensor
import config as C
import xai_common as xc
from dataset import head_roi_slices
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

proj_dir = Path(os.environ.get("PROJECT_DIR", str(repo_root)))
out_dir = Path(os.environ.get("CASES_OUT_DIR", str(proj_dir / "segcls_pack" / "data" / "cases")))
qc_pos = None
coh = None

def worker(row_dict):
    cid = row_dict["id"]
    target_path = out_dir / f"{cid}.pt"
    if target_path.exists():
        return cid, True
        
    try:
        # 1. x: preprocessed cached tensor
        x_pt = xc.load_tensor(cid).half()
        
        # 2. head: (x[0] > 0.01)
        head_mask = (x_pt[0] > 0.01).to(torch.uint8)
        
        # 3. brain: resample mask_path
        img = nib.as_closest_canonical(nib.load(row_dict["path"]))
        arr = img.get_fdata(dtype=np.float32)
        sl = head_roi_slices(arr, img.header.get_zooms())
        mask_nii = nib.as_closest_canonical(nib.load(row_dict["mask_path"]))
        mask_arr = (mask_nii.get_fdata() > 0)[sl]
        vol = MetaTensor(torch.from_numpy(mask_arr.astype(np.float32))[None],
                         affine=torch.as_tensor(img.slicer[sl].affine, dtype=torch.float64))
        geometry = mt.Compose([mt.Orientation(axcodes="SPL"),
                               mt.Spacing(pixdim=(C.SPACING_MM,) * 3, mode="nearest", padding_mode="zeros")])
        brain_mask = mt.ResizeWithPadOrCrop(spatial_size=list(C.SPATIAL_SIZE))(geometry(vol)).as_tensor()[0].to(torch.uint8)
        
        # 4. lesion mask
        if row_dict["label"] == 1:
            l_mask, info = xc.lesion_mask(cid, row_dict["path"], check=False)
            lesion_mask = l_mask.to(torch.uint8)
            lesion_vox = int(info["model_vox"])
            lesion_ml = float(info["model_ml"])
            lesion_in_mask = float(qc_pos.loc[cid, "lesion_in_mask"]) if cid in qc_pos.index else 1.0
        else:
            lesion_mask = torch.zeros((96, 128, 128), dtype=torch.uint8)
            lesion_vox = 0
            lesion_ml = 0.0
            lesion_in_mask = None

        meta = {
            "id": cid,
            "patient": str(row_dict["patient"]),
            "label": int(row_dict["label"]),
            "split": str(row_dict["split"]),
            "fold": int(row_dict["fold"]),
            "lesion_vox": lesion_vox,
            "lesion_ml": lesion_ml,
            "lesion_in_mask": lesion_in_mask
        }
        
        data = {
            "x": x_pt,
            "brain": brain_mask,
            "head": head_mask,
            "lesion": lesion_mask,
            "meta": meta
        }
        torch.save(data, target_path)
        return cid, False
    except Exception as e:
        return cid, str(e)

def main():
    global coh, qc_pos, out_dir
    parser = argparse.ArgumentParser(description="Build packaged case .pt files for segcls_pack")
    parser.add_argument("--cohort", type=str, default=os.environ.get("COHORT_CSV", str(proj_dir / "data" / "cohort_masked.csv")),
                        help="Path to cohort_masked.csv or segcls_pack/cohort.csv")
    parser.add_argument("--qc-pos", type=str, default=os.environ.get("QC_POS_CSV", str(proj_dir / "data" / "brain_mask" / "positives_192" / "masks" / "brainmask_qc.csv")),
                        help="Path to brainmask_qc.csv for positives")
    parser.add_argument("--out-dir", type=str, default=os.environ.get("CASES_OUT_DIR", str(proj_dir / "segcls_pack" / "data" / "cases")),
                        help="Output directory for cases/*.pt")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coh_path = Path(args.cohort)
    assert coh_path.exists(), f"Cohort file not found: {coh_path}"
    coh = pd.read_csv(coh_path)

    qc_path = Path(args.qc_pos)
    if qc_path.exists():
        qc_pos = pd.read_csv(qc_path).set_index("case")
    else:
        print(f"Notice: {qc_path} not found, lesion_in_mask will default to 1.0")
        qc_pos = None

    records = coh.to_dict(orient="records")
    print(f"Starting processing {len(records)} cases with {args.workers} workers...", flush=True)
    t0 = time.time()
    n_done, n_skipped, n_err = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r) for r in records]
        for f in as_completed(futures):
            cid, res = f.result()
            if res is True:
                n_skipped += 1
            elif res is False:
                n_done += 1
            else:
                print(f"Error on {cid}: {res}", flush=True)
                n_err += 1
            total = n_done + n_skipped + n_err
            if total % 100 == 0 or total == len(records):
                print(f"Progress: {total}/{len(records)} (done: {n_done}, skipped: {n_skipped}, err: {n_err}) in {time.time()-t0:.1f}s", flush=True)

    print(f"All done in {time.time()-t0:.1f}s! Total files in {out_dir}: {len(os.listdir(out_dir))}", flush=True)

if __name__ == "__main__":
    main()
