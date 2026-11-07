import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from dataset import HeadCTDataset, cache_key

def process_one(args):
    row_dict, masked, cache_dir = args
    cid = row_dict["id"]
    out_pt = Path(cache_dir) / f"{cid}.pt"
    if out_pt.exists():
        return cid, "cached"
    try:
        from dataset import load_roi_volume, get_transforms
        geometry, finish = get_transforms()
        vol = geometry(load_roi_volume(row_dict["path"], row_dict["mask_path"] if masked else None))
        x = finish(vol).as_tensor().clone().float()
        torch.save(x, out_pt)
        return cid, "ok"
    except Exception as e:
        return cid, f"error: {type(e).__name__}: {e}"

def main(n_jobs=8, masked=True):
    cohort_path = C.DATA_DIR / "cohort_masked.csv"
    df = pd.read_csv(cohort_path)
    ckey = cache_key(masked=masked)
    cache_dir = C.CACHE_DIR / ckey
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building cache for {len(df)} scans | masked={masked} | cache_dir={cache_dir} | workers={n_jobs}")
    
    tasks = [(r.to_dict(), masked, str(cache_dir)) for _, r in df.iterrows()]
    
    t0 = time.time()
    errors = []
    cached_count = 0
    ok_count = 0
    
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for cid, status in tqdm(ex.map(process_one, tasks), total=len(tasks)):
            if status == "ok":
                ok_count += 1
            elif status == "cached":
                cached_count += 1
            else:
                errors.append((cid, status))
                print(f"Error on {cid}: {status}")
                
    elapsed = time.time() - t0
    print(f"\nCache Generation Complete in {elapsed:.1f}s!")
    print(f"Newly processed: {ok_count} | Already cached: {cached_count} | Errors: {len(errors)}")
    assert len(errors) == 0, f"{len(errors)} errors occurred during caching"
    
    n_files = len(list(cache_dir.glob("*.pt")))
    print(f"Total .pt files in {cache_dir}: {n_files}")
    assert n_files == len(df), f"Expected {len(df)} files, got {n_files}"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--masked", action="store_true", default=True)
    a = ap.parse_args()
    main(a.n_jobs, a.masked)
