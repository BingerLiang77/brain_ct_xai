#!/usr/bin/env python3
"""Classical intracranial (brain) mask for non-contrast head CT. No GPU, no labels needed.

Idea: the skull is a closed bone ring in most axial slices. Close small gaps in that ring, fill it, keep what is
inside and has soft-tissue HU (brain, CSF, blood), then clean up in 3D. Scalp is outside the ring, sinus air is
excluded by HU, so neither is selected.

CLI:
  python brain_mask_classical.py run   --in_dir CT_DIR --out_dir MASK_DIR [--lesion_dir DIR] [--n_jobs 4]
  python brain_mask_classical.py qc    --in_dir CT_DIR --out_dir MASK_DIR --n 30
Masks are saved with the CT's own affine (canonical RAS array order). Always align by affine, never by array index.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

PARAMS = dict(
    smooth_mm=1.0,        # in-plane Gaussian before thresholding (noise); never smooths across slices (5 mm slices)
    bone_hu=250.0,        # >= this (after smoothing) is bone
    close_mm=4.0,         # closes sutures / small skull gaps before filling the ring
    rescue_min_frac=0.6,  # a slice is "suspicious" if its interior is < this fraction of its neighbours' largest interior
    rescue_close_mm=(8.0, 12.0, 16.0, 24.0, 36.0, 48.0),   # larger closings tried ONLY for slices where the ring did not close (skull base, craniectomy)
    tissue_lo=-10.0,      # soft-tissue range kept inside the ring (CSF ~0, brain 25-45, acute blood 50-90)
    tissue_hi=130.0,
    keep_comp_frac=0.15,  # keep every 3D component at least this fraction of the largest one
    erode_mm=4.0,         # breaks thin bridges (optic canal, foramina) before keeping the largest component
    final_close_mm=3.0,   # re-includes sulcal CSF / partial-volume voxels
    dilate_mm=3.0,        # safety margin so blood against the inner table is not clipped
    dilate_z=0,           # extra slices added above/below (0 = off). >0 can leak into scalp at the vertex.
)


def disk(r_px):
    r = max(int(round(r_px)), 0)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def closing_padded(img2d, st):
    r = st.shape[0] // 2
    p = np.pad(img2d, r, mode="constant")
    return ndi.binary_closing(p, st)[r:-r, r:-r] if r > 0 else img2d


def load_hu(path):
    """Returns (hu float32 canonical RAS, canonical nib image, notes). Fixes the common +1024 offset."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    d = np.asarray(img.dataobj).astype(np.float32)
    d = np.squeeze(d)
    if d.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape {d.shape}")
    notes = []
    if d.min() > -200:                      # air should be near -1000 (or at least clearly negative)
        if d.min() >= 0 and d.max() > 1500:
            d = d - 1024.0; notes.append("offset_-1024_applied")
        else:
            notes.append("NO_AIR_FOUND_unreliable")      # windowed / clipped image: bone cannot be separated
    if d.max() < 300:
        notes.append("NO_BONE_FOUND_unreliable")
    return d, img, notes


def compute_mask(hu, zooms, P=PARAMS):
    dx, dy, dz = [float(z) for z in zooms[:3]]
    px = (dx + dy) / 2.0
    sm = ndi.gaussian_filter(hu, sigma=(P["smooth_mm"] / dx, P["smooth_mm"] / dy, 0))
    bone = sm >= P["bone_hu"]
    def ring_interior(b, radius_mm):
        filled = ndi.binary_fill_holes(closing_padded(b, disk(radius_mm / px)))
        return filled & ~b

    nz = hu.shape[2]
    interior = np.zeros(hu.shape, bool)
    for k in range(nz):                                  # pass 1: base closing
        b = bone[:, :, k]
        if b.any():
            interior[:, :, k] = ring_interior(b, P["close_mm"])
    area = interior.sum(axis=(0, 1)).astype(float)
    rescued = 0
    max_rescue = 0.0
    for k in range(nz):                                  # pass 2: re-try slices whose ring did not close
        nb_ = [area[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < nz and area[j] > 0]
        ref = max(nb_) if nb_ else 0.0
        if ref == 0 or area[k] >= P["rescue_min_frac"] * ref or not bone[:, :, k].any():
            continue
        for rmm in P["rescue_close_mm"]:                 # accept the first radius that gives a plausible area
            inter = ring_interior(bone[:, :, k], rmm)
            if inter.sum() >= P["rescue_min_frac"] * ref:
                interior[:, :, k] = inter; area[k] = inter.sum(); rescued += 1
                if rmm > max_rescue:
                    max_rescue = float(rmm)
                break
    cand = interior & (sm >= P["tissue_lo"]) & (sm <= P["tissue_hi"])
    if not cand.any():
        return np.zeros(hu.shape, bool), dict(status="empty_after_ring_fill", max_rescue_mm_used=0.0)

    r = max(int(round(P["erode_mm"] / px)), 1)
    st = disk(r)[:, :, None]
    er = ndi.binary_erosion(cand, st)
    if not er.any():                                    # very small head / thin slices: skip the bridge-breaking step
        keep = cand
    else:
        lab, n = ndi.label(er, structure=np.ones((3, 3, 3)))
        sizes = ndi.sum(er, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, 1 + np.where(sizes >= P["keep_comp_frac"] * sizes.max())[0])   # brain may be split by a bad slice; eyes are far smaller
        st1 = disk(1)[:, :, None]
        for _ in range(r):                              # geodesic dilation back inside `cand`
            keep = ndi.binary_dilation(keep, st1) & cand

    m = keep.copy()
    st_fc = disk(P["final_close_mm"] / px)
    for k in range(m.shape[2]):
        s = m[:, :, k]
        if s.any():
            m[:, :, k] = ndi.binary_fill_holes(closing_padded(s, st_fc))
    rd = int(round(P["dilate_mm"] / px))
    if rd > 0:
        m = ndi.binary_dilation(m, disk(rd)[:, :, None])
    if P["dilate_z"] > 0:
        m = ndi.binary_dilation(m, np.ones((1, 1, 2 * P["dilate_z"] + 1), bool)) & (sm >= P["tissue_lo"]) & (sm <= P["tissue_hi"]) | m
    return m, dict(status="ok", n_slices_rescued=rescued, max_rescue_mm_used=max_rescue)


def process_one(args):
    ct_path, out_dir, lesion_dir, P = args
    stem = Path(ct_path).name.replace(".nii.gz", "").replace(".nii", "")
    out = Path(out_dir) / f"{stem}_brainmask.nii.gz"
    row = dict(case=stem)
    try:
        hu, img, notes = load_hu(ct_path)
        zooms = img.header.get_zooms()[:3]
        m, info = compute_mask(hu, zooms, P)
        row.update(info); row["notes"] = ";".join(notes)
        row.update(shape="x".join(map(str, hu.shape)), zx=zooms[0], zy=zooms[1], zz=zooms[2],
                   n_slices=hu.shape[2], mask_ml=float(m.sum() * np.prod(zooms) / 1000.0),
                   n_slices_mask=int(m.any(axis=(0, 1)).sum()), head_air_sd=float(hu[:8, :8, :].std()))
        brain_vals = hu[m & (hu > 0) & (hu < 100)]
        row.update(brain_hu_mean=float(brain_vals.mean()) if brain_vals.size else np.nan,
                   brain_hu_sd=float(brain_vals.std()) if brain_vals.size else np.nan)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part.nii.gz")
        nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine), str(tmp)); os.replace(tmp, out)
        if lesion_dir:                                   # QC ONLY: never use the lesion to change the mask
            lp = next((p for p in Path(lesion_dir).glob(f"{stem}*.nii*")), None)
            if lp is not None:
                les = np.squeeze(np.asarray(nib.as_closest_canonical(nib.load(str(lp))).dataobj)) > 0
                if les.shape == m.shape and les.any():
                    row["lesion_vox"] = int(les.sum()); row["lesion_in_mask"] = float((les & m).sum() / les.sum())
                elif les.shape != m.shape:
                    row["lesion_shape_mismatch"] = True
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}")
    return row


def run(in_dir, out_dir, lesion_dir=None, n_jobs=4, P=PARAMS):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    cts = sorted(list(in_dir.glob("*.nii.gz")) + list(in_dir.glob("*.nii")))
    csv = out_dir / "brainmask_qc.csv"
    done = set(pd.read_csv(csv)["case"]) if csv.exists() else set()
    todo = [p for p in cts if p.name.replace(".nii.gz", "").replace(".nii", "") not in done]
    print(f"{len(cts)} CTs, {len(done)} done, {len(todo)} to do")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, row in enumerate(ex.map(process_one, [(str(p), str(out_dir), lesion_dir, P) for p in todo]), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(todo):
                old = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(csv, index=False); rows = []
                print(f"  {i}/{len(todo)}")
    (out_dir / "params.json").write_text(json.dumps(P, indent=1))
    return pd.read_csv(csv)


def summarize(df):
    flags = pd.Series(False, index=df.index)
    if "notes" in df: flags |= df["notes"].astype(str).str.contains("unreliable")
    flags |= df["status"].ne("ok")
    flags |= ~df["mask_ml"].between(600, 2000)                       # warn only; children can be smaller
    if "lesion_in_mask" in df: flags |= df["lesion_in_mask"].lt(0.98)
    if "max_rescue_mm_used" in df: flags |= df["max_rescue_mm_used"].gt(24.0)
    df = df.assign(flag=flags)
    print(f"flagged {int(flags.sum())} / {len(df)}")
    return df


def qc_montage(in_dir, out_dir, n=30, seed=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cts = sorted(Path(in_dir).glob("*.nii*")); rng = np.random.default_rng(seed)
    pick = [cts[i] for i in rng.choice(len(cts), min(n, len(cts)), replace=False)]
    fig, ax = plt.subplots(len(pick), 6, figsize=(14, 2.3 * len(pick)), squeeze=False)
    for r, p in enumerate(pick):
        stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        hu, img, _ = load_hu(p)
        mp = Path(out_dir) / f"{stem}_brainmask.nii.gz"
        m = np.asarray(nib.load(str(mp)).dataobj) > 0 if mp.exists() else np.zeros(hu.shape, bool)
        ks = np.linspace(0, hu.shape[2] - 1, 8)[1:-1].astype(int)[:6]
        for c, k in enumerate(ks):
            ax[r, c].imshow(np.clip((hu[:, :, k].T - 0) / 80, 0, 1), cmap="gray", origin="lower")
            if m[:, :, k].any():
                ax[r, c].contour(m[:, :, k].T, levels=[0.5], colors="r", linewidths=0.8)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].set_title(f"{stem[-12:]} z={k}", fontsize=6)
    plt.tight_layout(); fp = Path(out_dir) / "qc_montage.png"; plt.savefig(fp, dpi=90); print("saved", fp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "qc"])
    ap.add_argument("--in_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lesion_dir"); ap.add_argument("--n_jobs", type=int, default=4); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "run":
        df = summarize(run(a.in_dir, a.out_dir, a.lesion_dir, a.n_jobs)); df.to_csv(Path(a.out_dir) / "brainmask_qc_flagged.csv", index=False)
    else:
        qc_montage(a.in_dir, a.out_dir, a.n)
