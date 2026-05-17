"""Export: writes the warped sample + cropped atlas as NIfTI and
multiscale OME-Zarr, using the same resample code path that step3 uses
for its WYSIWYG napari display.

Both the sample and the atlas go through one shared writer with one
identical diagonal affine `diag([z_um, y_um, x_um, 1])`, so any reader
either gets both axis-flipped or neither — the overlay you see in the
QC viewer matches what step3 showed.

With `--tps` and ≥4 landmark pairs from step3, a thin-plate spline is
fit to the *residuals* between the rigid prediction and your landmark
clicks (not to the absolute targets). Far from landmarks the correction
vanishes and the rigid result is preserved; at landmarks the sample
lands exactly on its CCF counterpart.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import dask.array as da
import nibabel as nib
import numpy as np

from ..atlas import load_ccf
from ..io import open_multiscale
from ..state import save as save_state
from ..transform import RigidTransform


def run(
    state_path: Path,
    out_dir: Path,
    level: Optional[int] = None,
    tps: bool = False,
    tps_smoothing: float = 0.0,
    skip_view: bool = False,
) -> None:
    from scipy.ndimage import affine_transform, map_coordinates
    from ..state import load

    state = load(state_path)
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run step1 first.")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    used_level = level if level is not None else state.sample_level
    print(f"\n[export] level={used_level}  atlas={state.atlas_name}  tps={tps}")

    # ---------- sample (same loader as step3) -------------------------------
    ms = open_multiscale(state.sample_zarr)
    arr = ms.level(used_level)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    sample_um = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                 else ms.spacing(used_level))

    # If the requested level still gives a huge volume, decimate down to a
    # CCF-comparable working resolution with BLOCK MEAN (anti-aliased). At
    # full level, this is a no-op.
    ccf = load_ccf(state.atlas_name)
    target_um = ccf.voxel_um[0]
    factors = tuple(max(1, int(round(target_um / s))) for s in sample_um)
    if any(f > 1 for f in factors):
        crop_s = tuple(slice(0, (arr.shape[i] // factors[i]) * factors[i])
                       for i in range(3))
        arr = arr[crop_s]
        coarsen_axes = {i: factors[i] for i in range(3)}
        arr = da.coarsen(np.mean, arr, coarsen_axes, trim_excess=True).astype(arr.dtype)
        sample_um = tuple(s * f for s, f in zip(sample_um, factors))
        print(f"[export] decimated x{factors} → {arr.shape} @ {sample_um} µm")
    print(f"[export] loading {arr.shape} into RAM...", flush=True)
    sample_np = np.ascontiguousarray(arr.compute())

    # ---------- atlas with crop (same as step3) -----------------------------
    ccf_ref_full = np.asarray(ccf.reference)
    b = state.ccf_crop_bbox
    if b is None:
        ccf_data = ccf_ref_full
        crop_voxel_origin = np.zeros(3)
    else:
        z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
        ccf_data = ccf_ref_full[z0:z1, y0:y1, x0:x1]
        crop_voxel_origin = np.array([z0, y0, x0], dtype=float)
    print(f"[export] CCF grid: {ccf_data.shape} @ {ccf.voxel_um} µm")

    # ---------- rigid transform (same as step3) -----------------------------
    rigid = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        flip_z=bool(state.transform.get("flip_z", False)),
        flip_y=bool(state.transform.get("flip_y", False)),
        flip_x=bool(state.transform.get("flip_x", False)),
        center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))),
    )

    # ---------- resample: COPY of step3._resample(), order=1 for export ----
    M = rigid.for_voxel_grid(sample_um, ccf.voxel_um)
    Minv = np.linalg.inv(M)
    offset = Minv[:3, :3] @ crop_voxel_origin + Minv[:3, 3]
    print(f"[export] rigid resample → {ccf_data.shape}...", flush=True)
    warped = affine_transform(
        sample_np, Minv[:3, :3], offset=offset,
        output_shape=ccf_data.shape, order=1, mode="constant", cval=0,
    )

    # ---------- optional TPS-ON-RIGID-RESIDUALS -----------------------------
    # We do NOT fit TPS directly on ccf_µm → sample_µm — with only a handful
    # of landmarks that throws away the perfectly good rigid alignment and
    # extrapolates wildly. Instead: keep the rigid everywhere, fit TPS only
    # to the *residuals* the landmarks imply (in sample-µm). Far from
    # landmarks the correction → 0 and we recover the rigid; at landmarks
    # the correction lands the sample point exactly on its CCF target.
    if tps:
        sample_lms_raw = state.landmarks.get("sample_um", []) if state.landmarks else []
        ccf_lms_raw    = state.landmarks.get("ccf_um",    []) if state.landmarks else []
        n_pairs = min(len(sample_lms_raw), len(ccf_lms_raw))
        if n_pairs < 4:
            print(f"[export] --tps requested but only {n_pairs} pairs — "
                  "skipping TPS (need ≥4). Writing rigid-only output.")
        else:
            from scipy.interpolate import RBFInterpolator
            sample_lms = np.asarray(sample_lms_raw[:n_pairs], dtype=float)
            ccf_lms    = np.asarray(ccf_lms_raw   [:n_pairs], dtype=float)

            # Rigid's prediction of "what sample µm point corresponds to
            # this CCF µm point": sample_pred = Minv_um @ ccf_µm.
            M_um = rigid.matrix()
            Minv_um = np.linalg.inv(M_um)
            ccf_h = np.hstack([ccf_lms, np.ones((n_pairs, 1))])
            rigid_pred_sample = (Minv_um @ ccf_h.T).T[:, :3]
            residuals = sample_lms - rigid_pred_sample
            res_mag = np.linalg.norm(residuals, axis=1)
            print(f"[export] TPS-on-residuals  n={n_pairs}  "
                  f"smoothing={tps_smoothing}")
            print(f"[export] per-landmark residual µm: "
                  f"min={res_mag.min():.1f}  mean={res_mag.mean():.1f}  "
                  f"max={res_mag.max():.1f}")

            tps_fn = RBFInterpolator(ccf_lms, residuals,
                                     kernel="thin_plate_spline",
                                     smoothing=float(tps_smoothing))

            # Build the same grid in CCF µm.
            zs = np.arange(ccf_data.shape[0]) + crop_voxel_origin[0]
            ys = np.arange(ccf_data.shape[1]) + crop_voxel_origin[1]
            xs = np.arange(ccf_data.shape[2]) + crop_voxel_origin[2]
            Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
            ccf_um_grid = np.stack([Z.ravel(), Y.ravel(), X.ravel()],
                                    axis=-1) * np.array(ccf.voxel_um)
            del Z, Y, X
            n_pts = ccf_um_grid.shape[0]

            # Rigid baseline: where each CCF µm point maps back in sample µm.
            ccf_h_grid = np.hstack([ccf_um_grid, np.ones((n_pts, 1))])
            sample_um_grid = (Minv_um @ ccf_h_grid.T).T[:, :3]
            del ccf_h_grid

            # TPS correction (chunked so a 50M-row eval doesn't OOM).
            chunk = 1_000_000
            print(f"[export] evaluating TPS at {n_pts:,} points "
                  f"(chunks of {chunk:,})...", flush=True)
            for i in range(0, n_pts, chunk):
                sample_um_grid[i:i+chunk] += tps_fn(ccf_um_grid[i:i+chunk])
            del ccf_um_grid

            sample_vox_grid = (sample_um_grid /
                                np.array(sample_um)).T
            del sample_um_grid
            print(f"[export] sampling via map_coordinates...", flush=True)
            warped = map_coordinates(
                sample_np, sample_vox_grid, order=1,
                mode="constant", cval=0,
            ).reshape(ccf_data.shape).astype(sample_np.dtype)
            del sample_vox_grid

    # ---------- write outputs through ONE shared writer --------------------
    # PLAIN DIAGONAL AFFINE — no brainglobe-orientation games. Both files
    # use the same affine, so any reader either gets both axis-flipped or
    # neither, and they overlay correctly.
    vox = np.asarray(ccf.voxel_um, dtype=float) * 1e-3   # µm → mm
    diag_affine = np.diag([vox[0], vox[1], vox[2], 1.0])

    sample_nii = out_dir / "sample_in_ccf.nii.gz"
    atlas_nii  = out_dir / "atlas_cropped.nii.gz"
    _save_nifti(warped, diag_affine, sample_nii)
    _save_nifti(ccf_data, diag_affine, atlas_nii)
    print(f"[export] wrote {sample_nii.name} + {atlas_nii.name}")

    sample_zarr = out_dir / "sample_in_ccf.zarr"
    atlas_zarr  = out_dir / "atlas_cropped.zarr"
    _save_ngff(warped,   sample_zarr, ccf.voxel_um)
    _save_ngff(ccf_data, atlas_zarr,  ccf.voxel_um)
    print(f"[export] wrote {sample_zarr.name} + {atlas_zarr.name}")

    state.add_history(
        "export",
        f"level={used_level} shape={list(ccf_data.shape)} tps={bool(tps)}")
    save_state(state, state_path)

    if skip_view:
        return
    _qc_view(sample_nii, atlas_nii, ccf.voxel_um)


def _save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), affine), str(path))


def _save_ngff(arr: np.ndarray, zarr_out: Path,
                voxel_um, n_levels: int = 5) -> None:
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    if zarr_out.exists():
        shutil.rmtree(zarr_out)
    store = parse_url(str(zarr_out), mode="w").store
    root = zarr.group(store=store)
    scales = [list(map(float, voxel_um))] * n_levels
    write_image(
        image=np.ascontiguousarray(arr), group=root, axes="zyx",
        coordinate_transformations=[[{"type": "scale", "scale": s}]
                                     for s in scales],
        storage_options=dict(chunks=(64, 64, 64)),
    )


def _qc_view(sample_path: Path, atlas_path: Path, voxel_um) -> None:
    """QC viewer that EXACTLY mirrors step3's display pattern.

    Both layers come through `nib.load(...).dataobj` (preserves on-disk
    axis order) and both are shown with `scale=voxel_um`. Symmetric in
    every way that matters for overlay correctness.
    """
    import napari
    sample = np.asarray(nib.load(str(sample_path)).dataobj)
    atlas  = np.asarray(nib.load(str(atlas_path)).dataobj)
    print(f"\n[export] QC: atlas {atlas.shape}  sample {sample.shape}")
    v = napari.Viewer(ndisplay=2, title="zrot export — QC")
    v.add_image(atlas, name="ATLAS (cropped)", scale=voxel_um,
                colormap="gray", blending="additive", opacity=0.5)
    v.add_image(sample, name="SAMPLE (warped)", scale=voxel_um,
                colormap="magenta", blending="additive", opacity=0.6)
    napari.run()
