"""Export: writes the warped sample + cropped atlas as NIfTI and
multiscale OME-Zarr, using the same resample code path that the `landmarks` step uses
for its WYSIWYG napari display.

Both the sample and the atlas go through one shared writer with one
identical diagonal affine `diag([z_um, y_um, x_um, 1])`, so any reader
either gets both axis-flipped or neither — the overlay you see in the
QC viewer matches what `landmarks` showed.

With `--tps` and ≥4 landmark pairs from `landmarks`, a thin-plate spline is
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
from ..frame import compute_output_frame, extract_ccf
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
    write_transform: bool = False,
) -> None:
    from scipy.ndimage import affine_transform, map_coordinates
    from ..state import load

    state = load(state_path)
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run `vol2atlas prealign` first.")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    used_level = level if level is not None else state.sample_level
    print(f"\n[export] level={used_level}  atlas={state.atlas_name}  tps={tps}")

    # ---------- sample (same loader as `landmarks`) -------------------------------
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

    # ---------- atlas crop (= user's bbox, no union) ------------------
    ccf_ref_full = np.asarray(ccf.reference)
    b = state.ccf_crop_bbox
    if b is None:
        ccf_data = ccf_ref_full
        crop_voxel_origin = np.zeros(3)
    else:
        z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
        ccf_data = ccf_ref_full[z0:z1, y0:y1, x0:x1]
        crop_voxel_origin = np.array([z0, y0, x0], dtype=float)
    print(f"[export] CCF crop: shape={ccf_data.shape}  "
          f"origin(vox)={tuple(int(v) for v in crop_voxel_origin)}  @ {ccf.voxel_um} µm")

    rigid = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))),
    )

    # ---------- compose rigid + (optional) affine refinement -----------
    # Combined µm transform: sample_µm  --rigid-->  CCF_µm  --affine-->  CCF_µm
    M_um = rigid.matrix()
    if state.affine is not None:
        A_um = np.asarray(state.affine, dtype=float)
        M_um = A_um @ M_um
        print(f"[export] composing state.affine on top of rigid")
    # Convert combined µm transform to voxel space.
    S_in_smp  = np.diag([sample_um[0], sample_um[1], sample_um[2], 1.0])
    S_out_ccf = np.diag([1.0 / ccf.voxel_um[0], 1.0 / ccf.voxel_um[1],
                          1.0 / ccf.voxel_um[2], 1.0])
    M = S_out_ccf @ M_um @ S_in_smp
    Minv = np.linalg.inv(M)
    offset = Minv[:3, :3] @ crop_voxel_origin + Minv[:3, 3]
    print(f"[export] resample → {ccf_data.shape}...", flush=True)

    # Local refinements: if any are set in state, replace the affine_transform
    # call with a map_coordinates call that uses a per-voxel blended source
    # location. Pass-through (affine_transform) when no local refinements.
    local_refs = []
    if getattr(state, "local_refinements", None):
        from ..local_refinement import LocalRefinement, blended_inverse_sample_coords
        local_refs = [LocalRefinement.from_dict(d) for d in state.local_refinements]
        print(f"[export] composing {len(local_refs)} local refinement(s): "
              f"{[L.name for L in local_refs]}")

    if local_refs:
        # Build CCF voxel coordinate grid for the output crop, in
        # CHUNKS to keep peak memory bounded.
        warped = np.empty(ccf_data.shape, dtype=np.float32)
        z_step = max(1, min(64, ccf_data.shape[0]))
        for z0 in range(0, ccf_data.shape[0], z_step):
            z1 = min(z0 + z_step, ccf_data.shape[0])
            zz, yy, xx = np.meshgrid(
                np.arange(z0, z1, dtype=np.float32),
                np.arange(ccf_data.shape[1], dtype=np.float32),
                np.arange(ccf_data.shape[2], dtype=np.float32),
                indexing="ij",
            )
            coords = np.stack([zz, yy, xx], axis=0)   # (3, dz, Y, X)
            sample_voxel_coords = blended_inverse_sample_coords(
                coords, M_um, sample_um, ccf.voxel_um,
                tuple(crop_voxel_origin * np.asarray(ccf.voxel_um)),
                local_refinements=local_refs,
            )
            warped[z0:z1] = map_coordinates(
                sample_np, sample_voxel_coords,
                order=1, mode="constant", cval=0.0,
            ).astype(np.float32)
    else:
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

            # Baseline prediction = COMBINED transform (rigid + affine).
            # TPS is fit on the residuals between this baseline and
            # the actual landmark clicks. So if state.affine was set
            # (e.g., by "Fit AFFINE" in landmarks), TPS layers on top
            # of the affine; otherwise on top of rigid alone.
            M_um_base = rigid.matrix()
            if state.affine is not None:
                M_um_base = np.asarray(state.affine, dtype=float) @ M_um_base
            Minv_um = np.linalg.inv(M_um_base)
            ccf_h = np.hstack([ccf_lms, np.ones((n_pairs, 1))])
            baseline_pred_sample = (Minv_um @ ccf_h.T).T[:, :3]
            residuals = sample_lms - baseline_pred_sample
            res_mag = np.linalg.norm(residuals, axis=1)
            base_name = "affine" if state.affine is not None else "rigid"
            print(f"[export] TPS-on-residuals (baseline={base_name})  "
                  f"n={n_pairs}  smoothing={tps_smoothing}")
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

    # ---------- sample presence mask ---------------------------------------
    # Warp a ones-array through the same rigid transform to get a binary
    # "where the sample exists in CCF space" mask. Lets the QC viewer and
    # downstream tools tell the sample's zero-background apart from the
    # CCF's zero-background.
    print("[export] computing sample presence mask...", flush=True)
    mask_in = np.ones(sample_np.shape, dtype=np.uint8)
    sample_mask = affine_transform(
        mask_in, Minv[:3, :3], offset=offset,
        output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
    ).astype(np.uint8)

    # ---------- write outputs through ONE shared writer --------------------
    # PLAIN DIAGONAL AFFINE — no brainglobe-orientation games. All files
    # use the same affine, so any reader either gets both axis-flipped or
    # neither, and they overlay correctly.
    vox = np.asarray(ccf.voxel_um, dtype=float) * 1e-3   # µm → mm
    diag_affine = np.diag([vox[0], vox[1], vox[2], 1.0])

    sample_nii = out_dir / "sample_in_ccf.nii.gz"
    atlas_nii  = out_dir / "atlas_cropped.nii.gz"
    mask_nii   = out_dir / "sample_mask.nii.gz"
    _save_nifti(warped, diag_affine, sample_nii)
    _save_nifti(ccf_data, diag_affine, atlas_nii)
    _save_nifti(sample_mask, diag_affine, mask_nii)
    print(f"[export] wrote {sample_nii.name} + {atlas_nii.name} + {mask_nii.name}")

    sample_zarr = out_dir / "sample_in_ccf.zarr"
    atlas_zarr  = out_dir / "atlas_cropped.zarr"
    mask_zarr   = out_dir / "sample_mask.zarr"
    _save_ngff(warped,      sample_zarr, ccf.voxel_um)
    _save_ngff(ccf_data,    atlas_zarr,  ccf.voxel_um)
    _save_ngff(sample_mask, mask_zarr,   ccf.voxel_um)
    print(f"[export] wrote {sample_zarr.name} + {atlas_zarr.name} + {mask_zarr.name}")

    # ---------- portable transform export ---------------------------------
    if write_transform:
        from ..transform_io import (
            write_itk_mat, write_ngff_transform, write_landmarks_csv,
        )
        mat_path = out_dir / "transform.mat"
        ngff_path = out_dir / "transform.json"
        try:
            write_itk_mat(M_um, mat_path)
            print(f"[export] wrote {mat_path.name} "
                  f"(ITK GenericAffine; ANTs / Slicer readable)")
        except Exception as e:
            print(f"[export] WARN: could not write {mat_path.name}: {e}")
        write_ngff_transform(M_um, ccf.voxel_um, ngff_path,
                              atlas_name=state.atlas_name,
                              local_refinements=getattr(
                                  state, "local_refinements", None))
        print(f"[export] wrote {ngff_path.name} "
              f"(OME-NGFF coordinateTransformations)")
        if state.landmarks:
            smp_lms = state.landmarks.get("sample_um", [])
            ccf_lms = state.landmarks.get("ccf_um", [])
            if smp_lms and ccf_lms:
                csv_path = out_dir / "landmarks.csv"
                write_landmarks_csv(smp_lms, ccf_lms, csv_path)
                print(f"[export] wrote {csv_path.name} "
                      f"({min(len(smp_lms), len(ccf_lms))} pairs, "
                      f"BigWarp-format)")

    state.add_history(
        "export",
        f"level={used_level} shape={list(ccf_data.shape)} tps={bool(tps)} "
        f"write_transform={bool(write_transform)}")
    save_state(state, state_path)

    if skip_view:
        return
    _qc_view(sample_nii, atlas_nii, mask_nii, ccf.voxel_um)


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


def _qc_view(sample_path: Path, atlas_path: Path, mask_path: Path,
              voxel_um) -> None:
    """QC viewer. Uses the sample_mask to set out-of-sample voxels to
    NaN so napari renders them transparent — the sample background no
    longer washes over the CCF.
    """
    import napari
    sample = np.asarray(nib.load(str(sample_path)).dataobj).astype(np.float32)
    atlas  = np.asarray(nib.load(str(atlas_path)).dataobj)
    mask   = np.asarray(nib.load(str(mask_path )).dataobj).astype(bool)
    sample[~mask] = np.nan
    print(f"\n[export] QC: atlas {atlas.shape}  sample {sample.shape}  "
          f"(mask {mask.sum()/mask.size*100:.1f}% covered)")
    v = napari.Viewer(ndisplay=2, title="vol2atlas export — QC")
    v.add_image(atlas, name="ATLAS (cropped)", scale=voxel_um,
                colormap="gray", blending="additive", opacity=0.5)
    v.add_image(sample, name="SAMPLE (warped)", scale=voxel_um,
                colormap="gray", blending="additive", opacity=0.7)
    napari.run()
