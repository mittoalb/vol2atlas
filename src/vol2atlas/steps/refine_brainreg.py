"""Refine an `export` output with BrainGlobe brainreg.

Reads <export_dir>/sample_in_ccf.nii.gz (prealigned sample, atlas grid),
runs brainreg with --orientation matching the atlas (since the
prealigned sample is already in atlas-native axis order), then converts
brainreg's `downsampled_standard.tiff` back to the same NIfTI +
multiscale-zarr layout as `export` so `scripts/qc_export.py` Just Works.

WARNING: brainreg assumes a full brain or a complete hemisphere. On a
partial sample, the affine+B-spline fit tries to stretch your tissue
across the missing regions — the registration will likely look wrong
no matter what.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from ..atlas import load_ccf
from ..state import load as load_state, save as save_state


def run(
    export_dir: Path,
    out_dir: Path,
    atlas_name: str = "allen_mouse_25um",
    orientation: str = "asr",
    brain_geometry: str = "full",
    skip_view: bool = False,
    state_path: Path | None = None,
) -> None:
    if shutil.which("brainreg") is None:
        sys.exit("[brainreg] `brainreg` not on PATH. Run: pip install brainreg")

    export_dir = Path(export_dir); out_dir = Path(out_dir)
    sample_in  = export_dir / "sample_in_ccf.nii.gz"
    atlas_in   = export_dir / "atlas_cropped.nii.gz"
    if not sample_in.exists() or not atlas_in.exists():
        sys.exit(f"[brainreg] missing inputs in {export_dir} — run `vol2atlas export` first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Voxel sizes from the atlas (same grid as export's output).
    ccf      = load_ccf(atlas_name)
    voxel_um = tuple(float(v) for v in ccf.voxel_um)

    brainreg_out = out_dir / "brainreg_out"
    if brainreg_out.exists():
        shutil.rmtree(brainreg_out)
    cmd = [
        "brainreg", str(sample_in), str(brainreg_out),
        "-v", str(voxel_um[0]), str(voxel_um[1]), str(voxel_um[2]),
        "--orientation",    orientation,
        "--atlas",          atlas_name,
        "--brain_geometry", brain_geometry,
    ]
    print(f"[brainreg] running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(f"[brainreg] failed (exit {res.returncode}).")

    standard_tiff = brainreg_out / "downsampled_standard.tiff"
    if not standard_tiff.exists():
        sys.exit(f"[brainreg] expected {standard_tiff} not found.")

    import tifffile
    warped     = tifffile.imread(str(standard_tiff))
    atlas_nib  = nib.load(str(atlas_in))
    sample_out = out_dir / "sample_in_ccf.nii.gz"
    atlas_out  = out_dir / "atlas_cropped.nii.gz"
    nib.save(nib.Nifti1Image(np.ascontiguousarray(warped), atlas_nib.affine),
             str(sample_out))
    shutil.copy(str(atlas_in), str(atlas_out))
    print(f"[brainreg] wrote {sample_out.name} + {atlas_out.name}")

    _save_ngff(warped,                          out_dir / "sample_in_ccf.zarr",  voxel_um)
    _save_ngff(np.asarray(atlas_nib.dataobj),   out_dir / "atlas_cropped.zarr",  voxel_um)

    if state_path is not None and Path(state_path).exists():
        st = load_state(Path(state_path))
        st.add_history("brainreg",
                        f"geom={brain_geometry} orientation={orientation}")
        save_state(st, Path(state_path))

    if skip_view:
        return
    _qc(sample_out, atlas_out, voxel_um)


def _save_ngff(arr: np.ndarray, zarr_out: Path, voxel_um, n_levels: int = 5) -> None:
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    if zarr_out.exists():
        shutil.rmtree(zarr_out)
    store = parse_url(str(zarr_out), mode="w").store
    root  = zarr.group(store=store)
    write_image(
        image=np.ascontiguousarray(arr), group=root, axes="zyx",
        coordinate_transformations=[
            [{"type": "scale", "scale": list(map(float, voxel_um))}]
        ] * n_levels,
        storage_options=dict(chunks=(64, 64, 64)),
    )


def _qc(sample_path: Path, atlas_path: Path, voxel_um) -> None:
    import napari
    sample = np.asarray(nib.load(str(sample_path)).dataobj)
    atlas  = np.asarray(nib.load(str(atlas_path)).dataobj)
    v = napari.Viewer(ndisplay=2, title="vol2atlas brainreg — QC")
    v.add_image(atlas,  name="ATLAS",         scale=voxel_um,
                colormap="gray",    blending="additive", opacity=0.5)
    v.add_image(sample, name="brainreg warped", scale=voxel_um,
                colormap="magenta", blending="additive", opacity=0.6)
    napari.run()
