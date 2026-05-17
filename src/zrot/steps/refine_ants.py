"""Refine an `export` output with ANTs SyNOnly + brain masks.

Reads <export_dir>/sample_in_ccf.nii.gz (prealigned sample, atlas grid)
and <export_dir>/atlas_cropped.nii.gz (the cropped atlas), runs ANTs
SyNOnly with masks built from the data, writes the warped result back
through the SAME diagonal-affine / multiscale-zarr writer that `export`
uses — so the QC viewer (`scripts/qc_export.py`) Just Works.

Why SyNOnly + masks (not SyN):
  - SyN runs a rigid+affine stage that overwrites the prealignment.
  - On a partial sample, an unmasked metric scores empty (zero) regions
    as "missing structure to be deformed in", so the displacement field
    runs wild. Masking restricts the metric to actual brain tissue.

Requires: pip install antspyx
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from ..state import load as load_state, save as save_state


def run(
    export_dir: Path,
    out_dir: Path,
    transform_type: str = "SyNOnly",
    reg_iterations: tuple = (10, 5, 0),
    flow_sigma: float = 6.0,
    total_sigma: float = 2.0,
    skip_view: bool = False,
    state_path: Path | None = None,
) -> None:
    try:
        import ants
    except ImportError:
        sys.exit("[ants] not installed. Run: pip install antspyx")

    export_dir = Path(export_dir); out_dir = Path(out_dir)
    sample_in  = export_dir / "sample_in_ccf.nii.gz"
    atlas_in   = export_dir / "atlas_cropped.nii.gz"
    if not sample_in.exists() or not atlas_in.exists():
        sys.exit(f"[ants] missing inputs in {export_dir} — run `zrot export` first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed       = ants.image_read(str(atlas_in))
    moving      = ants.image_read(str(sample_in))
    fixed_mask  = ants.get_mask(fixed)
    moving_mask = ants.get_mask(moving)
    print(f"[ants] fixed  {fixed.shape}  spacing {fixed.spacing}")
    print(f"[ants] moving {moving.shape}  spacing {moving.spacing}")
    print(f"[ants] type_of_transform = {transform_type}")
    print(f"[ants] reg_iterations    = {reg_iterations}")
    print(f"[ants] flow_sigma={flow_sigma}  total_sigma={total_sigma}")

    reg = ants.registration(
        fixed=fixed, moving=moving,
        mask=fixed_mask, moving_mask=moving_mask,
        type_of_transform=transform_type,
        initial_transform="Identity",
        reg_iterations=tuple(reg_iterations),
        flow_sigma=float(flow_sigma),
        total_sigma=float(total_sigma),
        verbose=True,
    )

    # Save warped output through the SAME pattern as export: read the
    # cropped atlas's affine (diagonal) and reuse it for the warped
    # sample so the QC viewer overlays them pixel-for-pixel.
    atlas_nib   = nib.load(str(atlas_in))
    diag_affine = atlas_nib.affine
    warped_np   = np.ascontiguousarray(reg["warpedmovout"].numpy())
    sample_out  = out_dir / "sample_in_ccf.nii.gz"
    atlas_out   = out_dir / "atlas_cropped.nii.gz"
    nib.save(nib.Nifti1Image(warped_np, diag_affine), str(sample_out))
    shutil.copy(str(atlas_in), str(atlas_out))
    print(f"[ants] wrote {sample_out.name} + {atlas_out.name}")

    # Multiscale OME-Zarr for both, mirroring export's layout.
    voxel_um = tuple(float(s) for s in fixed.spacing)
    _save_ngff(warped_np,                    out_dir / "sample_in_ccf.zarr",  voxel_um)
    _save_ngff(np.asarray(atlas_nib.dataobj), out_dir / "atlas_cropped.zarr", voxel_um)

    # Persist ANTs transforms for later application to other channels / levels.
    tx_dir = out_dir / "ants_transforms"
    if tx_dir.exists():
        shutil.rmtree(tx_dir)
    tx_dir.mkdir()
    for i, p in enumerate(reg.get("fwdtransforms", [])):
        if Path(p).exists():
            shutil.copy(p, tx_dir / f"fwd_{i}_{Path(p).name}")
    for i, p in enumerate(reg.get("invtransforms", [])):
        if Path(p).exists():
            shutil.copy(p, tx_dir / f"inv_{i}_{Path(p).name}")
    print(f"[ants] saved ANTs transforms to {tx_dir}/")

    if state_path is not None and Path(state_path).exists():
        st = load_state(Path(state_path))
        st.add_history("ants",
                        f"transform={transform_type} reg_iter={reg_iterations} "
                        f"flow_sigma={flow_sigma} total_sigma={total_sigma}")
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
    v = napari.Viewer(ndisplay=2, title="zrot ANTs — QC")
    v.add_image(atlas,  name="ATLAS",   scale=voxel_um,
                colormap="gray",    blending="additive", opacity=0.5)
    v.add_image(sample, name="ANTs warped", scale=voxel_um,
                colormap="magenta", blending="additive", opacity=0.6)
    napari.run()
