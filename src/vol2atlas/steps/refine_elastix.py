"""Refine an `export` output with elastix B-spline FFD (Humbel et al. recipe).

Reads <export_dir>/sample_in_ccf.nii.gz + atlas_cropped.nii.gz, runs
elastix B-spline non-rigid registration with cross-modality MI metric
and a heavy bending-energy penalty (Humbel 2024: grid 16 voxels,
bending-energy weight 1000), writes the warped sample back in the same
NIfTI + multiscale-zarr layout as `export` so the QC viewer Just Works.

Requires: pip install itk-elastix
(NOT plain `SimpleITK` — its pip wheel does not ship the elastix bindings.)
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
    grid_voxels: int = 16,
    bending_energy: float = 1000.0,
    iterations: int = 1000,
    resolutions: int = 4,
    metric: str = "AdvancedMattesMutualInformation",
    use_masks: bool = True,
    skip_view: bool = False,
    state_path: Path | None = None,
) -> None:
    try:
        import itk
    except ImportError:
        sys.exit("[elastix] itk-elastix not installed. Run: pip install itk-elastix")
    if not hasattr(itk, "elastix_registration_method"):
        sys.exit("[elastix] `itk` is installed but without elastix support. "
                 "Run: pip install itk-elastix")

    export_dir = Path(export_dir); out_dir = Path(out_dir)
    sample_in  = export_dir / "sample_in_ccf.nii.gz"
    atlas_in   = export_dir / "atlas_cropped.nii.gz"
    if not sample_in.exists() or not atlas_in.exists():
        sys.exit(f"[elastix] missing inputs in {export_dir} — "
                 "run `vol2atlas export` first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed  = itk.imread(str(atlas_in),  itk.F)
    moving = itk.imread(str(sample_in), itk.F)
    print(f"[elastix] fixed  size={itk.size(fixed)}   spacing={itk.spacing(fixed)}")
    print(f"[elastix] moving size={itk.size(moving)}  spacing={itk.spacing(moving)}")
    print(f"[elastix] B-spline  grid={grid_voxels} voxels  "
          f"bending_energy={bending_energy}  iter={iterations}  "
          f"resolutions={resolutions}  metric={metric}")

    # Parameter map: B-spline default + overrides for Humbel's regime.
    # NOTE: elastix's default B-spline map ships `FinalGridSpacingInPhysicalUnits`
    # and refuses to have both that and `FinalGridSpacingInVoxels`. Convert the
    # user's voxel-grid intent into physical units (mm) using the fixed image's
    # spacing and set ONLY the physical-units key.
    spacing_mm = float(itk.spacing(fixed)[0])   # isotropic atlas → take axis 0
    final_grid_mm = grid_voxels * spacing_mm
    print(f"[elastix] FinalGridSpacingInPhysicalUnits = {final_grid_mm:.4f} mm "
          f"({grid_voxels} voxels × {spacing_mm:.4f} mm/voxel)")

    pmap = itk.ParameterObject.New().GetDefaultParameterMap("bspline")
    pmap["FinalGridSpacingInPhysicalUnits"] = (f"{final_grid_mm}",)
    pmap["MaximumNumberOfIterations"]       = (str(iterations),)
    pmap["NumberOfResolutions"]             = (str(resolutions),)
    pmap["Registration"]                    = ("MultiMetricMultiResolutionRegistration",)
    pmap["Metric"]                          = (metric, "TransformBendingEnergyPenalty")
    pmap["Metric0Weight"]                   = ("1.0",)
    pmap["Metric1Weight"]                   = (str(bending_energy),)
    pmap["ResultImageFormat"]               = ("nii.gz",)
    # Sampler choice on a masked partial sample is tricky:
    #   - Random           (default) picks N points anywhere in the fixed
    #                       image, then transforms them through the current
    #                       B-spline guess. Boundary samples that land
    #                       outside the moving mask trigger
    #                       "Too many samples map outside moving image buffer".
    #   - Full             uses every fixed voxel; deterministic but does NOT
    #                       support NewSamplesEveryIteration=true (which the
    #                       default B-spline param map sets).
    #   - RandomSparseMask is the sampler elastix's own warning recommends
    #                       for mask-related problems: it draws random samples
    #                       restricted to the *mask interior*, avoiding the
    #                       boundary-outside-buffer failure.
    # Why RandomCoordinate (not RandomSparseMask):
    # RandomSparseMask only draws from voxels marked valid in the mask.
    # On a small partial-sample mask it found 342 candidates and elastix
    # bailed despite 99.7% being inside-buffer (a hardcoded check we can't
    # tune from the param map). RandomCoordinate draws continuously
    # (sub-voxel positions) over the masked region, which is what the
    # elastix model-zoo presets use for mask-based registration.
    pmap["ImageSampler"]                    = ("RandomCoordinate",)
    pmap["NumberOfSpatialSamples"]          = ("5000",)
    pmap["NewSamplesEveryIteration"]        = ("true",)
    pmap["UseRandomSampleRegion"]           = ("false",)
    pmap["MaximumNumberOfSamplingAttempts"] = ("50",)
    pmap["RequiredRatioOfValidSamples"]     = ("0.05",)
    # The ASGD optimizer runs an "automatic parameter estimation" phase
    # BEFORE the real iterations that samples gradients with its own internal
    # sampler. That phase ignores our mask settings and bails with
    # "Too many samples map outside moving image buffer" on partial samples.
    # Disable it; supply manual step parameters instead.
    pmap["AutomaticParameterEstimation"]    = ("false",)
    pmap["SP_a"]                            = ("1.0",)
    pmap["SP_A"]                            = ("20.0",)
    pmap["SP_alpha"]                        = ("0.602",)

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(pmap)

    kwargs = {"parameter_object": parameter_object, "log_to_console": True}
    if use_masks:
        fixed_mask  = _otsu_mask(fixed)
        moving_mask = _otsu_mask(moving)
        kwargs["fixed_mask"]  = fixed_mask
        kwargs["moving_mask"] = moving_mask
        print("[elastix] using Otsu masks on both images")

    result_image, _ = itk.elastix_registration_method(fixed, moving, **kwargs)

    # Wrap into vol2atlas's NIfTI + ngff layout.
    atlas_nib  = nib.load(str(atlas_in))
    warped_np  = np.ascontiguousarray(itk.array_view_from_image(result_image)).copy()
    sample_out = out_dir / "sample_in_ccf.nii.gz"
    atlas_out  = out_dir / "atlas_cropped.nii.gz"
    nib.save(nib.Nifti1Image(warped_np, atlas_nib.affine), str(sample_out))
    shutil.copy(str(atlas_in), str(atlas_out))
    print(f"[elastix] wrote {sample_out.name} + {atlas_out.name}")

    voxel_um = tuple(float(z) * 1000.0
                      for z in atlas_nib.header.get_zooms()[:3])
    _save_ngff(warped_np,                     out_dir / "sample_in_ccf.zarr",  voxel_um)
    _save_ngff(np.asarray(atlas_nib.dataobj), out_dir / "atlas_cropped.zarr", voxel_um)

    if state_path is not None and Path(state_path).exists():
        st = load_state(Path(state_path))
        st.add_history("elastix",
                        f"grid={grid_voxels} bending={bending_energy} "
                        f"iter={iterations} res={resolutions} metric={metric}")
        save_state(st, Path(state_path))

    if skip_view:
        return
    _qc(sample_out, atlas_out, voxel_um)


def _otsu_mask(image):
    """Otsu-thresholded brain mask as a uint8 ITK image."""
    import itk
    arr  = itk.array_view_from_image(image)
    # Otsu threshold via numpy for portability (skimage would also work).
    flat = arr[arr > 0]
    if flat.size == 0:
        thresh = 0.0
    else:
        hist, edges = np.histogram(flat, bins=256)
        # 1-D Otsu on the positive-intensity histogram.
        p = hist / max(hist.sum(), 1)
        omega = np.cumsum(p)
        mu    = np.cumsum(p * (edges[:-1] + np.diff(edges) / 2))
        muT   = mu[-1]
        denom = omega * (1 - omega)
        denom[denom == 0] = 1e-12
        sigma_b2 = (muT * omega - mu) ** 2 / denom
        k        = int(np.argmax(sigma_b2))
        thresh   = float(edges[k])
    mask_np = (arr > thresh).astype(np.uint8)
    mask    = itk.image_from_array(mask_np)
    mask.CopyInformation(image)
    return mask


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
    v = napari.Viewer(ndisplay=2, title="vol2atlas elastix — QC")
    v.add_image(atlas,  name="ATLAS",          scale=voxel_um,
                colormap="gray",    blending="additive", opacity=0.5)
    v.add_image(sample, name="elastix warped", scale=voxel_um,
                colormap="magenta", blending="additive", opacity=0.6)
    napari.run()
