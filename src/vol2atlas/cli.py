"""Step-based CLI for the vol2atlas → brainreg workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False,
                  help="Step-based alignment of OME-Zarr volumes to the Allen CCF.")


@app.command()
def init(
    zarr_path: Path = typer.Argument(..., help="Path to OME-Zarr group."),
    state: Path = typer.Option(Path("state.json"), "--state", "-s"),
    level: int = typer.Option(2, "--level", "-l"),
    channel: int = typer.Option(0, "--channel", "-c"),
    atlas: str = typer.Option(
        "allen_mouse_25um", "--atlas",
        help="BrainGlobe atlas name. Allen CCFv3 at: allen_mouse_10um, "
             "_25um (default), _50um, _100um. Other mouse atlases: "
             "kim_mouse_*, osten_mouse_*, princeton_mouse_20um, "
             "gubra_mouse_20um, allen_mouse_bluebrain_ccfv3_augmented_*, "
             "etc. Run `vol2atlas list-atlases` for the full set."),
    voxel_um: Optional[str] = typer.Option(
        None, "--voxel-um",
        help="Override source voxel size in µm. Single float or comma-separated z,y,x."),
    sample_orientation: Optional[str] = typer.Option(
        None, "--orientation",
        help="3-letter BrainGlobe orientation code for the SAMPLE volume "
             "(R/L, A/P, S/I — e.g. 'asr', 'psr', 'sai'). Each letter "
             "says which anatomical direction the corresponding numpy "
             "axis INCREASES TOWARD: R=right, A=anterior, S=superior, "
             "L=left, P=posterior, I=inferior. Allen CCF default is "
             "'asr'. If supplied, init pre-populates state.transform "
             "with the rotation that maps sample → atlas orientation, "
             "so prealign opens with the sample already roughly aligned "
             "(no manual ±90° / 180° rotations needed)."),
):
    """Create a new state.json from a zarr path. Does not open napari."""
    from .io import open_multiscale
    from .state import State, save as save_state
    from .atlas import load_ccf
    from .orientation import rotation_between, euler_zyx_degrees_from_matrix
    from .transform import RigidTransform

    try:
        ms = open_multiscale(zarr_path)
    except Exception as e:
        typer.secho(f"could not open zarr: {e}", fg="red", err=True); raise typer.Exit(2)
    if not 0 <= level < ms.n_levels():
        typer.secho(
            f"--level {level} not in 0..{ms.n_levels()-1}\n\n{ms.summary()}",
            fg="red", err=True); raise typer.Exit(2)

    vox_list = None
    if voxel_um:
        parts = [float(p) for p in voxel_um.split(",")]
        if len(parts) == 1: parts = parts * 3
        if len(parts) != 3:
            typer.secho(f"--voxel-um needs 1 or 3 floats, got {voxel_um}",
                        fg="red", err=True); raise typer.Exit(2)
        vox_list = parts

    initial_transform = None
    if sample_orientation:
        try:
            ccf = load_ccf(atlas)
        except Exception as e:
            typer.secho(f"could not load atlas {atlas!r}: {e}",
                        fg="red", err=True); raise typer.Exit(2)
        try:
            R = rotation_between(sample_orientation, ccf.orientation)
        except ValueError as e:
            typer.secho(str(e), fg="red", err=True); raise typer.Exit(2)
        rz, ry, rx = euler_zyx_degrees_from_matrix(R)
        # Use sample volume center as the rotation pivot so the
        # pre-rotation doesn't push the sample out of view.
        sample_shape = ms.level(level).shape
        if "c" in ms.axes:
            sample_shape = sample_shape[1:]   # drop channel dim
        sample_sp = (tuple(vox_list) if vox_list else ms.spacing(level))
        center_um = tuple((s - 1) * v / 2.0
                          for s, v in zip(sample_shape, sample_sp))
        initial_transform = RigidTransform(
            rz_deg=rz, ry_deg=ry, rx_deg=rx,
            tz_um=0.0, ty_um=0.0, tx_um=0.0,
            center_um=center_um,
        ).to_dict()
        typer.echo(
            f"orientation: sample {sample_orientation!r} → atlas "
            f"{ccf.orientation!r}  ⇒  initial rotation "
            f"(rz={rz:+.1f}, ry={ry:+.1f}, rx={rx:+.1f}) deg"
        )

    s = State(
        sample_zarr=str(zarr_path.resolve()),
        sample_level=level,
        sample_voxel_um=vox_list,
        sample_channel=channel,
        atlas_name=atlas,
        sample_orientation=sample_orientation,
        transform=initial_transform,
    )
    s.add_history(
        "init",
        f"level={level} voxel_um={vox_list} orientation={sample_orientation}",
    )
    save_state(s, state)
    typer.echo(f"wrote {state}")
    typer.echo(ms.summary())
    typer.echo(f"\nNext: vol2atlas prealign {state}")


@app.command()
def prealign(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    level: Optional[int] = typer.Option(
        None, "--level",
        help="Pyramid level to load. Default = state.sample_level."),
    preview_size: int = typer.Option(
        192, "--preview-size",
        help="Cube root of max preview voxel budget. Default 192."),
    orientation: Optional[str] = typer.Option(
        None, "--orientation",
        help="3-letter BrainGlobe orientation code for the SAMPLE "
             "(e.g. 'pir', 'psl'). Overrides any existing transform "
             "with the rotation that maps sample → atlas orientation, "
             "so napari opens with the sample roughly aligned. Other "
             "state fields (ccf_crop_bbox, landmarks, history, affine) "
             "are NOT touched. Re-launch prealign with a different "
             "code to iterate. Sample must be right-handed; refused "
             "if it requires a flip."),
):
    """3D rough prealign (translation/rotation sliders) + CCF crop ROI."""
    from .steps.step1_prealign import run
    run(state, level=level, preview_size=preview_size,
        orientation=orientation)


@app.command()
def refine(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    level: Optional[int] = typer.Option(
        None, "--level",
        help="Pyramid level to load. Default = state.sample_level. "
             "Use a finer level (lower number) for higher-resolution preview."),
    preview_size: int = typer.Option(
        192, "--preview-size",
        help="Cube root of max preview voxel budget. Default 192 "
             "(=7 M voxels, fast). Higher = sharper preview, slower."),
):
    """Fine refinement in axial / coronal / sagittal views with tight slider ranges."""
    from .steps.step2_refine import run
    run(state, level=level, preview_size=preview_size)


@app.command()
def landmarks(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    level: Optional[int] = typer.Option(
        None, "--level",
        help="Pyramid level to load for landmark picking. Default = "
             "state.sample_level. Use a finer level (lower number) "
             "for higher-resolution preview."),
    preview_size: int = typer.Option(
        192, "--preview-size",
        help="Cube root of max preview voxel budget. Default 192 "
             "(=7 M voxels, fast). Raise for higher-res preview at "
             "the cost of slower rendering. e.g. 360 = ~47 M voxels."),
):
    """Pick landmark pairs on sample + CCF; rigid Procrustes OR
    full-affine least-squares fit."""
    from .steps.step3_landmarks import run
    run(state, level=level, preview_size=preview_size)


@app.command()
def mi(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    level: Optional[int] = typer.Option(
        None, "--level",
        help="Pyramid level to register on. Default = state.sample_level."),
    mask: bool = typer.Option(
        False, "--mask",
        help="Enable Otsu brain masks (default: off — masks tend to "
             "trap MI on partial / cross-modality data)."),
    affine: bool = typer.Option(
        False, "--affine",
        help="Run AFFINE (12-DOF: rigid + scale + shear) instead of "
             "rigid. Result saved to state.affine and composed on top "
             "of state.transform at export/alignFull time."),
    shape: bool = typer.Option(
        False, "--shape",
        help="Register the SHAPE (signed distance transforms of brain "
             "masks) instead of raw intensities."),
    mask_percentile: float = typer.Option(
        50.0, "--mask-percentile",
        help="Percentile of nonzero SAMPLE voxels for the brain mask. "
             "Higher = tighter (only brightest tissue). For µCT 50-80, "
             "for fluorescence 10-30."),
    mask_threshold: Optional[float] = typer.Option(
        None, "--mask-threshold",
        help="Absolute sample-intensity threshold; overrides "
             "--mask-percentile."),
    ccf_mask_percentile: float = typer.Option(
        60.0, "--ccf-mask-percentile",
        help="Percentile of nonzero CCF voxels for the CCF brain mask. "
             "Default 60 (good for the Allen Nissl template)."),
    ccf_mask_threshold: Optional[float] = typer.Option(
        None, "--ccf-mask-threshold",
        help="Absolute CCF-intensity threshold; overrides "
             "--ccf-mask-percentile."),
    check_mask: bool = typer.Option(
        False, "--check-mask",
        help="Show the brain mask in napari and exit. Use to tune "
             "--mask-percentile / --mask-threshold without running "
             "the registration."),
    skip_view: bool = typer.Option(
        False, "--skip-view",
        help="Headless: save without the napari Accept/Reject preview."),
):
    """Automated rigid (or affine) refinement via Mutual Information.

    Default: Rigid + Mattes MI on raw intensities → state.transform.
    --affine: Affine (12-DOF) → state.affine (composed on top of rigid).
    --shape:  register signed distance transforms of brain masks
              instead of intensities — cross-modality safe.
    Requires antspyx (`pip install antspyx` or `pip install -e .\\[ants]`)."""
    from .steps.step_mi import run
    run(state, level=level, mask=mask, skip_view=skip_view,
        affine=affine, shape=shape,
        mask_percentile=mask_percentile,
        mask_threshold=mask_threshold,
        ccf_mask_percentile=ccf_mask_percentile,
        ccf_mask_threshold=ccf_mask_threshold,
        check_mask=check_mask)


@app.command()
def export(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    out_dir: Path = typer.Option(Path("out/export"), "--out", "-o"),
    level: Optional[int] = typer.Option(
        None, "--level", help="Pyramid level. Default = state's sample_level."),
    tps: bool = typer.Option(
        False, "--tps",
        help="Add a thin-plate-spline correction ON TOP of the rigid "
             "transform (needs ≥4 landmark pairs from `landmarks`). Far from "
             "landmarks the correction vanishes and the rigid result is "
             "preserved."),
    tps_smoothing: float = typer.Option(
        0.0, "--tps-smoothing",
        help="TPS smoothing parameter (0 = exact landmark interpolation). "
             "Increase (e.g. 1.0, 10.0) if landmarks are noisy or the "
             "TPS overshoots between them."),
    skip_view: bool = typer.Option(False, "--skip-view"),
    write_transform: bool = typer.Option(
        False, "--write-transform",
        help="Also emit the registration transform as standalone files: "
             "transform.mat (ITK GenericAffine, ANTs / Slicer readable), "
             "transform.json (OME-NGFF coordinateTransformations), and "
             "landmarks.csv (BigWarp-format, if any). Lets the transform "
             "travel independently of state.json. Reapply with "
             "`vol2atlas apply-transform`."),
):
    """Export sample+atlas to NIfTI and multiscale OME-Zarr through the
    SAME code path as the WYSIWYG resample from `landmarks`. No brainreg, no ANTs, no
    orientation tricks — symmetric save/load so atlas and sample overlay
    the same way you saw in `landmarks`."""
    from .steps.step_export import run
    run(state, out_dir, level=level, tps=tps,
        tps_smoothing=tps_smoothing, skip_view=skip_view,
        write_transform=write_transform)


@app.command("apply-transform")
def apply_transform(
    transform: Path = typer.Argument(...,
        help="Either transform.mat (ITK GenericAffine) OR transform.json "
             "(OME-NGFF) produced by `export --write-transform`."),
    input_zarr: Path = typer.Argument(...,
        help="OME-Zarr volume to warp (sample frame)."),
    out_zarr: Path = typer.Option(..., "--out", "-o",
        help="Output OME-Zarr path (will be multiscale)."),
    level: int = typer.Option(
        0, "--level",
        help="Input pyramid level. Default 0 = full resolution."),
    channel: int = typer.Option(0, "--channel", "-c"),
    out_voxel_um: Optional[str] = typer.Option(
        None, "--out-voxel-um",
        help="Output voxel size in µm, comma-separated z,y,x. Defaults to "
             "the voxel size baked into the transform.json (if .json) or "
             "25 µm (if .mat — no voxel info in ITK format)."),
    out_shape: Optional[str] = typer.Option(
        None, "--out-shape",
        help="Output shape in voxels, comma-separated z,y,x. Defaults to "
             "the input shape transformed via the matrix bounding box."),
):
    """Apply a saved transform to another volume.

    Reads the input OME-Zarr at `level`, applies the affine, writes
    multiscale OME-Zarr output. Use for re-warping a different channel
    or a different acquisition with the same registration."""
    from .io import open_multiscale
    from .transform_io import read_itk_mat, read_ngff_transform
    import numpy as np
    from scipy.ndimage import affine_transform as scp_aff

    tp = str(transform).lower()
    local_refs = []
    if tp.endswith(".mat"):
        M_smp2atlas = read_itk_mat(transform)
        atlas_voxel = None
    elif tp.endswith(".json"):
        M_smp2atlas = read_ngff_transform(transform)
        try:
            import json as _json
            payload = _json.loads(transform.read_text())
            atlas_voxel = tuple(payload["output"]["voxel_um"])
            from .local_refinement import LocalRefinement
            local_refs = [LocalRefinement.from_dict(d)
                          for d in payload.get("local_refinements", [])]
            if local_refs:
                typer.echo(f"transform carries {len(local_refs)} "
                           f"local refinement(s): "
                           f"{[L.name for L in local_refs]}")
        except Exception:
            atlas_voxel = None
    else:
        typer.secho(
            f"unknown transform format {transform.suffix}; "
            f"use .mat or .json", fg="red", err=True)
        raise typer.Exit(2)

    if out_voxel_um:
        parts = [float(p) for p in out_voxel_um.split(",")]
        if len(parts) == 1:
            parts = parts * 3
        out_vox = tuple(parts)
    elif atlas_voxel:
        out_vox = atlas_voxel
    else:
        typer.secho(
            "no --out-voxel-um given and the transform file has no "
            "voxel info; specify --out-voxel-um", fg="red", err=True)
        raise typer.Exit(2)

    ms = open_multiscale(input_zarr)
    arr = ms.level(level)
    if "c" in ms.axes:
        arr = arr[channel]
    sample_vox = ms.spacing(level)
    sample_np = np.ascontiguousarray(arr.compute()).astype(np.float32)
    typer.echo(f"input: {sample_np.shape} @ {sample_vox} µm")

    if out_shape:
        target_shape = tuple(int(x) for x in out_shape.split(","))
    else:
        # Bounding box of warped sample corners → output shape (in
        # output voxel units).
        Z, Y, X = sample_np.shape
        corners_smp_um = np.array([
            [0, 0, 0],
            [Z * sample_vox[0], 0, 0],
            [0, Y * sample_vox[1], 0],
            [0, 0, X * sample_vox[2]],
            [Z * sample_vox[0], Y * sample_vox[1], 0],
            [Z * sample_vox[0], 0, X * sample_vox[2]],
            [0, Y * sample_vox[1], X * sample_vox[2]],
            [Z * sample_vox[0], Y * sample_vox[1], X * sample_vox[2]],
        ])
        corners_atlas_um = (M_smp2atlas[:3, :3] @ corners_smp_um.T).T + \
                            M_smp2atlas[:3, 3]
        mn = corners_atlas_um.min(0)
        mx = corners_atlas_um.max(0)
        target_shape = tuple(int(np.ceil((mx[i] - mn[i]) / out_vox[i]))
                              for i in range(3))

    typer.echo(f"output: {target_shape} @ {out_vox} µm")

    if local_refs:
        # Apply blended global+local via per-voxel map_coordinates.
        from scipy.ndimage import map_coordinates
        from .local_refinement import blended_inverse_sample_coords
        warped = np.empty(target_shape, dtype=np.float32)
        z_step = max(1, min(64, target_shape[0]))
        for z0 in range(0, target_shape[0], z_step):
            z1 = min(z0 + z_step, target_shape[0])
            zz, yy, xx = np.meshgrid(
                np.arange(z0, z1, dtype=np.float32),
                np.arange(target_shape[1], dtype=np.float32),
                np.arange(target_shape[2], dtype=np.float32),
                indexing="ij",
            )
            coords = np.stack([zz, yy, xx], axis=0)
            sv = blended_inverse_sample_coords(
                coords, M_smp2atlas, sample_vox, out_vox,
                crop_origin_um=(0.0, 0.0, 0.0),
                local_refinements=local_refs,
            )
            warped[z0:z1] = map_coordinates(
                sample_np, sv, order=1, mode="constant", cval=0.0,
            ).astype(np.float32)
    else:
        # Build voxel-space matrix: sample_vox ← atlas_vox (inverse of
        # what we want, because scipy uses output→input mapping).
        S_in_smp = np.diag([sample_vox[0], sample_vox[1], sample_vox[2], 1.0])
        S_out_atlas = np.diag([1.0/out_vox[0], 1.0/out_vox[1],
                                1.0/out_vox[2], 1.0])
        M_vox = S_out_atlas @ M_smp2atlas @ S_in_smp
        Minv = np.linalg.inv(M_vox)
        warped = scp_aff(sample_np, Minv[:3, :3], offset=Minv[:3, 3],
                          output_shape=target_shape,
                          order=1, mode="constant", cval=0.0)
    typer.echo(f"warped: {warped.shape}, range "
               f"[{warped.min():.3f}, {warped.max():.3f}]")

    # Write as multiscale OME-Zarr (reuse step_export's writer).
    from .steps.step_export import _save_ngff
    _save_ngff(warped, out_zarr, out_vox)
    typer.echo(f"wrote {out_zarr}")


@app.command()
def ants(
    export_dir: Path = typer.Argument(...,
        help="Output dir of a previous `vol2atlas export` run."),
    out_dir: Path = typer.Option(Path("out/ants"), "--out", "-o"),
    transform_type: str = typer.Option(
        "SyNOnly", "--transform",
        help="ANTs transform: 'SyNOnly' (recommended), 'SyN', 'Affine', etc. "
             "SyN re-runs rigid+affine and overwrites the prealignment — "
             "stick with SyNOnly unless you have a reason."),
    reg_iter_high:   int = typer.Option(10, "--iter-high"),
    reg_iter_mid:    int = typer.Option(5,  "--iter-mid"),
    reg_iter_low:    int = typer.Option(0,  "--iter-low"),
    flow_sigma:    float = typer.Option(6.0, "--flow-sigma"),
    total_sigma:   float = typer.Option(2.0, "--total-sigma"),
    state: Path  = typer.Option(Path("state.json"), "--state", "-s",
                                 help="State file to log this run into."),
    skip_view:     bool = typer.Option(False, "--skip-view"),
):
    """Refine a `vol2atlas export` output with ANTs SyNOnly + brain masks."""
    from .steps.refine_ants import run
    run(export_dir, out_dir,
        transform_type=transform_type,
        reg_iterations=(reg_iter_high, reg_iter_mid, reg_iter_low),
        flow_sigma=flow_sigma, total_sigma=total_sigma,
        state_path=state, skip_view=skip_view)


@app.command()
def brainreg(
    export_dir: Path = typer.Argument(...,
        help="Output dir of a previous `vol2atlas export` run."),
    out_dir: Path = typer.Option(Path("out/brainreg"), "--out", "-o"),
    atlas: str   = typer.Option("allen_mouse_25um", "--atlas"),
    orientation: str = typer.Option(
        "asr", "--orientation",
        help="brainglobe orientation of the prealigned input. For an "
             "atlas-native export this matches the atlas (asr for "
             "allen_mouse_25um)."),
    brain_geometry: str = typer.Option(
        "full", "--brain-geometry",
        help="brainreg --brain_geometry: full / hemisphere_l / hemisphere_r. "
             "Partial samples do NOT match any of these well."),
    state: Path  = typer.Option(Path("state.json"), "--state", "-s"),
    skip_view: bool = typer.Option(False, "--skip-view"),
):
    """Refine a `vol2atlas export` output with BrainGlobe brainreg.

    NOTE: brainreg assumes a full brain or a complete hemisphere; on a
    partial sample it will distort to fill the missing region."""
    from .steps.refine_brainreg import run
    run(export_dir, out_dir, atlas_name=atlas, orientation=orientation,
        brain_geometry=brain_geometry, state_path=state, skip_view=skip_view)


@app.command()
def elastix(
    export_dir: Path = typer.Argument(...,
        help="Output dir of a previous `vol2atlas export` run."),
    out_dir: Path = typer.Option(Path("out/elastix"), "--out", "-o"),
    grid_voxels: int = typer.Option(
        16, "--grid",
        help="B-spline grid spacing in voxels (Humbel et al. used 16)."),
    bending_energy: float = typer.Option(
        1000.0, "--bending",
        help="Bending-energy penalty weight (Humbel et al. used 1000). "
             "Higher = smoother (more rigid-like) warp."),
    iterations: int = typer.Option(1000, "--iter"),
    resolutions: int = typer.Option(
        4, "--resolutions",
        help="Number of multi-resolution pyramid levels."),
    metric: str = typer.Option(
        "AdvancedMattesMutualInformation", "--metric",
        help="elastix image similarity metric. MI is right for cross-modality "
             "(µCT vs LSFM/Nissl atlas); switch to "
             "AdvancedNormalizedCorrelation only for same-modality."),
    no_masks: bool = typer.Option(False, "--no-masks",
        help="Disable Otsu brain masks (default: masks ON)."),
    state: Path = typer.Option(Path("state.json"), "--state", "-s"),
    skip_view: bool = typer.Option(False, "--skip-view"),
):
    """Refine a `vol2atlas export` output with elastix B-spline FFD.

    Defaults match Humbel et al. 2024 (µCT mouse brain → CCF):
    grid 16 voxels, bending-energy 1000, 4-level pyramid, MI metric."""
    from .steps.refine_elastix import run
    run(export_dir, out_dir,
        grid_voxels=grid_voxels, bending_energy=bending_energy,
        iterations=iterations, resolutions=resolutions, metric=metric,
        use_masks=not no_masks,
        state_path=state, skip_view=skip_view)


@app.command("alignFull")
def align_full(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    out: Path = typer.Option(
        Path("out/full_in_ccf.zarr"), "--out", "-o",
        help="Output multiscale OME-Zarr (overwritten)."),
    levels: Optional[str] = typer.Option(
        None, "--levels",
        help="Comma-separated input pyramid levels to warp "
             "(e.g. '1,2,3'). Default: all available levels of the input "
             "zarr. Note level 0 of a typical µCT zarr is ~TB; consider "
             "starting from level 1 or 2."),
    chunks: Optional[str] = typer.Option(
        None, "--chunks",
        help="Output zarr chunk size as z,y,x. Default: mirror the input "
             "zarr's chunks at each level (aligns reads = much faster)."),
):
    """Apply state.json's rigid transform to the FULL OME-Zarr chunkwise.

    For each requested input pyramid level, lazily reads sample's level k,
    applies the inverse rigid via dask_image.ndinterp.affine_transform,
    writes the warped result as level k of a new multiscale OME-Zarr in
    CCF coordinates. RAM stays bounded by chunk size — does NOT load any
    level fully.

    Currently rigid-only. ANTs/elastix deformable composition: TODO."""
    from .steps.align_full import run
    lvls = [int(x) for x in levels.split(",")] if levels else None
    chk: Optional[tuple[int, int, int]] = None
    if chunks:
        parts = tuple(int(x) for x in chunks.split(","))
        if len(parts) != 3:
            raise typer.BadParameter("--chunks needs 3 ints, e.g. 64,64,64")
        chk = parts
    run(state, out, levels=lvls, output_chunks=chk)


@app.command()
def info(zarr_path: Path = typer.Argument(...)):
    """Print the OME-Zarr pyramid structure."""
    from .io import open_multiscale
    try:
        ms = open_multiscale(zarr_path)
    except Exception as e:
        typer.secho(f"could not open zarr: {e}", fg="red", err=True); raise typer.Exit(2)
    typer.echo(ms.summary())


@app.command("landmarks-export")
def landmarks_export(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    out: Path = typer.Option(..., "--out", "-o",
        help="Output file. Format inferred from extension: "
             ".csv = BigWarp-format (sample/CCF µm pairs, "
             "x,y,z columns); .json = vol2atlas-native "
             "(state.landmarks dict)."),
):
    """Export the landmark set from state.json to a file.

    CSV format: `Pt-i,True,sx,sy,sz,cx,cy,cz` per row (BigWarp's
    convention). NOTE the x,y,z column order in the file is the
    REVERSE of vol2atlas's internal (z,y,x) tuples — the swap is
    handled on write.
    """
    from .state import load as load_state
    from .transform_io import write_landmarks_csv
    import json as _json
    s = load_state(state)
    smp = (s.landmarks or {}).get("sample_um", [])
    ccf = (s.landmarks or {}).get("ccf_um", [])
    n = min(len(smp), len(ccf))
    if n == 0:
        typer.secho("no landmarks in state", fg="yellow")
        raise typer.Exit(0)
    suf = out.suffix.lower()
    if suf == ".csv":
        write_landmarks_csv(smp[:n], ccf[:n], out)
    elif suf == ".json":
        out.write_text(_json.dumps({
            "sample_um": [list(p) for p in smp[:n]],
            "ccf_um":    [list(p) for p in ccf[:n]],
        }, indent=2))
    else:
        typer.secho(
            f"unknown extension {out.suffix!r}; use .csv or .json",
            fg="red", err=True); raise typer.Exit(2)
    typer.echo(f"exported {n} landmark pairs → {out}")


@app.command("landmarks-import")
def landmarks_import(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    src: Path = typer.Argument(...,
        help="Landmarks file (.csv BigWarp-format or .json vol2atlas)."),
    mode: str = typer.Option(
        "replace", "--mode",
        help="'replace' overwrites current state.landmarks; 'append' "
             "adds new pairs to the existing set."),
):
    """Import landmarks from an external file into state.json.

    The imported pairs are stored in PHYSICAL µm — they're invariant
    to atlas resolution (you can `change-atlas` later). Sample-side
    coordinates are in the SAMPLE raw µm frame; CCF-side in CCF µm.
    """
    from .state import load as load_state, save as save_state
    from .transform_io import read_landmarks_csv
    import json as _json
    s = load_state(state)
    suf = src.suffix.lower()
    if suf == ".csv":
        smp, ccf = read_landmarks_csv(src)
    elif suf == ".json":
        d = _json.loads(src.read_text())
        smp = [tuple(p) for p in d.get("sample_um", [])]
        ccf = [tuple(p) for p in d.get("ccf_um", [])]
    else:
        typer.secho(
            f"unknown extension {src.suffix!r}; use .csv or .json",
            fg="red", err=True); raise typer.Exit(2)
    n = min(len(smp), len(ccf))
    if n == 0:
        typer.secho(f"no usable pairs in {src}", fg="red", err=True)
        raise typer.Exit(2)
    if mode == "replace":
        s.landmarks = {
            "sample_um": [list(p) for p in smp[:n]],
            "ccf_um":    [list(p) for p in ccf[:n]],
        }
        typer.echo(f"replaced landmark set with {n} pairs from {src}")
    elif mode == "append":
        s.landmarks = s.landmarks or {"sample_um": [], "ccf_um": []}
        s.landmarks["sample_um"].extend(list(p) for p in smp[:n])
        s.landmarks["ccf_um"].extend(list(p)    for p in ccf[:n])
        typer.echo(
            f"appended {n} pairs from {src}; total now "
            f"{len(s.landmarks['sample_um'])}")
    else:
        typer.secho(f"--mode must be 'replace' or 'append', got {mode!r}",
                    fg="red", err=True); raise typer.Exit(2)
    s.add_history("landmarks-import",
                   f"src={src} mode={mode} n_pairs={n}")
    save_state(s, state)


@app.command("add-local-refinement")
def add_local_refinement(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    name: str = typer.Option(..., "--name",
        help="Short label, e.g. 'left_lobe_fix'."),
    landmarks: str = typer.Option(..., "--landmarks",
        help="Comma-separated 0-based indices into state.landmarks to "
             "USE for the local fit, e.g. '5,6,7,8,9'. ≥4 indices "
             "required (closed-form affine)."),
    falloff_um: float = typer.Option(
        300.0, "--falloff-um",
        help="Sigmoid falloff width at the mask boundary, in µm. "
             "Higher = smoother blend, more global → local interpolation "
             "zone. Default 300 µm (~one cortical thickness)."),
    radius_pad_um: float = typer.Option(
        200.0, "--radius-pad-um",
        help="Extra margin on the sphere radius, in µm, beyond the "
             "furthest landmark from the centroid. Default 200 µm."),
):
    """Add a local affine refinement that only affects the region near
    a subset of your landmarks (spherical mask with sigmoid falloff).

    The local affine is fit ONLY from the selected landmarks. The mask
    sphere is centered on their centroid and sized to enclose them all
    plus `--radius-pad-um`. Outside the sphere + falloff zone the
    output is unchanged from the global transform.

    Use case: after global rigid+affine, you notice one anatomical
    region (e.g., one lobe of the cortex) is slightly off. Pick extra
    landmarks ONLY in that region via the existing `landmarks` step,
    note their indices in the terminal output, then run this command
    to add a localized correction.
    """
    import numpy as np
    from .state import load as load_state, save as save_state
    from .local_refinement import fit_from_landmarks, LocalRefinement
    s = load_state(state)
    if not s.landmarks or not s.landmarks.get("sample_um"):
        typer.secho("state has no landmarks; add some via "
                    "`vol2atlas landmarks` first", fg="red", err=True)
        raise typer.Exit(2)
    try:
        idx = [int(x) for x in landmarks.split(",")]
    except ValueError as e:
        typer.secho(f"could not parse --landmarks: {e}",
                    fg="red", err=True); raise typer.Exit(2)
    n_total = min(len(s.landmarks["sample_um"]),
                   len(s.landmarks["ccf_um"]))
    bad = [i for i in idx if not (0 <= i < n_total)]
    if bad:
        typer.secho(f"bad landmark indices {bad} (have {n_total} "
                    f"valid pairs)", fg="red", err=True)
        raise typer.Exit(2)
    sample_pts = np.asarray([s.landmarks["sample_um"][i] for i in idx],
                              dtype=float)
    ccf_pts    = np.asarray([s.landmarks["ccf_um"][i]    for i in idx],
                              dtype=float)
    existing_names = {r["name"] for r in s.local_refinements}
    if name in existing_names:
        typer.secho(f"local refinement {name!r} already exists; pick "
                    f"another name or remove the old one first.",
                    fg="red", err=True); raise typer.Exit(2)
    try:
        L = fit_from_landmarks(
            sample_pts, ccf_pts,
            name=name, falloff_um=falloff_um,
            radius_pad_um=radius_pad_um,
            landmark_indices=idx,
        )
    except ValueError as e:
        typer.secho(str(e), fg="red", err=True); raise typer.Exit(2)
    s.local_refinements.append(L.to_dict())
    s.add_history(
        "add-local-refinement",
        f"name={name} n_lms={len(idx)} indices={idx} "
        f"center_um={L.center_um} radius_um={L.radius_um:.1f} "
        f"falloff_um={L.falloff_um}")
    save_state(s, state)
    typer.echo(f"added local refinement {name!r}: "
               f"{len(idx)} landmarks, center {L.center_um}, "
               f"radius {L.radius_um:.0f} µm, falloff {L.falloff_um} µm")
    sv = np.linalg.svd(L.affine[:3, :3], compute_uv=False)
    typer.echo(f"  local affine singular values: "
               f"{sv[0]:.3f} {sv[1]:.3f} {sv[2]:.3f} (1.0 = no scale)")


@app.command("list-local-refinements")
def list_local_refinements(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """List all local refinements stored in state.json."""
    from .state import load as load_state
    s = load_state(state)
    if not s.local_refinements:
        typer.echo("(no local refinements)")
        return
    for i, r in enumerate(s.local_refinements):
        c = r["center_um"]
        typer.echo(
            f"[{i}] {r['name']}: "
            f"center=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) µm "
            f"radius={r['radius_um']:.0f} µm "
            f"falloff={r['falloff_um']:.0f} µm "
            f"landmarks={r.get('landmark_indices', [])}")


@app.command("remove-local-refinement")
def remove_local_refinement(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    name: str = typer.Option(..., "--name",
        help="Name of the local refinement to remove."),
):
    """Remove a named local refinement from state.json."""
    from .state import load as load_state, save as save_state
    s = load_state(state)
    before = len(s.local_refinements)
    s.local_refinements = [r for r in s.local_refinements
                            if r["name"] != name]
    after = len(s.local_refinements)
    if before == after:
        typer.secho(f"no local refinement named {name!r}",
                    fg="red", err=True); raise typer.Exit(2)
    s.add_history("remove-local-refinement", f"name={name}")
    save_state(s, state)
    typer.echo(f"removed {name!r} ({before} → {after} refinements)")


@app.command("change-atlas")
def change_atlas(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
    to: str = typer.Option(..., "--to",
        help="New BrainGlobe atlas name (e.g. allen_mouse_10um)."),
):
    """Switch the atlas in state.json (e.g. 25µm → 10µm) WITHOUT losing
    landmarks / transform / affine.

    Landmarks (sample_um, ccf_um), state.transform, and state.affine are
    all stored in physical µm and are resolution-independent — they stay
    valid as-is. Only ccf_crop_bbox is stored in CCF VOXEL indices and
    must be rescaled by the new/old voxel-size ratio.
    """
    from .state import load as load_state, save as save_state
    from .atlas import load_ccf
    s = load_state(state)
    if to == s.atlas_name:
        typer.echo(f"atlas already = {to}; nothing to do.")
        return
    try:
        old_ccf = load_ccf(s.atlas_name)
        new_ccf = load_ccf(to)
    except Exception as e:
        typer.secho(f"could not load atlas: {e}", fg="red", err=True)
        raise typer.Exit(2)
    old_vox = old_ccf.voxel_um
    new_vox = new_ccf.voxel_um
    if s.ccf_crop_bbox is not None:
        rescaled = {}
        for axis, ov, nv in zip(("z", "y", "x"), old_vox, new_vox):
            ratio = ov / nv
            lo, hi = s.ccf_crop_bbox[axis]
            rescaled[axis] = [int(round(lo * ratio)),
                              int(round(hi * ratio))]
        typer.echo(
            f"ccf_crop_bbox rescaled by {old_vox} → {new_vox} µm:"
        )
        for axis in ("z", "y", "x"):
            typer.echo(
                f"  {axis}: {s.ccf_crop_bbox[axis]} → {rescaled[axis]}")
        s.ccf_crop_bbox = rescaled
    else:
        typer.echo("no ccf_crop_bbox to rescale.")
    s.atlas_name = to
    s.add_history("change-atlas",
                  f"{old_ccf.name} ({old_vox}) → {to} ({new_vox})")
    save_state(s, state)
    typer.echo(f"saved {state}: atlas → {to}")
    typer.echo(
        "Landmarks, transform, and affine are unchanged (physical µm).\n"
        "Re-run any step — it'll use the new atlas resolution."
    )


@app.command("list-atlases")
def list_atlases(
    species: str = typer.Option(
        "mouse", "--species",
        help="Filter by species name substring (case-insensitive). "
             "Default 'mouse'; pass empty string for all atlases."),
    downloaded_only: bool = typer.Option(
        False, "--downloaded",
        help="Show only atlases already in ~/.brainglobe/ cache."),
):
    """List BrainGlobe atlases you can pass to `init --atlas <name>`.

    Resolutions for Allen Mouse CCFv3: 10, 25 (default), 50, 100 µm.
    Other mouse atlases include Kim Lab, Osten, Princeton, Gubra
    (LSFM & MRI), BlueBrain CCFv3-augmented, DeMBA developmental, etc.
    """
    try:
        from brainglobe_atlasapi.list_atlases import (
            get_all_atlases_lastversions,
            get_downloaded_atlases,
        )
    except ImportError as e:
        typer.secho(f"brainglobe-atlasapi not installed: {e}",
                    fg="red", err=True); raise typer.Exit(2)
    try:
        if downloaded_only:
            entries = [(name, "") for name in get_downloaded_atlases()]
        else:
            data = get_all_atlases_lastversions()
            # data is {name: version}; for richer info we'd need the
            # remote metadata, but name + version is enough for picking.
            entries = sorted(data.items())
    except Exception as e:
        typer.secho(f"failed to list atlases: {e}", fg="red", err=True)
        raise typer.Exit(2)
    needle = species.lower().strip()
    if needle:
        entries = [(n, v) for n, v in entries if needle in n.lower()]
    if not entries:
        typer.echo(f"(no atlases match species filter {species!r})")
        return
    typer.echo(f"{'atlas_name':<48}  version")
    typer.echo("-" * 60)
    for name, version in entries:
        typer.echo(f"{name:<48}  {version}")
    typer.echo()
    typer.echo(
        f"Pass any name to: vol2atlas init <zarr> --atlas <atlas_name>\n"
        f"First load downloads + caches to ~/.brainglobe/ (one-time, ~MB)."
    )


@app.command("help-all")
def help_all():
    """Show full --help (args, options, defaults) for every subcommand."""
    import click
    group = typer.main.get_command(app)
    for name, cmd in group.commands.items():
        if name == "help-all":
            continue
        sub_ctx = click.Context(cmd, info_name=f"vol2atlas {name}")
        typer.echo("=" * 72)
        typer.echo(cmd.get_help(sub_ctx))
        typer.echo("")


if __name__ == "__main__":
    app()
