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
):
    """Export sample+atlas to NIfTI and multiscale OME-Zarr through the
    SAME code path as the WYSIWYG resample from `landmarks`. No brainreg, no ANTs, no
    orientation tricks — symmetric save/load so atlas and sample overlay
    the same way you saw in `landmarks`."""
    from .steps.step_export import run
    run(state, out_dir, level=level, tps=tps,
        tps_smoothing=tps_smoothing, skip_view=skip_view)


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
