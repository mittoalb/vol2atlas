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
    atlas: str = typer.Option("allen_mouse_25um", "--atlas"),
    voxel_um: Optional[str] = typer.Option(
        None, "--voxel-um",
        help="Override source voxel size in µm. Single float or comma-separated z,y,x."),
):
    """Create a new state.json from a zarr path. Does not open napari."""
    from .io import open_multiscale
    from .state import State, save as save_state

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

    s = State(
        sample_zarr=str(zarr_path.resolve()),
        sample_level=level,
        sample_voxel_um=vox_list,
        sample_channel=channel,
        atlas_name=atlas,
    )
    s.add_history("init", f"level={level} voxel_um={vox_list}")
    save_state(s, state)
    typer.echo(f"wrote {state}")
    typer.echo(ms.summary())
    typer.echo(f"\nNext: vol2atlas prealign {state}")


@app.command()
def prealign(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """3D rough prealign (translation/rotation sliders, axis flips) + CCF crop ROI."""
    from .steps.step1_prealign import run
    run(state)


@app.command()
def refine(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """Fine refinement in axial / coronal / sagittal views with tight slider ranges."""
    from .steps.step2_refine import run
    run(state)


@app.command()
def landmarks(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """Pick landmark pairs on sample + CCF; optional Procrustes rigid fit."""
    from .steps.step3_landmarks import run
    run(state)


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


if __name__ == "__main__":
    app()
