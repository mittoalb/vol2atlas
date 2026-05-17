"""Step-based CLI for the zrot → brainreg workflow."""
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
    typer.echo(f"\nNext: zrot step1 {state}")


@app.command()
def step1(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """Step 1: 3D rough prealign (sliders, flips, jog) + CCF crop ROI."""
    from .steps.step1_prealign import run
    run(state)


@app.command()
def step2(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """Step 2: fine refinement in 3 simultaneous orthogonal views."""
    from .steps.step2_refine import run
    run(state)


@app.command()
def step3(
    state: Path = typer.Argument(Path("state.json"), help="State file."),
):
    """Step 3: landmark-based rigid fit (Procrustes)."""
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
             "transform (needs ≥4 landmark pairs from step3). Far from "
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
    SAME code path as step3's working display. No brainreg, no ANTs, no
    orientation tricks — symmetric save/load so atlas and sample overlay
    the same way you saw in step3."""
    from .steps.step_export import run
    run(state, out_dir, level=level, tps=tps,
        tps_smoothing=tps_smoothing, skip_view=skip_view)


@app.command()
def ants(
    export_dir: Path = typer.Argument(...,
        help="Output dir of a previous `zrot export` run."),
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
    """Refine a `zrot export` output with ANTs SyNOnly + brain masks."""
    from .steps.refine_ants import run
    run(export_dir, out_dir,
        transform_type=transform_type,
        reg_iterations=(reg_iter_high, reg_iter_mid, reg_iter_low),
        flow_sigma=flow_sigma, total_sigma=total_sigma,
        state_path=state, skip_view=skip_view)


@app.command()
def brainreg(
    export_dir: Path = typer.Argument(...,
        help="Output dir of a previous `zrot export` run."),
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
    """Refine a `zrot export` output with BrainGlobe brainreg.

    NOTE: brainreg assumes a full brain or a complete hemisphere; on a
    partial sample it will distort to fill the missing region."""
    from .steps.refine_brainreg import run
    run(export_dir, out_dir, atlas_name=atlas, orientation=orientation,
        brain_geometry=brain_geometry, state_path=state, skip_view=skip_view)


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
