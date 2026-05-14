"""Command-line entry points. Hybrid workflow: hand-align here, then brainreg."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False,
                  help="Interactive rigid prealignment of OME-Zarr volumes "
                       "to the Allen CCF. Pair with `brainreg` for refinement.")


@app.command()
def align(
    zarr_path: Path = typer.Argument(..., help="Path to OME-Zarr group."),
    level: int = typer.Option(2, "--level", "-l"),
    channel: Optional[int] = typer.Option(0, "--channel", "-c"),
    atlas: str = typer.Option("allen_mouse_25um", "--atlas", "-a"),
    save_to: Path = typer.Option(Path("transform.json"), "--save", "-s"),
    initial: Optional[Path] = typer.Option(None, "--initial"),
    preview_voxels: int = typer.Option(256 ** 3, "--preview-voxels"),
    ndisplay: int = typer.Option(2, "--ndisplay"),
):
    """Open the interactive aligner and save the resulting rigid transform."""
    from .align import align_interactive
    from .atlas import load_ccf
    from .io import open_multiscale
    from .transform import load_transform

    try:
        ms = open_multiscale(zarr_path)
    except Exception as e:
        typer.secho(f"Could not open zarr: {e}", fg="red", err=True)
        raise typer.Exit(2)
    if not 0 <= level < ms.n_levels():
        typer.secho(
            f"\nLevel {level} not available — pyramid has {ms.n_levels()} "
            f"levels (valid: 0..{ms.n_levels() - 1}).\n",
            fg="red", err=True,
        )
        typer.echo(ms.summary(), err=True)
        raise typer.Exit(2)

    init = load_transform(initial) if initial else None
    ccf = load_ccf(atlas)
    align_interactive(
        sample=ms,
        level=level,
        channel=channel,
        ccf=ccf,
        initial=init,
        save_to=save_to,
        preview_max_voxels=preview_voxels,
        ndisplay=ndisplay,
    )
    typer.echo(f"saved transform to {save_to}")
    typer.echo(f"\nNext: prealign for brainreg:\n"
               f"  python scripts/prealign_for_brainreg.py {zarr_path} "
               f"{save_to} sample_prealigned.nii.gz --atlas {atlas}")


@app.command()
def info(zarr_path: Path = typer.Argument(...)):
    """Show pyramid info for an OME-Zarr group."""
    from .io import open_multiscale
    try:
        ms = open_multiscale(zarr_path)
    except Exception as e:
        typer.secho(f"Could not open zarr: {e}", fg="red", err=True)
        raise typer.Exit(2)
    typer.echo(ms.summary())


if __name__ == "__main__":
    app()
