#!/usr/bin/env python3
"""Open a registered (atlas-aligned) volume alongside the Allen CCF in napari.

Shows three layers in the same voxel grid:
  - CCF reference (gray)
  - your registered sample (magenta)
  - CCF region annotations (labels, hidden by default — toggle on for IDs)

Optionally adds a difference image to highlight misalignment.

Usage:
    python compare_with_ccf.py out/sample_in_ccf.zarr
    python compare_with_ccf.py out/sample_in_ccf.zarr --atlas allen_mouse_25um --diff
    python compare_with_ccf.py out/sample_in_ccf.zarr --grid     # side-by-side instead of overlay
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("registered", type=Path,
                    help="Path to registered volume. Accepts .zarr (multiscale "
                         "OME-Zarr) or .tiff/.tif (single-resolution).")
    ap.add_argument("--atlas", default="allen_mouse_25um",
                    help="BrainGlobe atlas name (default: allen_mouse_25um).")
    ap.add_argument("--level", type=int, default=0,
                    help="If reading OME-Zarr, which pyramid level (default: 0).")
    ap.add_argument("--grid", action="store_true",
                    help="Show layers side-by-side instead of overlaid.")
    ap.add_argument("--diff", action="store_true",
                    help="Add a |z-scored difference| layer to highlight misalignment.")
    ap.add_argument("--ndisplay", type=int, default=2,
                    help="2 = slice view (default), 3 = volume MIP.")
    args = ap.parse_args()

    import napari
    import numpy as np
    from brainglobe_atlasapi import BrainGlobeAtlas

    # ---- load registered volume ------------------------------------------
    p = args.registered
    if p.suffix == ".zarr" or (p.is_dir() and (p / ".zgroup").exists()):
        import zarr
        g = zarr.open(str(p), mode="r")
        sample = np.asarray(g[str(args.level)])
    elif p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        sample = tifffile.imread(str(p))
    else:
        sys.exit(f"unsupported input: {p} (expected .zarr or .tiff)")
    print(f"sample: shape={sample.shape}  dtype={sample.dtype}  "
          f"size={sample.nbytes / 1e6:.1f} MB")

    # ---- load CCF --------------------------------------------------------
    atlas = BrainGlobeAtlas(args.atlas)
    ref = atlas.reference
    ann = atlas.annotation
    print(f"CCF:    shape={ref.shape}  voxel={atlas.resolution} µm")

    if sample.shape != ref.shape:
        print(f"WARNING: shape mismatch — sample {sample.shape} vs "
              f"CCF {ref.shape}. Registration may have failed, or the volume "
              f"isn't actually in atlas space.")

    # ---- viewer ----------------------------------------------------------
    v = napari.Viewer(ndisplay=args.ndisplay)

    v.add_image(ref, name="CCF reference",
                scale=atlas.resolution,
                colormap="gray", blending="additive", opacity=0.6)
    v.add_image(sample, name="sample (registered)",
                scale=atlas.resolution,
                colormap="magenta", blending="additive", opacity=0.6)
    v.add_labels(ann, name="CCF regions", scale=atlas.resolution,
                 opacity=0.35, visible=False)

    if args.diff and sample.shape == ref.shape:
        s_n = (sample.astype(float) - sample.mean()) / (sample.std() + 1e-9)
        r_n = (ref.astype(float)    - ref.mean())    / (ref.std()    + 1e-9)
        v.add_image(np.abs(s_n - r_n), name="|diff|",
                    scale=atlas.resolution,
                    colormap="inferno", blending="additive",
                    opacity=0.5, visible=False)

    if args.grid:
        v.grid.enabled = True

    print("napari open. tips:")
    print("  - scroll z (bottom slider) to walk through slices")
    print("  - toggle 'CCF regions' on; click a voxel to see region ID in status bar")
    print("  - toggle '|diff|' on (if --diff) for misalignment heatmap")
    print("  - press Ctrl+G to switch grid <-> overlay")
    napari.run()


if __name__ == "__main__":
    main()
