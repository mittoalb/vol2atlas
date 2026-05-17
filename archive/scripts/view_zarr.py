#!/usr/bin/env python3
"""Open an OME-Zarr volume in napari at a chosen pyramid level.

Examples
--------
    # Default: read level 2, full lazy view
    python view_zarr.py /data/sample.zarr

    # Pick a level explicitly, run in 3D MIP mode
    python view_zarr.py /data/sample.zarr --level 0 --ndisplay 3

    # Specific channel for 4D (c,z,y,x) data
    python view_zarr.py /data/sample.zarr --level 1 --channel 0

    # Decimate by N for a fast preview (good for large levels over slow disks)
    python view_zarr.py /data/sample.zarr --level 0 --stride 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path", type=Path, help="OME-Zarr group path.")
    ap.add_argument("--level", type=int, default=2,
                    help="Pyramid level to display (default: 2).")
    ap.add_argument("--channel", type=int, default=None,
                    help="If the array is 4D (c,z,y,x), pick this channel.")
    ap.add_argument("--ndisplay", type=int, default=2,
                    help="2 = slice view (default), 3 = volume MIP.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Read every Nth voxel (default 1 = no decimation). "
                         "Use 2 or 4 for a fast preview of large levels.")
    ap.add_argument("--name", type=str, default=None,
                    help="Layer name (default: derived from path + level).")
    args = ap.parse_args()

    import dask.array as da
    import napari
    import numpy as np
    import zarr

    root = zarr.open(str(args.zarr_path), mode="r")

    # Pyramid summary
    if "multiscales" in root.attrs:
        ms = root.attrs["multiscales"][0]
        axes = tuple(a["name"] for a in ms["axes"])
        print(f"OME-Zarr: {args.zarr_path}")
        print(f"  axes:   {axes}")
        print(f"  levels: {len(ms['datasets'])}")
        if not 0 <= args.level < len(ms["datasets"]):
            sys.exit(f"  --level {args.level} not in 0..{len(ms['datasets'])-1}")
        ds = ms["datasets"][args.level]
        # Read scale (voxel size) for the chosen level
        scale = next(t["scale"] for t in ds.get("coordinateTransformations", [])
                     if t["type"] == "scale")
        voxel = tuple(float(s) for s in scale[-3:])  # last 3 = spatial
        unit = next((a.get("unit", "") for a in ms["axes"]
                     if a["name"] in ("z", "y", "x")), "")
        print(f"  selected level {args.level}: voxel={voxel} {unit}")
        dataset_path = ds["path"]
    else:
        # Plain zarr group without OME-Zarr metadata
        axes = None
        voxel = (1.0, 1.0, 1.0)
        unit = "voxel"
        dataset_path = str(args.level)

    arr = da.from_zarr(root[dataset_path])
    print(f"  shape:  {arr.shape}  dtype: {arr.dtype}  "
          f"size: {arr.nbytes / 1e9:.2f} GB (lazy)")

    if arr.ndim == 4 and args.channel is not None:
        arr = arr[args.channel]
        print(f"  picked channel {args.channel}: {arr.shape}")

    if args.stride > 1:
        arr = arr[..., ::args.stride, ::args.stride, ::args.stride]
        voxel = tuple(v * args.stride for v in voxel)
        print(f"  strided x{args.stride}: shape {arr.shape}, voxel {voxel} {unit}")

    # Contrast limits from a cheap sample (no full read)
    sample = arr[..., ::max(1, arr.shape[-3] // 32),
                       ::max(1, arr.shape[-2] // 32),
                       ::max(1, arr.shape[-1] // 32)].compute()
    lo, hi = np.percentile(sample[sample > 0] if (sample > 0).any() else sample,
                            [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    print(f"  contrast: [{lo:.1f}, {hi:.1f}]")

    name = args.name or f"{args.zarr_path.name} L{args.level}"
    v = napari.Viewer(ndisplay=args.ndisplay)
    v.add_image(
        arr, name=name,
        scale=voxel,
        contrast_limits=(float(lo), float(hi)),
        rendering="mip",
        interpolation2d="nearest", interpolation3d="nearest",
    )
    print(f"napari open. units: {unit}. ndisplay={args.ndisplay}.")
    napari.run()


if __name__ == "__main__":
    main()
