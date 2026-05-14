#!/usr/bin/env python3
"""Convert a 3D TIFF (e.g. brainreg's downsampled_standard.tiff) into a
multiscale OME-Zarr suitable for Neuroglancer / napari.

Usage:
    python tiff_to_ngff.py downsampled_standard.tiff out.zarr --voxel-um 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiff_in", type=Path)
    ap.add_argument("zarr_out", type=Path)
    ap.add_argument("--voxel-um", type=float, default=25.0,
                    help="Isotropic voxel size in µm (default: 25, atlas resolution).")
    ap.add_argument("--n-levels", type=int, default=5,
                    help="Pyramid levels (each 2x downsampled). Default: 5.")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    print(f"reading {args.tiff_in}...")
    data = tifffile.imread(str(args.tiff_in))
    print(f"  shape={data.shape}  dtype={data.dtype}  "
          f"size={data.nbytes / 1e6:.1f} MB")

    if args.zarr_out.exists():
        import shutil; shutil.rmtree(args.zarr_out)
    store = parse_url(str(args.zarr_out), mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=data,
        group=root,
        axes="zyx",
        coordinate_transformations=[
            [{"type": "scale", "scale": [args.voxel_um] * 3}]
        ] * args.n_levels,
        storage_options=dict(chunks=(args.chunk, args.chunk, args.chunk)),
    )
    print(f"wrote multiscale OME-Zarr: {args.zarr_out}")
    print(f"  serve with:  python -m http.server 8000")
    print(f"  Neuroglancer: zarr://http://127.0.0.1:8000/{args.zarr_out.name}")


if __name__ == "__main__":
    main()
