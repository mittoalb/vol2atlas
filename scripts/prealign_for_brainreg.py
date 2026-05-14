#!/usr/bin/env python3
"""Use a hand-aligned `zrot` transform to write a NIfTI in CCF coordinates,
ready for brainreg's deformable refinement.

Workflow:
    # 1. Hand-align in zrot (focus on orientation; translation can be rough)
    zrot-align align /data/sample.zarr --level 2 --save transform.json \
        --ndisplay 2 --no-write-into-zarr

    # 2. Use this script to produce a NIfTI in CCF coords
    python scripts/prealign_for_brainreg.py /data/sample.zarr transform.json \
        sample_prealigned.nii.gz --atlas allen_mouse_25um

    # 3. Run brainreg without orientation guesses (it's already in CCF orientation)
    brainreg sample_prealigned.nii.gz ./brainreg_out \
        -v 25 25 25 --orientation asr --atlas allen_mouse_25um \
        --brain_geometry hemisphere_r
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from zrot.atlas import load_ccf
from zrot.io import open_multiscale
from zrot.transform import load_transform
from zrot.refine import _resample_to_ccf_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_in", type=Path, help="Input OME-Zarr group.")
    ap.add_argument("transform", type=Path, help="zrot rigid transform JSON.")
    ap.add_argument("nifti_out", type=Path, help="Output NIfTI (.nii.gz).")
    ap.add_argument("--atlas", default="allen_mouse_25um")
    ap.add_argument("--input-level", type=int, default=None,
                    help="Pyramid level. Default: closest to atlas resolution.")
    ap.add_argument("--channel", type=int, default=0)
    args = ap.parse_args()

    ms = open_multiscale(str(args.zarr_in))
    rigid = load_transform(args.transform)
    ccf = load_ccf(args.atlas)

    if args.input_level is None:
        target = float(ccf.voxel_um[0])
        args.input_level = min(range(ms.n_levels()),
                               key=lambda i: abs(min(ms.spacing(i)) - target))
    print(f"using level {args.input_level} @ {ms.spacing(args.input_level)} µm")
    print(f"writing prewarp at {ccf.voxel_um} µm into CCF grid...")
    warped = _resample_to_ccf_grid(ms, rigid, ccf,
                                    level=args.input_level,
                                    channel=args.channel)
    print(f"prewarp shape: {warped.shape}, dtype: {warped.dtype}")

    # NIfTI: spacing in mm, isotropic, RAS frame (4x4 affine = diag of mm)
    sp_mm = tuple(v * 1e-3 for v in ccf.voxel_um)  # (z, y, x) µm -> mm
    affine = np.diag([sp_mm[2], sp_mm[1], sp_mm[0], 1.0])  # x, y, z order
    args.nifti_out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.transpose(warped, (2, 1, 0)).astype(warped.dtype), affine),
        str(args.nifti_out),
    )
    print(f"wrote: {args.nifti_out}  ({args.nifti_out.stat().st_size / 1e6:.1f} MB)")
    print(f"\nNext: brainreg {args.nifti_out} ./brainreg_out \\")
    print(f"        -v {ccf.voxel_um[0]} {ccf.voxel_um[1]} {ccf.voxel_um[2]} \\")
    print(f"        --orientation asr --atlas {args.atlas} \\")
    print(f"        --brain_geometry hemisphere_r   # adjust to your hemisphere")


if __name__ == "__main__":
    main()
