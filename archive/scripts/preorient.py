#!/usr/bin/env python3
"""Manually flip / 90°-rotate an OME-Zarr volume into the desired orientation
and save the result as a NIfTI (or zarr) for downstream registration.

Operations are LOSSLESS (np.flip and np.rot90 — no interpolation, no resampling).
Use this when your µCT was acquired in a different orientation than the atlas
expects, before feeding it to brainreg / `prealign_for_brainreg.py`.

Workflow:
    python preorient.py /data/sample.zarr --level 2 --out sample_oriented.nii.gz
    # napari opens. Click flip/rotate buttons until the brain matches the
    # atlas's convention (use `view_atlas.py` in a second window to compare).
    # Click "Save" — the operations are applied to the chosen level and
    # written to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path", type=Path, help="OME-Zarr group path.")
    ap.add_argument("--level", type=int, default=2,
                    help="Pyramid level to load (default: 2).")
    ap.add_argument("--channel", type=int, default=0,
                    help="If 4D (c,z,y,x), pick this channel.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Decimate by N for fast preview (does NOT affect output).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output path. Extension picks the format: "
                         ".nii.gz / .nii (NIfTI) or .npy (numpy).")
    args = ap.parse_args()

    import dask.array as da
    import napari
    import numpy as np
    import zarr
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                QLabel)

    # ---- read the chosen level + voxel size ------------------------------
    root = zarr.open(str(args.zarr_path), mode="r")
    if "multiscales" not in root.attrs:
        sys.exit("not an OME-Zarr group (no multiscales metadata).")
    ms = root.attrs["multiscales"][0]
    if not 0 <= args.level < len(ms["datasets"]):
        sys.exit(f"--level {args.level} not in 0..{len(ms['datasets'])-1}")
    ds = ms["datasets"][args.level]
    scale = next(t["scale"] for t in ds.get("coordinateTransformations", [])
                 if t["type"] == "scale")
    voxel_um = tuple(float(s) for s in scale[-3:])  # (z, y, x)
    print(f"level {args.level}: voxel={voxel_um} µm")

    arr = da.from_zarr(root[ds["path"]])
    if arr.ndim == 4:
        arr = arr[args.channel]
    if args.stride > 1:
        arr = arr[::args.stride, ::args.stride, ::args.stride]
        voxel_um = tuple(v * args.stride for v in voxel_um)
        print(f"strided x{args.stride}: shape {arr.shape}, voxel {voxel_um}")
    print(f"loading into RAM ({arr.nbytes / 1e9:.2f} GB)...", flush=True)
    data_original = np.ascontiguousarray(arr.compute())
    print(f"loaded: {data_original.shape}")

    # ---- state: list of operations to apply ------------------------------
    # Each op is ("flip", axis) or ("rot90", (axis_a, axis_b), k)
    ops: list = []

    def apply_ops(src):
        out = src
        for op in ops:
            if op[0] == "flip":
                out = np.flip(out, axis=op[1])
            elif op[0] == "rot90":
                out = np.rot90(out, k=op[2], axes=op[1])
        return np.ascontiguousarray(out)

    # ---- viewer setup ----------------------------------------------------
    viewer = napari.Viewer(ndisplay=3)
    layer = viewer.add_image(
        data_original,
        name="sample (orient me)",
        scale=voxel_um,
        rendering="mip",
        colormap="gray",
        interpolation2d="nearest", interpolation3d="nearest",
    )

    status = QLabel("ops: (none)")

    def _redraw():
        new = apply_ops(data_original)
        layer.data = new
        # rotations swap axes, so the displayed scale must follow
        # Track the current voxel ordering: start as (z, y, x).
        order = list(range(3))
        for op in ops:
            if op[0] == "rot90":
                a, b = op[1]
                # rot90 by k swaps axes a and b (k odd) — keep it simple:
                if op[2] % 2 == 1:
                    order[a], order[b] = order[b], order[a]
        layer.scale = tuple(voxel_um[i] for i in order)
        status.setText("ops: " + (" ".join(_op_str(o) for o in ops) or "(none)"))

    def _op_str(op):
        if op[0] == "flip":
            return f"flip{'zyx'[op[1]]}"
        if op[0] == "rot90":
            a, b = op[1]
            axname = "zyx"[a] + "zyx"[b]
            return f"rot90_{axname}({op[2]})"
        return str(op)

    # ---- buttons ---------------------------------------------------------
    panel = QWidget()
    pl = QVBoxLayout(panel)
    pl.addWidget(QLabel("<b>flips</b>"))
    for axis_idx, axis_name in enumerate(("z", "y", "x")):
        b = QPushButton(f"flip {axis_name}")
        b.clicked.connect(lambda _, i=axis_idx: (ops.append(("flip", i)), _redraw()))
        pl.addWidget(b)

    pl.addWidget(QLabel("<b>90° rotations</b>"))
    # rot90 around an axis = swap the OTHER two
    #   around z (axis 0) → swap (1, 2) = y-x plane
    #   around y (axis 1) → swap (0, 2) = z-x plane
    #   around x (axis 2) → swap (0, 1) = z-y plane
    rot_axes = {
        "z (in y-x plane)": (1, 2),
        "y (in z-x plane)": (0, 2),
        "x (in z-y plane)": (0, 1),
    }
    for label, axes_pair in rot_axes.items():
        row = QHBoxLayout()
        b_pos = QPushButton(f"+90° around {label}")
        b_neg = QPushButton(f"-90° around {label}")
        b_pos.clicked.connect(lambda _, ax=axes_pair: (
            ops.append(("rot90", ax, +1)), _redraw()))
        b_neg.clicked.connect(lambda _, ax=axes_pair: (
            ops.append(("rot90", ax, -1)), _redraw()))
        row.addWidget(b_pos); row.addWidget(b_neg)
        rw = QWidget(); rw.setLayout(row); pl.addWidget(rw)

    pl.addWidget(QLabel("<b>edit</b>"))
    undo_btn = QPushButton("Undo last")
    undo_btn.clicked.connect(lambda: (ops.pop() if ops else None, _redraw()))
    pl.addWidget(undo_btn)
    reset_btn = QPushButton("Reset (clear all)")
    reset_btn.clicked.connect(lambda: (ops.clear(), _redraw()))
    pl.addWidget(reset_btn)

    pl.addWidget(QLabel("<b>save</b>"))
    pl.addWidget(status)
    save_btn = QPushButton(f"Save → {args.out.name}")

    def _save():
        out_data = apply_ops(data_original)
        # current scale after ops
        order = list(range(3))
        for op in ops:
            if op[0] == "rot90" and op[2] % 2 == 1:
                a, b = op[1]
                order[a], order[b] = order[b], order[a]
        out_voxel_um = tuple(voxel_um[i] for i in order)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        suffix = "".join(args.out.suffixes).lower()
        if suffix in (".nii", ".nii.gz"):
            import nibabel as nib
            sz, sy, sx = (v * 1e-3 for v in out_voxel_um)
            affine = np.diag([sx, sy, sz, 1.0])
            nib.save(
                nib.Nifti1Image(np.transpose(out_data, (2, 1, 0)), affine),
                str(args.out),
            )
        elif suffix == ".npy":
            np.save(str(args.out), out_data)
        else:
            sys.exit(f"unsupported extension '{suffix}'. Use .nii.gz or .npy.")

        msg = (f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)  "
               f"shape={out_data.shape}  voxel={out_voxel_um} µm  ops: {ops}")
        print(msg)
        viewer.status = msg

    save_btn.clicked.connect(_save)
    pl.addWidget(save_btn)

    viewer.window.add_dock_widget(panel, name="orient", area="right")

    print("napari open. Click flip/rotate buttons; click Save when ready.")
    print("Tip: open `view_atlas.py` in another terminal to compare orientations.")
    napari.run()


if __name__ == "__main__":
    main()
