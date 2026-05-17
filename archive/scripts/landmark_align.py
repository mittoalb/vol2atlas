#!/usr/bin/env python3
"""Landmark-based rigid alignment of an OME-Zarr volume to the Allen CCF.

NO sliders, NO flips toggles, NO shared state with `zrot-align align`.
The volumes are shown SIDE-BY-SIDE (napari grid mode) — sample on one tile,
CCF on the other. You pick matching landmark pairs by clicking on each tile.
Procrustes computes the rigid transform from the pairs and writes it as a
`transform.json` ready to feed `prealign_for_brainreg.py`.

Usage:
    python scripts/landmark_align.py \
        /data/sample.zarr --save transform.json --level 2 --voxel-um 2.74
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr

from zrot.atlas import load_ccf
from zrot.io import open_multiscale
from zrot.transform import RigidTransform, save_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path", type=Path)
    ap.add_argument("--save", type=Path, default=Path("transform.json"))
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--voxel-um", type=float, default=None,
                    help="Override sample voxel size (µm). Single isotropic value.")
    ap.add_argument("--atlas", default="allen_mouse_25um")
    ap.add_argument("--preview-voxels", type=int, default=192 ** 3)
    args = ap.parse_args()

    # ---- load sample preview --------------------------------------------
    ms = open_multiscale(str(args.zarr_path))
    if not 0 <= args.level < ms.n_levels():
        sys.exit(f"--level {args.level} not in 0..{ms.n_levels() - 1}")
    arr = ms.level(args.level)
    if "c" in ms.axes:
        arr = arr[args.channel]
    sample_um = (args.voxel_um,) * 3 if args.voxel_um else ms.spacing(args.level)
    spatial = arr.shape[-3:]
    n_vox = int(np.prod(spatial))
    if n_vox > args.preview_voxels:
        factor = max(1, int(np.ceil((n_vox / args.preview_voxels) ** (1 / 3))))
        print(f"strided x{factor} for preview...", flush=True)
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    print(f"loading {arr.shape} {arr.dtype} into RAM "
          f"({arr.nbytes / 1e6:.1f} MB)...", flush=True)
    sample = np.ascontiguousarray(arr.compute())
    print(f"sample preview: {sample.shape} @ {sample_um} µm")

    # ---- load atlas ------------------------------------------------------
    ccf = load_ccf(args.atlas)
    print(f"atlas: {ccf.reference.shape} @ {ccf.voxel_um} µm")

    # ---- napari with sample + CCF in grid mode (separate tiles) ---------
    viewer = napari.Viewer(ndisplay=2, title="landmark_align")

    sample_layer = viewer.add_image(
        sample, name="sample", scale=sample_um,
        colormap="magenta", contrast_limits=_percentiles(sample),
        interpolation2d="nearest", interpolation3d="nearest",
    )
    ccf_layer = viewer.add_image(
        ccf.reference, name="CCF", scale=ccf.voxel_um,
        colormap="gray", contrast_limits=_percentiles(ccf.reference),
        interpolation2d="nearest", interpolation3d="nearest",
    )
    # Points layers, attached to their respective images so they share the
    # same world coords in grid mode.
    sample_pts = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=sample_um,
        name="sample landmarks", face_color="cyan", border_color="white",
        size=max(50.0, sample_um[0] * 5),
    )
    ccf_pts = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=ccf.voxel_um,
        name="ccf landmarks", face_color="yellow", border_color="white",
        size=max(50.0, ccf.voxel_um[0] * 5),
    )

    # Grid mode: each layer in its own tile. Stride=2 groups image+points.
    viewer.grid.enabled = True
    try:
        viewer.grid.stride = -2     # pair (sample, sample_pts) and (CCF, ccf_pts)
    except Exception:
        pass

    # ---- UI ---------------------------------------------------------------
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                QLabel, QListWidget, QGroupBox)

    panel = QWidget()
    v = QVBoxLayout(panel)
    v.setContentsMargins(4, 4, 4, 4)

    instr = QLabel(
        "<b>Workflow</b>:<br>"
        "1. In the LEFT layer list, click <b>sample landmarks</b>.<br>"
        "2. Press <b>P</b> (or click the '+' in the layer-controls toolbar) to enter add-points mode.<br>"
        "3. Click on a feature in the <b>magenta sample tile</b>.<br>"
        "4. Switch to <b>ccf landmarks</b>, click the matching feature in the <b>gray CCF tile</b>.<br>"
        "5. Repeat for ≥3 pairs. Same row index = corresponding pair.<br>"
        "6. Click <b>Fit + Save</b>."
    )
    instr.setWordWrap(True)
    v.addWidget(instr)

    lists_row = QHBoxLayout()
    sbox = QGroupBox("sample (sample µm)")
    sl = QVBoxLayout(sbox); s_list = QListWidget(); s_list.setMaximumHeight(180); sl.addWidget(s_list)
    cbox = QGroupBox("CCF (CCF µm)")
    cl = QVBoxLayout(cbox); c_list = QListWidget(); c_list.setMaximumHeight(180); cl.addWidget(c_list)
    lists_row.addWidget(sbox); lists_row.addWidget(cbox)
    lists_wrap = QWidget(); lists_wrap.setLayout(lists_row); v.addWidget(lists_wrap)

    fit_btn = QPushButton("Fit + Save transform")
    save_lbl = QLabel(f"will save to: {args.save.resolve()}")
    save_lbl.setStyleSheet("color: #888;")
    v.addWidget(fit_btn); v.addWidget(save_lbl)

    def _refresh_lists():
        s_list.clear(); c_list.clear()
        # Points layer data is in voxel coords; multiply by scale → µm.
        s_pts = np.asarray(sample_pts.data, dtype=float) * np.array(sample_um)
        c_pts = np.asarray(ccf_pts.data, dtype=float) * np.array(ccf.voxel_um)
        for i, (z, y, x) in enumerate(s_pts):
            s_list.addItem(f"#{i}  z={z:+8.1f}  y={y:+8.1f}  x={x:+8.1f}")
        for i, (z, y, x) in enumerate(c_pts):
            c_list.addItem(f"#{i}  z={z:+8.1f}  y={y:+8.1f}  x={x:+8.1f}")

    sample_pts.events.data.connect(lambda _: _refresh_lists())
    ccf_pts.events.data.connect(lambda _: _refresh_lists())

    def _fit_and_save():
        sp_vox = np.asarray(sample_pts.data, dtype=float)
        cp_vox = np.asarray(ccf_pts.data,    dtype=float)
        n = min(len(sp_vox), len(cp_vox))
        if n < 3:
            viewer.status = f"need ≥3 in each (sample={len(sp_vox)}, ccf={len(cp_vox)})"
            return
        sp_um = sp_vox[:n] * np.array(sample_um)   # sample-µm landmarks
        cp_um = cp_vox[:n] * np.array(ccf.voxel_um)  # CCF-µm landmarks

        # Procrustes / Kabsch: R @ sp + t ≈ cp
        sc = sp_um.mean(0); tc = cp_um.mean(0)
        H = (sp_um - sc).T @ (cp_um - tc)
        U, S, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = tc - R @ sc

        from scipy.spatial.transform import Rotation
        rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)

        # Sanity: per-pair residuals
        pred = (R @ sp_um.T).T + t
        per = np.sqrt(np.sum((pred - cp_um) ** 2, axis=1))
        rms = float(np.sqrt(np.mean(per ** 2)))

        # Save the transform — center_um=0, no flips. R + t are in µm.
        tr = RigidTransform(
            rz_deg=float(rz), ry_deg=float(ry), rx_deg=float(rx),
            tz_um=float(t[0]), ty_um=float(t[1]), tx_um=float(t[2]),
            flip_z=False, flip_y=False, flip_x=False,
            center_um=(0.0, 0.0, 0.0),
        )
        save_transform(tr, args.save)

        print(f"\n[landmark_align] fit {n} pairs, RMS = {rms:.1f} µm")
        print(f"  rotation   : rz={rz:+7.2f}  ry={ry:+7.2f}  rx={rx:+7.2f} deg")
        print(f"  translation: tz={t[0]:+9.1f}  ty={t[1]:+9.1f}  tx={t[2]:+9.1f} µm")
        print(f"  per-pair residuals (µm):")
        for i, r in enumerate(per):
            print(f"    pair {i:2d}: {r:7.1f}{'   ← outlier?' if r > 3 * rms else ''}")
        print(f"saved → {args.save.resolve()}")
        print(f"\nNext: python scripts/prealign_for_brainreg.py "
              f"{args.zarr_path} {args.save} sample_prealigned.nii.gz"
              + (f" --voxel-um {args.voxel_um}" if args.voxel_um else ""))
        viewer.status = f"fit {n} pairs, RMS {rms:.0f} µm — saved {args.save}"

    fit_btn.clicked.connect(_fit_and_save)
    viewer.window.add_dock_widget(panel, name="landmarks", area="right")

    print("napari open. Pick pairs, click 'Fit + Save'.")
    napari.run()


def _percentiles(arr: np.ndarray) -> tuple[float, float]:
    flat = arr.ravel()
    if flat.size > 1_000_000:
        flat = flat[:: flat.size // 1_000_000]
    lo, hi = np.percentile(flat[flat > 0] if (flat > 0).any() else flat, [1, 99.5])
    return float(lo), float(max(hi, lo + 1))


if __name__ == "__main__":
    main()
