"""Step 1: 3D rough prealignment of the sample onto the CCF, with CCF crop.

WYSIWYG: the only sample layer you see is a live scipy resample of your data
through the current transform onto the (possibly cropped) CCF voxel grid. What
you see IS what gets saved into state.json.

Reads + writes state.json (transform, ccf_crop_bbox).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..atlas import load_ccf
from ..io import open_multiscale
from ..state import State, save as save_state
from ..transform import RigidTransform


def run(state_path: Path) -> None:
    from ..state import load
    state = load(state_path)
    _run_napari(state, state_path)


def _run_napari(state: State, state_path: Path) -> None:
    import napari
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QCheckBox, QGroupBox, QTabWidget,
                                QDoubleSpinBox, QSlider)
    from qtpy.QtCore import Qt
    from scipy.ndimage import affine_transform

    # -------- load sample preview + atlas ---------------------------------
    ms = open_multiscale(state.sample_zarr)
    arr = ms.level(state.sample_level)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    sample_um = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                 else ms.spacing(state.sample_level))

    PREVIEW_MAX = 192 ** 3
    n_vox = int(np.prod(arr.shape))
    if n_vox > PREVIEW_MAX:
        factor = max(1, int(np.ceil((n_vox / PREVIEW_MAX) ** (1 / 3))))
        print(f"[step1] strided x{factor} for preview...", flush=True)
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    print(f"[step1] loading {arr.shape} into RAM...", flush=True)
    sample_np = np.ascontiguousarray(arr.compute())
    sample_center_um = tuple((s - 1) * v / 2.0 for s, v in zip(sample_np.shape, sample_um))
    print(f"[step1] preview: {sample_np.shape}  voxel={sample_um} µm")

    ccf = load_ccf(state.atlas_name)
    ccf_ref_full = np.asarray(ccf.reference)
    print(f"[step1] atlas: {ccf_ref_full.shape}  voxel={ccf.voxel_um} µm")

    # -------- contrast --------------------------------------------------
    def _percentiles(a, lo=1, hi=99.5):
        f = a.ravel()
        if f.size > 1_000_000:
            f = f[:: f.size // 1_000_000]
        f = f[f > 0] if (f > 0).any() else f
        a_lo, a_hi = np.percentile(f, [lo, hi])
        return float(a_lo), float(max(a_hi, a_lo + 1))

    sample_clim = _percentiles(sample_np)
    ccf_clim    = _percentiles(ccf_ref_full)

    # -------- initial transform ----------------------------------------
    if state.transform is None:
        ccf_center_um = tuple((s - 1) * v / 2.0 for s, v in zip(ccf_ref_full.shape, ccf.voxel_um))
        tr0 = RigidTransform(
            tz_um=ccf_center_um[0] - sample_center_um[0],
            ty_um=ccf_center_um[1] - sample_center_um[1],
            tx_um=ccf_center_um[2] - sample_center_um[2],
            center_um=sample_center_um,
        )
    else:
        saved_center = tuple(state.transform.get("center_um")) \
            if state.transform.get("center_um") is not None \
            else sample_center_um
        tr0 = RigidTransform(
            **{k: state.transform[k] for k in
               ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
            flip_z=bool(state.transform.get("flip_z", False)),
            flip_y=bool(state.transform.get("flip_y", False)),
            flip_x=bool(state.transform.get("flip_x", False)),
            center_um=saved_center,
        )

    box = {"transform": tr0, "ccf_data": None}

    # -------- apply crop to CCF view -----------------------------------
    def _apply_crop():
        b = state.ccf_crop_bbox
        if b is None:
            box["ccf_data"] = ccf_ref_full
            box["ccf_origin_um"] = (0.0, 0.0, 0.0)
        else:
            z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
            cropped = ccf_ref_full[z0:z1, y0:y1, x0:x1]
            box["ccf_data"] = cropped
            box["ccf_origin_um"] = (z0 * ccf.voxel_um[0],
                                    y0 * ccf.voxel_um[1],
                                    x0 * ccf.voxel_um[2])
    _apply_crop()

    # -------- napari viewer --------------------------------------------
    viewer = napari.Viewer(ndisplay=3, title="vol2atlas prealign: prealign + crop")

    ccf_layer = viewer.add_image(
        box["ccf_data"], name="CCF",
        scale=ccf.voxel_um, translate=box["ccf_origin_um"],
        colormap="gray", contrast_limits=ccf_clim, opacity=0.5,
        blending="additive", rendering="mip",
    )

    # Sample layer: a scipy resample onto the CCF (cropped) voxel grid.
    def _resample():
        ccf_data = box["ccf_data"]
        M = box["transform"].for_voxel_grid(sample_um, ccf.voxel_um)
        Minv = np.linalg.inv(M)
        # offset the resampling so it lands in the CROPPED grid
        b = state.ccf_crop_bbox
        if b is None:
            out_offset_voxel = np.zeros(3)
        else:
            out_offset_voxel = np.array([b["z"][0], b["y"][0], b["x"][0]])
        offset = Minv[:3, :3] @ out_offset_voxel + Minv[:3, 3]
        return affine_transform(
            sample_np, Minv[:3, :3], offset=offset,
            output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
        )

    sample_layer = viewer.add_image(
        _resample(), name="sample (live = saved)",
        scale=ccf.voxel_um, translate=box["ccf_origin_um"],
        colormap="magenta", contrast_limits=sample_clim, opacity=0.6,
        blending="additive", rendering="mip",
    )

    # (crop is driven by sliders below, not freehand shapes)

    # Debounced redraw
    redraw_timer = QTimer(); redraw_timer.setSingleShot(True)
    def _redraw():
        sample_layer.data = _resample()
    redraw_timer.timeout.connect(_redraw)
    def _schedule_redraw():
        redraw_timer.start(150)

    # -------- controls panel -------------------------------------------
    ctrl = QWidget(); ctrl_layout = QVBoxLayout(ctrl)
    ctrl_layout.setContentsMargins(4, 4, 4, 4)

    abs_lbl = QLabel(); ctrl_layout.addWidget(abs_lbl)
    def _refresh_label():
        t = box["transform"]
        flips = "".join(a for a, f in zip("zyx", (t.flip_z, t.flip_y, t.flip_x)) if f) or "—"
        abs_lbl.setText(
            f"<pre>tz={t.tz_um:+8.1f}  ty={t.ty_um:+8.1f}  tx={t.tx_um:+8.1f} µm\n"
            f"rz={t.rz_deg:+7.2f}  ry={t.ry_deg:+7.2f}  rx={t.rx_deg:+7.2f} deg\n"
            f"flips: {flips}</pre>"
        )

    def _spin(label, default, vmin, vmax, step, setter):
        from qtpy.QtWidgets import QSizePolicy
        row_w = QWidget(); row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(2)
        lbl = QLabel(label); lbl.setMinimumWidth(0)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        row.addWidget(lbl)
        n_ticks = max(2, int(round((vmax - vmin) / step)))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0); slider.setMaximum(n_ticks)
        slider.setValue(int(round((default - vmin) / step)))
        slider.setPageStep(max(1, n_ticks // 50))
        slider.setSingleStep(max(1, n_ticks // 500))
        # Critical: allow the slider to shrink horizontally.
        slider.setMinimumWidth(40)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(slider, 1)
        sb = QDoubleSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step)
        sb.setDecimals(2); sb.setValue(default)
        sb.setMinimumWidth(0)
        sb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        row.addWidget(sb)

        guard = {"x": False}
        def _on_slider(t):
            if guard["x"]: return
            v = vmin + t * step
            guard["x"] = True; sb.setValue(v); guard["x"] = False
            setter(v); _refresh_label(); _schedule_redraw()
        def _on_spin(v):
            if guard["x"]: return
            t = int(round((v - vmin) / step))
            guard["x"] = True; slider.setValue(t); guard["x"] = False
            setter(v); _refresh_label(); _schedule_redraw()
        slider.valueChanged.connect(_on_slider)
        sb.valueChanged.connect(_on_spin)
        return row_w, sb

    def _setter(attr):
        return lambda v: setattr(box["transform"], attr, float(v))

    bbox_um = max(s*v for s, v in zip(sample_np.shape, sample_um)) + \
              max(s*v for s, v in zip(ccf_ref_full.shape, ccf.voxel_um))
    ctrl_layout.addWidget(QLabel("<b>Translations (µm)</b>"))
    for attr in ("tz_um", "ty_um", "tx_um"):
        w, _ = _spin(attr, getattr(tr0, attr), -bbox_um, bbox_um, 50.0, _setter(attr))
        ctrl_layout.addWidget(w)
    ctrl_layout.addWidget(QLabel("<b>Rotations (deg)</b>"))
    for attr in ("rz_deg", "ry_deg", "rx_deg"):
        w, _ = _spin(attr, getattr(tr0, attr), -180.0, 180.0, 1.0, _setter(attr))
        ctrl_layout.addWidget(w)

    ctrl_layout.addWidget(QLabel("<b>Flips</b>"))
    flip_row = QHBoxLayout()
    flip_chks = {}
    for axis in ("z", "y", "x"):
        cb = QCheckBox(axis); cb.setChecked(getattr(tr0, f"flip_{axis}"))
        def _make(ax):
            return lambda checked: (setattr(box["transform"], f"flip_{ax}", bool(checked)),
                                     _refresh_label(), _schedule_redraw())
        cb.stateChanged.connect(_make(axis))
        flip_chks[axis] = cb; flip_row.addWidget(cb)
    flip_row.addStretch(1)
    fr = QWidget(); fr.setLayout(flip_row); ctrl_layout.addWidget(fr)

    ctrl_layout.addStretch(1)

    # -------- crop tab: 6 sliders (min/max per axis), live preview ----
    actions = QWidget(); actions_v = QVBoxLayout(actions)
    actions_v.setContentsMargins(4, 4, 4, 4)
    actions_v.addWidget(QLabel(
        "<b>CCF crop</b> — drag the min/max sliders for each axis. The CCF "
        "and sample layers update live to show only that region."
    ))

    # Initial bbox: existing or full atlas
    init_bbox = state.ccf_crop_bbox or {
        "z": [0, ccf_ref_full.shape[0]],
        "y": [0, ccf_ref_full.shape[1]],
        "x": [0, ccf_ref_full.shape[2]],
    }
    crop_spins = {}
    for i, ax in enumerate("zyx"):
        actions_v.addWidget(QLabel(f"<b>{ax}</b> (0 – {ccf_ref_full.shape[i]})"))
        for which, default in (("min", init_bbox[ax][0]), ("max", init_bbox[ax][1])):
            key = f"{ax}_{which}"
            row_w, sb = _spin(
                f"{ax}_{which}", float(default),
                0.0, float(ccf_ref_full.shape[i]), 1.0,
                lambda v, _k=key: _on_crop_change(_k, v),
            )
            actions_v.addWidget(row_w)
            crop_spins[key] = sb

    crop_state = {"bbox": dict(init_bbox), "rebuild_timer": QTimer()}
    crop_state["rebuild_timer"].setSingleShot(True)

    def _on_crop_change(key, v):
        ax, which = key.split("_")
        crop_state["bbox"][ax][0 if which == "min" else 1] = int(round(v))
        # Enforce min <= max
        if crop_state["bbox"][ax][0] >= crop_state["bbox"][ax][1]:
            crop_state["bbox"][ax][1] = crop_state["bbox"][ax][0] + 1
            crop_spins[f"{ax}_max"].blockSignals(True)
            crop_spins[f"{ax}_max"].setValue(crop_state["bbox"][ax][1])
            crop_spins[f"{ax}_max"].blockSignals(False)
        crop_state["rebuild_timer"].start(80)

    def _rebuild_crop():
        # Sync state and rebuild layers
        full = (crop_state["bbox"]["z"] == [0, ccf_ref_full.shape[0]]
                and crop_state["bbox"]["y"] == [0, ccf_ref_full.shape[1]]
                and crop_state["bbox"]["x"] == [0, ccf_ref_full.shape[2]])
        state.ccf_crop_bbox = None if full else dict(crop_state["bbox"])
        _apply_crop()
        ccf_layer.data = box["ccf_data"]
        ccf_layer.translate = box["ccf_origin_um"]
        sample_layer.translate = box["ccf_origin_um"]
        _schedule_redraw()
    crop_state["rebuild_timer"].timeout.connect(_rebuild_crop)

    clear_crop_btn = QPushButton("Reset crop (use full CCF)")
    def _reset_crop():
        for i, ax in enumerate("zyx"):
            crop_spins[f"{ax}_min"].setValue(0.0)
            crop_spins[f"{ax}_max"].setValue(float(ccf_ref_full.shape[i]))
    clear_crop_btn.clicked.connect(_reset_crop)
    actions_v.addWidget(clear_crop_btn)

    save_btn = QPushButton("Save state.json && exit")
    def _save():
        state.transform = box["transform"].to_dict()
        state.add_history("prealign", f"crop={state.ccf_crop_bbox}")
        save_state(state, state_path)
        print(f"[step1] saved -> {state_path}", flush=True)
        viewer.status = f"saved {state_path}"
        viewer.close()
    save_btn.clicked.connect(_save)
    actions_v.addWidget(save_btn)
    actions_v.addStretch(1)

    # -------- dock layout: tabbed -------------------------------------
    from qtpy.QtWidgets import QSizePolicy
    tabs = QTabWidget()
    tabs.addTab(ctrl,    "transform")
    tabs.addTab(actions, "crop / save")
    # Let the dock shrink — important on small monitors.
    tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    tabs.setMinimumWidth(0)
    dock = viewer.window.add_dock_widget(tabs, name="prealign", area="right")
    try:
        dock.setMinimumWidth(220)   # absolute floor
        dock.setMaximumWidth(900)   # cap so it doesn't auto-grow huge
    except Exception:
        pass

    _refresh_label()
    print("\n[step1] napari open in 3D MIP.")
    print("[step1] adjust translation/rotation/flips on the 'transform' tab.")
    print("[step1] draw a crop box on the 'CCF crop ROI' layer, then 'Apply crop'.")
    print("[step1] click 'Save state.json && exit' when done.")
    napari.run()
