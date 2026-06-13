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
from ..frame import compute_output_frame, enable_ccf_axes, extract_ccf
from ..io import open_multiscale
from ..state import State, save as save_state
from ..transform import RigidTransform


def run(state_path: Path, level: int | None = None,
        preview_size: int = 192,
        orientation: str | None = None) -> None:
    from ..state import load
    state = load(state_path)
    if orientation:
        # Compute rotation from sample → atlas orientation and seed
        # state.transform with it. Pivot on sample center so the rotation
        # doesn't push the volume out of frame. Other state fields are
        # untouched.
        from ..atlas import load_ccf
        from ..orientation import (rotation_between,
                                    euler_zyx_degrees_from_matrix)
        from ..transform import RigidTransform
        from ..io import open_multiscale
        ccf = load_ccf(state.atlas_name)
        R = rotation_between(orientation, ccf.orientation)
        rz, ry, rx = euler_zyx_degrees_from_matrix(R)
        ms = open_multiscale(state.sample_zarr)
        use_level = state.sample_level if level is None else int(level)
        sample_shape = ms.level(use_level).shape
        if "c" in ms.axes:
            sample_shape = sample_shape[1:]
        sample_sp = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                     else ms.spacing(use_level))
        center_um = tuple((s - 1) * v / 2.0
                          for s, v in zip(sample_shape, sample_sp))
        state.transform = RigidTransform(
            rz_deg=rz, ry_deg=ry, rx_deg=rx,
            tz_um=0.0, ty_um=0.0, tx_um=0.0,
            center_um=center_um,
        ).to_dict()
        state.sample_orientation = orientation
        print(f"[step1] orientation override: sample {orientation!r} → "
              f"atlas {ccf.orientation!r}  ⇒  rotation "
              f"(rz={rz:+.1f}, ry={ry:+.1f}, rx={rx:+.1f}) deg", flush=True)
    _run_napari(state, state_path, level=level, preview_size=preview_size)


def _run_napari(state: State, state_path: Path, *,
                level: int | None = None,
                preview_size: int = 192) -> None:
    import napari
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QCheckBox, QGroupBox, QTabWidget,
                                QDoubleSpinBox, QSlider)
    from qtpy.QtCore import Qt
    from scipy.ndimage import affine_transform

    # -------- load sample preview + atlas ---------------------------------
    ms = open_multiscale(state.sample_zarr)
    use_level = state.sample_level if level is None else int(level)
    arr = ms.level(use_level)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    if state.sample_voxel_um is not None and use_level != state.sample_level:
        anc = ms.spacing(state.sample_level)
        cur = ms.spacing(use_level)
        sample_um = tuple(v * (c / a) for v, a, c
                           in zip(state.sample_voxel_um, anc, cur))
    elif state.sample_voxel_um is not None:
        sample_um = tuple(state.sample_voxel_um)
    else:
        sample_um = ms.spacing(use_level)
    print(f"[step1] level {use_level}: {arr.shape} @ {sample_um} µm",
          flush=True)

    PREVIEW_MAX = int(preview_size) ** 3
    n_vox = int(np.prod(arr.shape))
    if n_vox > PREVIEW_MAX:
        factor = max(1, int(np.ceil((n_vox / PREVIEW_MAX) ** (1 / 3))))
        print(f"[step1] strided x{factor} for preview "
              f"(budget {preview_size}^3 = {PREVIEW_MAX/1e6:.1f}M voxels)...",
              flush=True)
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
            center_um=saved_center,
        )

    box = {"transform": tr0, "ccf_data": None, "ccf_origin_um": None}

    # -------- CCF layer reflects ONLY the user's crop bbox -------------
    def _apply_crop_ccf():
        b = state.ccf_crop_bbox
        if b is None:
            box["ccf_data"] = ccf_ref_full
            box["ccf_origin_um"] = (0.0, 0.0, 0.0)
        else:
            z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
            box["ccf_data"] = ccf_ref_full[z0:z1, y0:y1, x0:x1]
            box["ccf_origin_um"] = (z0 * ccf.voxel_um[0],
                                    y0 * ccf.voxel_um[1],
                                    x0 * ccf.voxel_um[2])
    _apply_crop_ccf()

    # -------- napari viewer --------------------------------------------
    viewer = napari.Viewer(ndisplay=3, title="vol2atlas prealign: prealign + crop")
    enable_ccf_axes(viewer, ccf.orientation)

    ccf_layer = viewer.add_image(
        box["ccf_data"], name="CCF",
        scale=ccf.voxel_um, translate=box["ccf_origin_um"],
        colormap="gray", contrast_limits=ccf_clim, opacity=0.5,
        blending="additive", rendering="mip",
    )

    # Sample shown via napari's GPU-applied layer.affine transform.
    # No scipy.affine_transform per slider tick — just an updated affine
    # matrix that vispy applies at render time. Same speed as ImageJ.
    sample_layer = viewer.add_image(
        sample_np, name="sample (live = saved)",
        scale=sample_um,
        colormap="gray", contrast_limits=sample_clim, opacity=0.6,
        blending="additive", rendering="mip",
    )
    sample_layer.affine = tr0.matrix()

    # Redraw: just update the affine matrix. GPU does the rest.
    def _redraw():
        sample_layer.affine = box["transform"].matrix()
    def _schedule_redraw():
        _redraw()      # instant — no debounce needed

    # -------- controls panel -------------------------------------------
    ctrl = QWidget(); ctrl_layout = QVBoxLayout(ctrl)
    ctrl_layout.setContentsMargins(4, 4, 4, 4)

    abs_lbl = QLabel(); ctrl_layout.addWidget(abs_lbl)
    def _refresh_label():
        t = box["transform"]
        abs_lbl.setText(
            f"<pre>tz={t.tz_um:+8.1f}  ty={t.ty_um:+8.1f}  tx={t.tx_um:+8.1f} µm\n"
            f"rz={t.rz_deg:+7.2f}  ry={t.ry_deg:+7.2f}  rx={t.rx_deg:+7.2f} deg</pre>"
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
        full = (crop_state["bbox"]["z"] == [0, ccf_ref_full.shape[0]]
                and crop_state["bbox"]["y"] == [0, ccf_ref_full.shape[1]]
                and crop_state["bbox"]["x"] == [0, ccf_ref_full.shape[2]])
        state.ccf_crop_bbox = None if full else dict(crop_state["bbox"])
        _apply_crop_ccf()
        ccf_layer.data = box["ccf_data"]
        ccf_layer.translate = box["ccf_origin_um"]
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
    print("[step1] adjust translation/rotation on the 'transform' tab.")
    print("[step1] draw a crop box on the 'CCF crop ROI' layer, then 'Apply crop'.")
    print("[step1] click 'Save state.json && exit' when done.")
    napari.run()
