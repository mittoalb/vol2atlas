"""Step 2: fine alignment in a single napari window with view-switch buttons
(axial / coronal / sagittal / 3D MIP) — no multi-window mess.

Loads state.json from step 1. Same WYSIWYG live scipy resample as step 1, but:
- Slider ranges are NARROWER (±3 mm translation, ±30° rotation) for fine work.
- View-switch buttons let you check the alignment in every orthogonal plane
  before saving.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..atlas import load_ccf
from ..io import open_multiscale
from ..state import save as save_state
from ..transform import RigidTransform


def run(state_path: Path) -> None:
    from ..state import load
    state = load(state_path)
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run `vol2atlas prealign` first.")
    _run_napari(state, state_path)


def _run_napari(state, state_path: Path) -> None:
    import napari
    from qtpy.QtCore import QTimer, Qt
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QCheckBox, QDoubleSpinBox, QSlider,
                                QSizePolicy, QTabWidget)
    from scipy.ndimage import affine_transform

    # --------- load sample + atlas (with crop) -----------------------
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
        print(f"[step2] strided x{factor} for preview...", flush=True)
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    print(f"[step2] loading {arr.shape} into RAM...", flush=True)
    sample_np = np.ascontiguousarray(arr.compute())
    sample_center_um = tuple((s - 1) * v / 2.0 for s, v in zip(sample_np.shape, sample_um))

    ccf = load_ccf(state.atlas_name)
    ccf_ref_full = np.asarray(ccf.reference)
    b = state.ccf_crop_bbox
    if b is None:
        ccf_data = ccf_ref_full
        ccf_origin = (0.0, 0.0, 0.0)
        crop_voxel_origin = np.zeros(3)
    else:
        z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
        ccf_data = ccf_ref_full[z0:z1, y0:y1, x0:x1]
        ccf_origin = (z0 * ccf.voxel_um[0], y0 * ccf.voxel_um[1], x0 * ccf.voxel_um[2])
        crop_voxel_origin = np.array([z0, y0, x0], dtype=float)
    print(f"[step2] CCF crop: {ccf_data.shape} at origin {ccf_origin} µm")

    # --------- initial transform -------------------------------------
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
    box = {"transform": tr0, "center_um": saved_center}
    print(f"[step2] loaded transform: rz={tr0.rz_deg:+.2f} ry={tr0.ry_deg:+.2f} "
          f"rx={tr0.rx_deg:+.2f}  tz={tr0.tz_um:+.1f} ty={tr0.ty_um:+.1f} "
          f"tx={tr0.tx_um:+.1f}  flips=("
          f"{'z' if tr0.flip_z else '·'}"
          f"{'y' if tr0.flip_y else '·'}"
          f"{'x' if tr0.flip_x else '·'})", flush=True)

    def _resample():
        M = box["transform"].for_voxel_grid(sample_um, ccf.voxel_um)
        Minv = np.linalg.inv(M)
        offset = Minv[:3, :3] @ crop_voxel_origin + Minv[:3, 3]
        return affine_transform(
            sample_np, Minv[:3, :3], offset=offset,
            output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
        )

    # --------- single viewer ----------------------------------------
    def _percentiles(a, lo=1, hi=99.5):
        f = a.ravel()
        if f.size > 1_000_000:
            f = f[:: f.size // 1_000_000]
        f = f[f > 0] if (f > 0).any() else f
        a_lo, a_hi = np.percentile(f, [lo, hi])
        return float(a_lo), float(max(a_hi, a_lo + 1))

    ccf_clim    = _percentiles(ccf_data)
    sample_clim = _percentiles(sample_np)

    viewer = napari.Viewer(ndisplay=2, title="vol2atlas refine — refine")
    viewer.add_image(ccf_data, name="CCF", scale=ccf.voxel_um, translate=ccf_origin,
                     colormap="gray", contrast_limits=ccf_clim, opacity=0.5,
                     blending="additive",
                     interpolation2d="nearest", interpolation3d="nearest")
    sample_layer = viewer.add_image(
        _resample(), name="sample (live=saved)",
        scale=ccf.voxel_um, translate=ccf_origin,
        colormap="magenta", contrast_limits=sample_clim, opacity=0.6,
        blending="additive", rendering="mip",
        interpolation2d="nearest", interpolation3d="nearest")

    redraw_timer = QTimer(); redraw_timer.setSingleShot(True)
    def _redraw():
        sample_layer.data = _resample()
    redraw_timer.timeout.connect(_redraw)
    def _schedule_redraw():
        redraw_timer.start(150)

    # --------- controls panel --------------------------------------
    ctrl = QWidget(); ctrl_v = QVBoxLayout(ctrl)
    ctrl_v.setContentsMargins(4, 4, 4, 4)

    abs_lbl = QLabel(); ctrl_v.addWidget(abs_lbl)
    def _refresh_label():
        t = box["transform"]
        flips = "".join(a for a, f in zip("zyx", (t.flip_z, t.flip_y, t.flip_x)) if f) or "—"
        abs_lbl.setText(
            f"<pre>tz={t.tz_um:+8.1f}  ty={t.ty_um:+8.1f}  tx={t.tx_um:+8.1f} µm\n"
            f"rz={t.rz_deg:+7.2f}  ry={t.ry_deg:+7.2f}  rx={t.rx_deg:+7.2f} deg\n"
            f"flips: {flips}</pre>"
        )

    def _spin(label, default, vmin, vmax, step, setter):
        row_w = QWidget(); row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(2)
        lbl = QLabel(label); lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        row.addWidget(lbl)
        n_ticks = max(2, int(round((vmax - vmin) / step)))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0); slider.setMaximum(n_ticks)
        slider.setValue(int(round((default - vmin) / step)))
        slider.setPageStep(max(1, n_ticks // 50))
        slider.setSingleStep(max(1, n_ticks // 500))
        slider.setMinimumWidth(40)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(slider, 1)
        sb = QDoubleSpinBox(); sb.setRange(vmin, vmax); sb.setSingleStep(step)
        sb.setDecimals(2); sb.setValue(default)
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
        return row_w

    def _setter(attr):
        return lambda v: setattr(box["transform"], attr, float(v))

    # Narrow ranges for fine work (±3 mm, ±30°)
    T_FINE = 3000.0
    R_FINE = 30.0
    ctrl_v.addWidget(QLabel("<b>Translation (µm, ±3000 from current)</b>"))
    for attr in ("tz_um", "ty_um", "tx_um"):
        cur = getattr(tr0, attr)
        ctrl_v.addWidget(_spin(attr, cur, cur - T_FINE, cur + T_FINE, 10.0, _setter(attr)))
    ctrl_v.addWidget(QLabel("<b>Rotation (deg, ±30 from current)</b>"))
    for attr in ("rz_deg", "ry_deg", "rx_deg"):
        cur = getattr(tr0, attr)
        ctrl_v.addWidget(_spin(attr, cur, cur - R_FINE, cur + R_FINE, 0.5, _setter(attr)))

    ctrl_v.addWidget(QLabel("<b>Flips</b>"))
    flip_row = QHBoxLayout()
    for axis in ("z", "y", "x"):
        cb = QCheckBox(axis); cb.setChecked(getattr(tr0, f"flip_{axis}"))
        def _mk(ax):
            return lambda checked: (setattr(box["transform"], f"flip_{ax}", bool(checked)),
                                     _refresh_label(), _schedule_redraw())
        cb.stateChanged.connect(_mk(axis))
        flip_row.addWidget(cb)
    flip_row.addStretch(1)
    fr = QWidget(); fr.setLayout(flip_row); ctrl_v.addWidget(fr)

    # --------- view-switch buttons ---------------------------------
    ctrl_v.addWidget(QLabel("<b>View</b>"))
    view_row = QHBoxLayout()
    def _set_view(order, ndisp):
        viewer.dims.ndisplay = ndisp
        viewer.dims.order = order
    for name, order in [("axial", (0, 1, 2)),
                         ("coronal", (1, 0, 2)),
                         ("sagittal", (2, 0, 1))]:
        b = QPushButton(name)
        b.clicked.connect(lambda _, o=order: _set_view(o, 2))
        view_row.addWidget(b)
    b3d = QPushButton("3D MIP")
    b3d.clicked.connect(lambda: _set_view((0, 1, 2), 3))
    view_row.addWidget(b3d)
    vw = QWidget(); vw.setLayout(view_row); ctrl_v.addWidget(vw)

    ctrl_v.addStretch(1)
    save_btn = QPushButton("Save state.json && exit")
    def _save_and_exit():
        box["transform"].center_um = box["center_um"]
        state.transform = box["transform"].to_dict()
        state.add_history("refine", "fine refine, single viewer with ortho swap")
        save_state(state, state_path)
        print(f"[step2] saved -> {state_path}", flush=True)
        viewer.close()
    save_btn.clicked.connect(_save_and_exit)
    ctrl_v.addWidget(save_btn)

    dock = viewer.window.add_dock_widget(ctrl, name="refine", area="right")
    try:
        dock.setMinimumWidth(240); dock.setMaximumWidth(900)
    except Exception:
        pass

    _refresh_label()
    print("\n[step2] single viewer. Use the View buttons (axial/coronal/sagittal/3D MIP)")
    print("[step2] to verify alignment across planes. Save when done.")
    napari.run()
