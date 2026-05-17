"""Step 3: landmark-based rigid fit.

Loads state.json (transform + crop from earlier steps). Single napari viewer.
Click "Add SAMPLE landmark" → next click in the canvas adds a landmark on
the (warped) sample. Click "Add CCF landmark" → next click adds one on the
gray CCF. Pair index = correspondence. Click "Fit" to run Procrustes — the
saved transform is updated and the prewarp redraws.

Landmarks AND transform are saved to state.json so you can quit and resume.
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
        raise RuntimeError("state.json has no transform — run `zrot step1` first.")
    _run_napari(state, state_path)


def _run_napari(state, state_path: Path) -> None:
    import napari
    from qtpy.QtCore import QTimer, Qt
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QListWidget, QGroupBox, QSizePolicy)
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    # --------- load sample + atlas with crop ----------------------------
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
        print(f"[step3] strided x{factor} for preview...", flush=True)
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    print(f"[step3] loading {arr.shape} into RAM...", flush=True)
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

    # --------- initial transform (with saved center) -------------------
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

    def _resample():
        M = box["transform"].for_voxel_grid(sample_um, ccf.voxel_um)
        Minv = np.linalg.inv(M)
        offset = Minv[:3, :3] @ crop_voxel_origin + Minv[:3, 3]
        return affine_transform(
            sample_np, Minv[:3, :3], offset=offset,
            output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
        )

    # --------- napari viewer + layers ----------------------------------
    def _percentiles(a, lo=1, hi=99.5):
        f = a.ravel()
        if f.size > 1_000_000: f = f[:: f.size // 1_000_000]
        f = f[f > 0] if (f > 0).any() else f
        a_lo, a_hi = np.percentile(f, [lo, hi])
        return float(a_lo), float(max(a_hi, a_lo + 1))

    viewer = napari.Viewer(ndisplay=2, title="zrot step3 — landmarks")
    viewer.add_image(ccf_data, name="CCF", scale=ccf.voxel_um, translate=ccf_origin,
                     colormap="gray", contrast_limits=_percentiles(ccf_data),
                     opacity=0.5, blending="additive",
                     interpolation2d="nearest", interpolation3d="nearest")
    sample_layer = viewer.add_image(
        _resample(), name="sample (live=saved)",
        scale=ccf.voxel_um, translate=ccf_origin,
        colormap="magenta", contrast_limits=_percentiles(sample_np),
        opacity=0.6, blending="additive",
        interpolation2d="nearest", interpolation3d="nearest")

    # Landmark layers (always visible)
    PHYS_LM_UM = max(80.0, ccf.voxel_um[0] * 3)   # visible dot size
    sample_lm_layer = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=(1.0, 1.0, 1.0),
        name="sample landmarks", face_color="cyan",
        border_color="white", size=PHYS_LM_UM,
    )
    ccf_lm_layer = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=ccf.voxel_um,
        name="ccf landmarks", face_color="yellow",
        border_color="white", size=PHYS_LM_UM / ccf.voxel_um[0],
    )

    # --------- in-memory landmark store --------------------------------
    # All coords in physical µm (sample_um in SAMPLE frame, ccf_um in CCF frame).
    landmarks_sample = [tuple(p) for p in state.landmarks.get("sample_um", [])]
    landmarks_ccf    = [tuple(p) for p in state.landmarks.get("ccf_um",    [])]

    def _refresh_sample_lm_display():
        if not landmarks_sample:
            sample_lm_layer.data = np.empty((0, 3)); return
        M = box["transform"].matrix()
        pts = np.asarray(landmarks_sample)
        world = (M[:3, :3] @ pts.T).T + M[:3, 3]
        sample_lm_layer.data = world

    def _refresh_ccf_lm_display():
        if not landmarks_ccf:
            ccf_lm_layer.data = np.empty((0, 3)); return
        pts = np.asarray(landmarks_ccf)
        ccf_lm_layer.data = pts / np.asarray(ccf.voxel_um)

    redraw_timer = QTimer(); redraw_timer.setSingleShot(True)
    def _redraw_sample():
        sample_layer.data = _resample()
        _refresh_sample_lm_display()
    redraw_timer.timeout.connect(_redraw_sample)
    def _schedule_redraw():
        redraw_timer.start(150)

    # --------- click capture for landmarks -----------------------------
    state_lm = {"pending": None}   # None | "sample" | "ccf"

    @viewer.mouse_drag_callbacks.append
    def _click_capture(_v, event):
        if state_lm["pending"] is None:
            return
        # event.position is in WORLD coords (z, y, x in µm) for 3D data
        slice_pos = list(viewer.dims.point)
        displayed = viewer.dims.displayed
        cursor = np.asarray(event.position, dtype=float)
        if cursor.size == len(displayed):
            for i, ax in enumerate(displayed):
                slice_pos[ax] = float(cursor[i])
        else:
            slice_pos = [float(c) for c in cursor]
        world = np.asarray(slice_pos)

        kind = state_lm["pending"]; state_lm["pending"] = None
        add_s_btn.setStyleSheet(""); add_c_btn.setStyleSheet("")

        if kind == "sample":
            # Map world (CCF µm) → sample µm via inverse current transform
            M = box["transform"].matrix()
            Minv = np.linalg.inv(M)
            s_pos = (Minv[:3, :3] @ world) + Minv[:3, 3]
            landmarks_sample.append(tuple(float(v) for v in s_pos))
            _refresh_sample_lm_display()
            print(f"[step3] + SAMPLE #{len(landmarks_sample)-1}: world={tuple(round(w,1) for w in world)} "
                  f"-> sample_µm={tuple(round(v,1) for v in s_pos)}", flush=True)
        else:
            landmarks_ccf.append(tuple(float(v) for v in world))
            _refresh_ccf_lm_display()
            print(f"[step3] + CCF    #{len(landmarks_ccf)-1}: ccf_µm={tuple(round(w,1) for w in world)}",
                  flush=True)
        _refresh_lists()

    # --------- controls panel -----------------------------------------
    ctrl = QWidget(); ctrl_v = QVBoxLayout(ctrl)
    ctrl_v.setContentsMargins(4, 4, 4, 4)

    ctrl_v.addWidget(QLabel("<b>Pick landmarks</b>"))
    btn_row = QHBoxLayout()
    add_s_btn = QPushButton("+ SAMPLE (next click)")
    add_c_btn = QPushButton("+ CCF (next click)")
    def _arm(kind):
        state_lm["pending"] = kind
        viewer.status = f"click on a {kind.upper()} feature in the canvas..."
        if kind == "sample": add_s_btn.setStyleSheet("background:#225588;")
        else: add_c_btn.setStyleSheet("background:#886600;")
    add_s_btn.clicked.connect(lambda: _arm("sample"))
    add_c_btn.clicked.connect(lambda: _arm("ccf"))
    btn_row.addWidget(add_s_btn); btn_row.addWidget(add_c_btn)
    bw = QWidget(); bw.setLayout(btn_row); ctrl_v.addWidget(bw)

    # Two lists
    lists_row = QHBoxLayout()
    sample_list = QListWidget(); sample_list.setMaximumHeight(150)
    ccf_list    = QListWidget(); ccf_list.setMaximumHeight(150)

    def _refresh_lists():
        sample_list.clear(); ccf_list.clear()
        for i, (z, y, x) in enumerate(landmarks_sample):
            sample_list.addItem(f"#{i} z={z:+.0f} y={y:+.0f} x={x:+.0f}")
        for i, (z, y, x) in enumerate(landmarks_ccf):
            ccf_list.addItem(f"#{i} z={z:+.0f} y={y:+.0f} x={x:+.0f}")

    sbox = QGroupBox("sample (µm)"); sb = QVBoxLayout(sbox)
    sb.setContentsMargins(2,2,2,2); sb.addWidget(sample_list)
    sdel = QPushButton("Delete selected"); sb.addWidget(sdel)
    cbox = QGroupBox("CCF (µm)"); cb = QVBoxLayout(cbox)
    cb.setContentsMargins(2,2,2,2); cb.addWidget(ccf_list)
    cdel = QPushButton("Delete selected"); cb.addWidget(cdel)
    lists_row.addWidget(sbox); lists_row.addWidget(cbox)
    lw = QWidget(); lw.setLayout(lists_row); ctrl_v.addWidget(lw)

    def _del(which):
        lst, store, refresh = ((sample_list, landmarks_sample, _refresh_sample_lm_display)
                                if which == "sample" else
                                (ccf_list, landmarks_ccf, _refresh_ccf_lm_display))
        rows = sorted({i.row() for i in lst.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(store): store.pop(r)
        refresh(); _refresh_lists()
    sdel.clicked.connect(lambda: _del("sample"))
    cdel.clicked.connect(lambda: _del("ccf"))

    # Fit button
    fit_btn = QPushButton("Fit rigid from landmarks (≥3 pairs)")
    def _fit():
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 3:
            viewer.status = f"need ≥3 pairs (sample={len(landmarks_sample)}, ccf={len(landmarks_ccf)})"
            return
        sp = np.asarray(landmarks_sample[:n]); cp = np.asarray(landmarks_ccf[:n])
        # If flips are currently on, pre-apply them to source (Procrustes solves rotation only).
        F = np.diag([
            -1.0 if box["transform"].flip_z else 1.0,
            -1.0 if box["transform"].flip_y else 1.0,
            -1.0 if box["transform"].flip_x else 1.0,
        ])
        c = np.asarray(box["center_um"])
        sp_F = (sp - c) * np.diag(F) + c
        src_c = sp_F.mean(0); tgt_c = cp.mean(0)
        H = (sp_F - src_c).T @ (cp - tgt_c)
        U, S, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = tgt_c - R @ src_c
        rms = float(np.sqrt(np.mean(np.sum(((R @ sp_F.T).T + t - cp) ** 2, axis=1))))
        per = np.sqrt(np.sum(((R @ sp_F.T).T + t - cp) ** 2, axis=1))
        rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)

        # Convert (R, t) → (rz, ry, rx, tz, ty, tx) given pivot c:
        #   M @ p = R @ p + t = R @ (p−c) + c + (tz, ty, tx)
        #   ⇒ (tz, ty, tx) = t + R @ c − c
        offset = R @ c - c
        new_tr = RigidTransform(
            rz_deg=float(rz), ry_deg=float(ry), rx_deg=float(rx),
            tz_um=float(t[0] + offset[0]),
            ty_um=float(t[1] + offset[1]),
            tx_um=float(t[2] + offset[2]),
            flip_z=box["transform"].flip_z,
            flip_y=box["transform"].flip_y,
            flip_x=box["transform"].flip_x,
            center_um=box["center_um"],
        )
        box["transform"] = new_tr
        sample_layer.data = _resample()
        _refresh_sample_lm_display()

        print(f"\n[step3] fit: {n} pairs, RMS = {rms:.1f} µm")
        print(f"  rotation:    rz={rz:+7.2f}  ry={ry:+7.2f}  rx={rx:+7.2f} deg")
        print(f"  translation: tz={new_tr.tz_um:+9.1f}  ty={new_tr.ty_um:+9.1f}  tx={new_tr.tx_um:+9.1f} µm")
        for i, r in enumerate(per):
            print(f"  pair {i:2d}: {r:7.1f} µm" + ("  ← outlier?" if r > 3*rms else ""))
        viewer.status = f"fit: {n} pairs, RMS {rms:.0f} µm — see terminal for per-pair"
    fit_btn.clicked.connect(_fit)
    ctrl_v.addWidget(fit_btn)

    clear_btn = QPushButton("Clear ALL landmarks")
    def _clear_all():
        landmarks_sample.clear(); landmarks_ccf.clear()
        _refresh_sample_lm_display(); _refresh_ccf_lm_display(); _refresh_lists()
    clear_btn.clicked.connect(_clear_all)
    ctrl_v.addWidget(clear_btn)

    # View buttons
    ctrl_v.addWidget(QLabel("<b>View</b>"))
    view_row = QHBoxLayout()
    def _set_view(order, ndisp):
        viewer.dims.ndisplay = ndisp; viewer.dims.order = order
    for name, order in [("axial", (0,1,2)), ("coronal", (1,0,2)), ("sagittal", (2,0,1))]:
        b = QPushButton(name); b.clicked.connect(lambda _, o=order: _set_view(o, 2))
        view_row.addWidget(b)
    b3 = QPushButton("3D MIP"); b3.clicked.connect(lambda: _set_view((0,1,2), 3))
    view_row.addWidget(b3)
    vw = QWidget(); vw.setLayout(view_row); ctrl_v.addWidget(vw)

    ctrl_v.addStretch(1)
    save_btn = QPushButton("Save state.json && exit")
    def _save_and_exit():
        box["transform"].center_um = box["center_um"]
        state.transform = box["transform"].to_dict()
        state.landmarks = {
            "sample_um": [list(p) for p in landmarks_sample],
            "ccf_um":    [list(p) for p in landmarks_ccf],
        }
        state.add_history("step3",
                          f"{len(landmarks_sample)} sample, {len(landmarks_ccf)} CCF landmarks")
        save_state(state, state_path)
        print(f"[step3] saved -> {state_path}", flush=True)
        viewer.close()
    save_btn.clicked.connect(_save_and_exit)
    ctrl_v.addWidget(save_btn)

    dock = viewer.window.add_dock_widget(ctrl, name="step3", area="right")
    try:
        dock.setMinimumWidth(280); dock.setMaximumWidth(900)
    except Exception:
        pass

    _refresh_sample_lm_display(); _refresh_ccf_lm_display(); _refresh_lists()
    print(f"\n[step3] loaded {len(landmarks_sample)} sample + {len(landmarks_ccf)} ccf landmarks")
    print("[step3] workflow: '+ SAMPLE (next click)' → click on a sample feature.")
    print("[step3]            '+ CCF (next click)' → click on the matching CCF feature.")
    print("[step3] same row number = corresponding pair.")
    print("[step3] 'Fit' runs Procrustes; the prewarp updates with the new transform.")
    print("[step3] save when done; reopen step3 to add more landmarks later.")
    napari.run()
