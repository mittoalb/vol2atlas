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
from ..frame import compute_output_frame, enable_ccf_axes, extract_ccf
from ..io import open_multiscale
from ..state import save as save_state
from ..transform import RigidTransform


def run(state_path: Path, level: int | None = None,
        preview_size: int = 192) -> None:
    from ..state import load
    state = load(state_path)
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run `vol2atlas prealign` first.")
    _run_napari(state, state_path, level=level, preview_size=preview_size)


def _run_napari(state, state_path: Path, *,
                level: int | None = None,
                preview_size: int = 192) -> None:
    import napari
    from qtpy.QtCore import QTimer, Qt
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QListWidget, QGroupBox, QSizePolicy)
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    # --------- load sample + atlas with crop ----------------------------
    ms = open_multiscale(state.sample_zarr)
    use_level = state.sample_level if level is None else int(level)
    arr = ms.level(use_level)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    # NOTE: state.sample_voxel_um is the LEVEL-0 voxel size override.
    # For a different level, scale by the pyramid factor.
    if state.sample_voxel_um is not None and use_level != state.sample_level:
        anc = ms.spacing(state.sample_level)
        cur = ms.spacing(use_level)
        sample_um = tuple(v * (c / a) for v, a, c
                           in zip(state.sample_voxel_um, anc, cur))
    elif state.sample_voxel_um is not None:
        sample_um = tuple(state.sample_voxel_um)
    else:
        sample_um = ms.spacing(use_level)
    print(f"[step3] level {use_level}: {arr.shape} @ {sample_um} µm",
          flush=True)

    PREVIEW_MAX = int(preview_size) ** 3
    n_vox = int(np.prod(arr.shape))
    if n_vox > PREVIEW_MAX:
        factor = max(1, int(np.ceil((n_vox / PREVIEW_MAX) ** (1 / 3))))
        print(f"[step3] strided x{factor} for preview "
              f"(budget {preview_size}^3 = {PREVIEW_MAX/1e6:.1f}M voxels)...",
              flush=True)
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    print(f"[step3] loading {arr.shape} into RAM...", flush=True)
    sample_np = np.ascontiguousarray(arr.compute())
    sample_center_um = tuple((s - 1) * v / 2.0 for s, v in zip(sample_np.shape, sample_um))

    ccf = load_ccf(state.atlas_name)
    ccf_ref_full = np.asarray(ccf.reference)

    # --------- initial transform (with saved center) -------------------
    saved_center = tuple(state.transform.get("center_um")) \
        if state.transform.get("center_um") is not None \
        else sample_center_um
    tr0 = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        center_um=saved_center,
    )
    # Hydrate local refinements from state (list of LocalRefinement dataclass
    # instances; persisted back as dicts in the Save handlers).
    from ..local_refinement import LocalRefinement
    box = {"transform": tr0, "center_um": saved_center,
           "ccf_data": None, "ccf_origin_um": None,
           "affine_matrix": (np.asarray(state.affine, dtype=float)
                              if state.affine is not None else None),
           "local_refinements": [LocalRefinement.from_dict(d)
                                  for d in (state.local_refinements or [])],
           # Undo stack: list of (transform_dict, affine_matrix-or-None)
           # snapshots, pushed BEFORE each Fit operation.
           "undo_stack": []}

    def _snapshot():
        """Push current (transform, affine) onto the undo stack."""
        snap_tr = (
            box["transform"].rz_deg, box["transform"].ry_deg, box["transform"].rx_deg,
            box["transform"].tz_um, box["transform"].ty_um, box["transform"].tx_um,
            tuple(box["center_um"]),
        )
        snap_aff = (None if box["affine_matrix"] is None
                    else box["affine_matrix"].copy())
        box["undo_stack"].append((snap_tr, snap_aff))

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
    ccf_data = box["ccf_data"]; ccf_origin = box["ccf_origin_um"]
    b = state.ccf_crop_bbox
    crop_voxel_origin = (np.array([b["z"][0], b["y"][0], b["x"][0]], dtype=float)
                         if b is not None else np.zeros(3))

    # ----- TPS preview state ------------------------------------------
    # When tps_enabled is True AND ≥4 landmark pairs are present,
    # _resample adds a TPS correction on top of the affine baseline,
    # using the same residuals-on-rigid pattern as step_export.run with
    # --tps. The fit cost is O(N^3) — milliseconds for typical landmark
    # counts (<100) — so we refit on every _resample call rather than
    # tracking invalidation across all the sites that mutate landmarks
    # or transforms.
    box["tps_enabled"] = False
    box["tps_smoothing"] = 0.0

    def _invalidate_tps():
        # Kept for API symmetry; the no-cache implementation makes it a
        # no-op. Call sites mark "TPS state changed; please redraw" by
        # also doing sample_layer.data = _resample().
        pass

    def _fit_tps():
        """Fit RBFInterpolator on landmark residuals (in sample µm)
        against the CURRENT baseline transform. Returns None if fewer
        than 4 pairs or if the fit fails."""
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 4:
            return None
        from scipy.interpolate import RBFInterpolator
        sp = np.asarray(landmarks_sample[:n], dtype=float)
        cp = np.asarray(landmarks_ccf[:n], dtype=float)
        M_base = box["transform"].matrix()
        if box.get("affine_matrix") is not None:
            M_base = box["affine_matrix"] @ M_base
        M_base_inv = np.linalg.inv(M_base)
        baseline_pred_sample = (M_base_inv[:3, :3] @ cp.T).T + M_base_inv[:3, 3]
        residuals = sp - baseline_pred_sample
        try:
            return RBFInterpolator(
                cp, residuals,
                smoothing=float(box["tps_smoothing"]),
                kernel="thin_plate_spline",
            )
        except Exception as e:
            print(f"[landmarks] TPS fit failed: {e} — disabling TPS preview")
            box["tps_enabled"] = False
            return None

    def _resample():
        # Combined µm transform = (state.affine if any) @ rigid
        M_um = box["transform"].matrix()
        if box.get("affine_matrix") is not None:
            M_um = box["affine_matrix"] @ M_um
        # µm → voxel (sample voxel → CCF crop voxel)
        S_in   = np.diag([sample_um[0], sample_um[1], sample_um[2], 1.0])
        S_out  = np.diag([1.0 / ccf.voxel_um[0], 1.0 / ccf.voxel_um[1],
                          1.0 / ccf.voxel_um[2], 1.0])
        M = S_out @ M_um @ S_in
        Minv = np.linalg.inv(M)
        offset = Minv[:3, :3] @ crop_voxel_origin + Minv[:3, 3]

        local_refs = box.get("local_refinements") or []
        if not box.get("tps_enabled") and not local_refs:
            return affine_transform(
                sample_np, Minv[:3, :3], offset=offset,
                output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
            )

        # If we have local refinements (with or without TPS), use the
        # blended_inverse_sample_coords path so the preview matches what
        # export / alignFull produce. TPS, when enabled, is layered on
        # top of the blended sample coords.
        if local_refs:
            from scipy.ndimage import map_coordinates
            from ..local_refinement import blended_inverse_sample_coords
            tps_fn = _fit_tps() if box.get("tps_enabled") else None
            out = np.empty(ccf_data.shape, dtype=sample_np.dtype)
            sample_um_arr = np.asarray(sample_um, dtype=float).reshape(3, 1, 1, 1)
            ccf_voxel_um_arr = np.asarray(ccf.voxel_um, dtype=float).reshape(3, 1, 1, 1)
            crop_origin_um_local = tuple(
                np.asarray(crop_voxel_origin, dtype=float) *
                np.asarray(ccf.voxel_um, dtype=float))
            z_step = max(1, min(16, ccf_data.shape[0]))
            for z0 in range(0, ccf_data.shape[0], z_step):
                z1 = min(z0 + z_step, ccf_data.shape[0])
                zz, yy, xx = np.meshgrid(
                    np.arange(z0, z1, dtype=np.float32),
                    np.arange(ccf_data.shape[1], dtype=np.float32),
                    np.arange(ccf_data.shape[2], dtype=np.float32),
                    indexing="ij",
                )
                coords = np.stack([zz, yy, xx], axis=0)
                smp_vox = blended_inverse_sample_coords(
                    coords, M_um, sample_um, ccf.voxel_um,
                    crop_origin_um_local,
                    local_refinements=local_refs,
                )
                if tps_fn is not None:
                    # Add TPS shift in µm on top of the blended source µm.
                    ccf_um = (coords + crop_voxel_origin.reshape(3, 1, 1, 1)) \
                              * ccf_voxel_um_arr
                    ccf_um_pts = np.moveaxis(ccf_um, 0, -1).reshape(-1, 3)
                    try:
                        shifts = tps_fn(ccf_um_pts).reshape(
                            z1 - z0, ccf_data.shape[1], ccf_data.shape[2], 3)
                        shifts = np.moveaxis(shifts, -1, 0)
                        smp_vox = smp_vox + shifts / sample_um_arr
                    except Exception as e:
                        print(f"[landmarks] TPS eval failed: {e}")
                out[z0:z1] = map_coordinates(
                    sample_np, smp_vox, order=0, mode="constant", cval=0,
                ).astype(sample_np.dtype)
            return out

        # TPS-on-affine-residuals path (no local refinements)
        tps_fn = _fit_tps()
        if tps_fn is None:
            return affine_transform(
                sample_np, Minv[:3, :3], offset=offset,
                output_shape=ccf_data.shape, order=0, mode="constant", cval=0,
            )
        from scipy.ndimage import map_coordinates
        # Build per-voxel sample coords: affine baseline + TPS(ccf_um).
        # Work in chunks along z to bound peak memory.
        out = np.empty(ccf_data.shape, dtype=sample_np.dtype)
        sample_um_arr = np.asarray(sample_um, dtype=float).reshape(3, 1, 1, 1)
        ccf_voxel_um_arr = np.asarray(ccf.voxel_um, dtype=float).reshape(3, 1, 1, 1)
        M_inv = np.linalg.inv(M_um)
        A_inv = M_inv[:3, :3]; t_inv = M_inv[:3, 3].reshape(3, 1, 1, 1)
        z_step = max(1, min(16, ccf_data.shape[0]))
        for z0 in range(0, ccf_data.shape[0], z_step):
            z1 = min(z0 + z_step, ccf_data.shape[0])
            zz, yy, xx = np.meshgrid(
                np.arange(z0, z1, dtype=np.float32),
                np.arange(ccf_data.shape[1], dtype=np.float32),
                np.arange(ccf_data.shape[2], dtype=np.float32),
                indexing="ij",
            )
            coords_vox = np.stack([zz, yy, xx], axis=0)  # (3, dz, Y, X)
            # CCF voxel → CCF µm (with crop origin)
            ccf_um = (coords_vox + crop_voxel_origin.reshape(3, 1, 1, 1)) \
                      * ccf_voxel_um_arr
            # Baseline source µm = M_inv @ ccf_um
            baseline_smp_um = np.tensordot(A_inv, ccf_um, axes=(1, 0)) + t_inv
            # TPS correction is evaluated on CCF µm query points; returns
            # sample-µm shift to add. Reshape ccf_um to (N, 3) for the call.
            ccf_um_pts = np.moveaxis(ccf_um, 0, -1).reshape(-1, 3)
            try:
                shifts = tps_fn(ccf_um_pts).reshape(
                    z1 - z0, ccf_data.shape[1], ccf_data.shape[2], 3)
            except Exception as e:
                print(f"[landmarks] TPS eval failed: {e} — falling back to "
                      f"affine for this redraw")
                return affine_transform(
                    sample_np, Minv[:3, :3], offset=offset,
                    output_shape=ccf_data.shape, order=0,
                    mode="constant", cval=0,
                )
            shifts = np.moveaxis(shifts, -1, 0)  # (3, dz, Y, X)
            smp_um = baseline_smp_um + shifts
            smp_vox = smp_um / sample_um_arr
            out[z0:z1] = map_coordinates(
                sample_np, smp_vox, order=0, mode="constant", cval=0,
            ).astype(sample_np.dtype)
        return out

    # --------- napari viewer + layers ----------------------------------
    def _percentiles(a, lo=1, hi=99.5):
        f = a.ravel()
        if f.size > 1_000_000: f = f[:: f.size // 1_000_000]
        f = f[f > 0] if (f > 0).any() else f
        a_lo, a_hi = np.percentile(f, [lo, hi])
        return float(a_lo), float(max(a_hi, a_lo + 1))

    viewer = napari.Viewer(ndisplay=2, title="vol2atlas landmarks — landmarks")
    enable_ccf_axes(viewer, ccf.orientation)
    ccf_layer = viewer.add_image(
        ccf_data, name="CCF", scale=ccf.voxel_um, translate=ccf_origin,
        colormap="gray", contrast_limits=_percentiles(ccf_data),
        opacity=0.5, blending="additive",
        interpolation2d="nearest", interpolation3d="nearest")
    sample_layer = viewer.add_image(
        _resample(), name="sample (live=saved)",
        scale=ccf.voxel_um, translate=ccf_origin,
        colormap="gray", contrast_limits=_percentiles(sample_np),
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

    def _M_full():
        """Combined sample µm → world (= CCF µm) transform.
        MUST match what _resample uses for the sample image, else
        landmark dots drift off the sample volume."""
        M = box["transform"].matrix()
        if box.get("affine_matrix") is not None:
            M = box["affine_matrix"] @ M
        return M

    def _refresh_sample_lm_display():
        if not landmarks_sample:
            sample_lm_layer.data = np.empty((0, 3)); return
        M = _M_full()
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
            # Map world (CCF µm) → raw sample µm via inverse of the FULL
            # transform that _resample applies to render the sample volume.
            # Using rigid-only here while _resample uses (affine @ rigid)
            # makes landmarks captured AFTER an affine fit live in a
            # different reference frame than landmarks captured BEFORE —
            # LSQ on a mixed-frame set produces garbage.
            M = _M_full()
            Minv = np.linalg.inv(M)
            s_pos = (Minv[:3, :3] @ world) + Minv[:3, 3]
            landmarks_sample.append(tuple(float(v) for v in s_pos))
            _refresh_sample_lm_display()
            print(f"[step3] + SAMPLE #{len(landmarks_sample)-1}: world={tuple(round(w,1) for w in world)} "
                  f"-> sample_µm={tuple(round(v,1) for v in s_pos)} "
                  f"(affine={'YES' if box.get('affine_matrix') is not None else 'no'})",
                  flush=True)
        else:
            landmarks_ccf.append(tuple(float(v) for v in world))
            _refresh_ccf_lm_display()
            print(f"[step3] + CCF    #{len(landmarks_ccf)-1}: ccf_µm={tuple(round(w,1) for w in world)}",
                  flush=True)
        _refresh_lists()
        # If TPS preview is enabled, refit on the new landmark set and
        # re-render. Cheap for typical landmark counts (<100).
        _invalidate_tps()
        if box.get("tps_enabled"):
            sample_layer.data = _resample()

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
        _invalidate_tps()
        if box.get("tps_enabled"):
            sample_layer.data = _resample()
    sdel.clicked.connect(lambda: _del("sample"))
    cdel.clicked.connect(lambda: _del("ccf"))

    # Fit button
    fit_btn = QPushButton("Fit rigid from landmarks (≥3 pairs)")
    def _fit():
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 3:
            viewer.status = f"need ≥3 pairs (sample={len(landmarks_sample)}, ccf={len(landmarks_ccf)})"
            return
        _snapshot()
        sp = np.asarray(landmarks_sample[:n]); cp = np.asarray(landmarks_ccf[:n])
        c = np.asarray(box["center_um"])
        src_c = sp.mean(0); tgt_c = cp.mean(0)
        H = (sp - src_c).T @ (cp - tgt_c)
        U, S, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = tgt_c - R @ src_c
        rms = float(np.sqrt(np.mean(np.sum(((R @ sp.T).T + t - cp) ** 2, axis=1))))
        per = np.sqrt(np.sum(((R @ sp.T).T + t - cp) ** 2, axis=1))
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

    # AFFINE fit button — 12 DOF least-squares from landmark pairs
    fit_aff_btn = QPushButton("Fit AFFINE from landmarks (≥4 pairs)")
    def _fit_affine():
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 4:
            viewer.status = (f"need ≥4 non-coplanar pairs for affine "
                             f"(have {n})")
            return
        _snapshot()
        sp = np.asarray(landmarks_sample[:n], dtype=float)  # sample µm
        cp = np.asarray(landmarks_ccf[:n],    dtype=float)  # CCF µm
        # Least-squares affine: solve  M @ [sp; 1] = cp
        # H = [sp, 1]_Nx4 ; A = H^+ @ cp  -> A is 4x3
        H = np.hstack([sp, np.ones((n, 1))])
        A_lstsq, *_ = np.linalg.lstsq(H, cp, rcond=None)
        # Per-pair residuals
        cp_pred = H @ A_lstsq
        per = np.linalg.norm(cp_pred - cp, axis=1)
        rms = float(np.sqrt(np.mean(per ** 2)))
        # Build full 4x4 affine matrix: M[:3, :3] @ sp + M[:3, 3] = cp
        # From A_lstsq (4x3): cp_i = H_i @ A_lstsq = sp_i @ A[:3, :] + A[3, :]
        M_full = np.eye(4)
        M_full[:3, :3] = A_lstsq[:3, :].T
        M_full[:3,  3] = A_lstsq[3, :]
        # Polar decomposition of M[:3, :3] = R @ S (R rotation, S stretch)
        # gives a rough scale read-out for the user.
        U, sv, Vt = np.linalg.svd(M_full[:3, :3])
        # Save: state.affine = the full 4x4 (sample µm → CCF µm)
        #       state.transform = identity rigid (the affine carries the
        #       whole transform, no rigid pre-step needed)
        box["transform"] = RigidTransform(center_um=box["center_um"])
        box["affine_matrix"] = M_full
        sample_layer.data = _resample()
        _refresh_sample_lm_display()

        print(f"\n[landmarks] AFFINE fit: {n} pairs, RMS = {rms:.1f} µm")
        print(f"  singular values (per-axis scale): "
              f"{sv[0]:.4f}  {sv[1]:.4f}  {sv[2]:.4f}  (1.0 = no scale)")
        print(f"  translation (µm): "
              f"{M_full[0,3]:+9.1f}  {M_full[1,3]:+9.1f}  {M_full[2,3]:+9.1f}")
        for i, r in enumerate(per):
            print(f"  pair {i:2d}: {r:7.1f} µm"
                  + ("  ← outlier?" if r > 3 * rms else ""))
        viewer.status = (f"affine fit: {n} pairs, RMS {rms:.0f} µm — see terminal")
    fit_aff_btn.clicked.connect(_fit_affine)
    ctrl_v.addWidget(fit_aff_btn)

    # --------- TPS live preview ---------------------------------------
    # When enabled, the sample image layer is rendered with a
    # thin-plate-spline correction layered on top of the affine
    # baseline (residuals at landmark positions are interpolated via
    # scipy RBFInterpolator). Recomputed on every redraw — refit cost
    # is ~ms for typical landmark counts.
    from qtpy.QtWidgets import (QDoubleSpinBox as _QDoubleSpinBox,
                                 QCheckBox as _QCheckBox)
    tps_row = QHBoxLayout()
    cb_tps = _QCheckBox("Live TPS preview")
    tps_row.addWidget(cb_tps)
    tps_row.addWidget(QLabel("smoothing:"))
    tps_sb = _QDoubleSpinBox()
    tps_sb.setRange(0.0, 1e6); tps_sb.setDecimals(2); tps_sb.setSingleStep(1.0)
    tps_sb.setValue(0.0)
    tps_row.addWidget(tps_sb)
    tps_row.addStretch(1)
    tw = QWidget(); tw.setLayout(tps_row); ctrl_v.addWidget(tw)

    def _on_tps_toggled(checked):
        box["tps_enabled"] = bool(checked)
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if checked and n < 4:
            viewer.status = (f"TPS preview needs ≥4 landmark pairs "
                              f"(have {n}); will activate once you have enough")
        else:
            viewer.status = (f"TPS preview {'ON' if checked else 'OFF'}")
        sample_layer.data = _resample()
    cb_tps.stateChanged.connect(_on_tps_toggled)

    def _on_tps_smoothing(v):
        box["tps_smoothing"] = float(v)
        if box.get("tps_enabled"):
            sample_layer.data = _resample()
    tps_sb.valueChanged.connect(_on_tps_smoothing)

    # --------- Local refinements (masked transforms) ------------------
    # Multi-select rows in the sample/CCF lists, choose name + falloff,
    # click "Add local refinement". List below shows existing ones with
    # Remove buttons. Preview re-renders so you see the masked
    # correction immediately.
    from qtpy.QtWidgets import (QLineEdit as _QLineEdit,
                                 QDoubleSpinBox as _QDSpin,
                                 QListWidget as _QListW)
    from ..local_refinement import fit_from_landmarks
    ctrl_v.addWidget(QLabel("<b>Local refinement (masked affine)</b>"))
    lr_name_row = QHBoxLayout()
    lr_name_row.addWidget(QLabel("name:"))
    lr_name_edit = _QLineEdit(); lr_name_edit.setPlaceholderText("e.g. left_lobe")
    lr_name_row.addWidget(lr_name_edit)
    lr_name_row.addWidget(QLabel("falloff µm:"))
    lr_falloff_sb = _QDSpin(); lr_falloff_sb.setRange(10.0, 5000.0)
    lr_falloff_sb.setDecimals(0); lr_falloff_sb.setSingleStep(50.0)
    lr_falloff_sb.setValue(300.0)
    lr_name_row.addWidget(lr_falloff_sb)
    lr_name_row.addWidget(QLabel("pad µm:"))
    lr_pad_sb = _QDSpin(); lr_pad_sb.setRange(0.0, 5000.0)
    lr_pad_sb.setDecimals(0); lr_pad_sb.setSingleStep(50.0); lr_pad_sb.setValue(200.0)
    lr_name_row.addWidget(lr_pad_sb)
    lrn_w = QWidget(); lrn_w.setLayout(lr_name_row); ctrl_v.addWidget(lrn_w)

    lr_add_btn = QPushButton(
        "Add local refinement from selected landmark rows (≥4)")
    lr_list_w = _QListW(); lr_list_w.setMaximumHeight(100)
    lr_remove_btn = QPushButton("Remove selected local refinement")
    ctrl_v.addWidget(lr_add_btn)
    ctrl_v.addWidget(lr_list_w)
    ctrl_v.addWidget(lr_remove_btn)

    def _refresh_lr_list():
        lr_list_w.clear()
        for L in box["local_refinements"]:
            c = L.center_um
            lr_list_w.addItem(
                f"{L.name}: center=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) µm "
                f"radius={L.radius_um:.0f} fall={L.falloff_um:.0f} "
                f"lms={L.landmark_indices}")
    _refresh_lr_list()

    def _selected_landmark_indices():
        rows = sorted({i.row() for i in sample_list.selectedIndexes()}
                       | {i.row() for i in ccf_list.selectedIndexes()})
        n_total = min(len(landmarks_sample), len(landmarks_ccf))
        return [r for r in rows if 0 <= r < n_total]

    def _add_local_refinement():
        name = lr_name_edit.text().strip()
        if not name:
            viewer.status = "give the local refinement a name"
            return
        if any(L.name == name for L in box["local_refinements"]):
            viewer.status = f"name {name!r} already used; pick another"
            return
        idx = _selected_landmark_indices()
        if len(idx) < 4:
            viewer.status = (f"select ≥4 landmark rows in the sample OR "
                              f"CCF list (have {len(idx)})")
            return
        sp = np.asarray([landmarks_sample[i] for i in idx], dtype=float)
        cp = np.asarray([landmarks_ccf[i]    for i in idx], dtype=float)
        try:
            L = fit_from_landmarks(
                sp, cp, name=name,
                falloff_um=float(lr_falloff_sb.value()),
                radius_pad_um=float(lr_pad_sb.value()),
                landmark_indices=idx,
            )
        except Exception as e:
            viewer.status = f"local refinement fit failed: {e}"
            return
        box["local_refinements"].append(L)
        S = np.linalg.svd(L.affine[:3, :3], compute_uv=False).tolist()
        print(f"[landmarks] added local refinement {name!r}: "
              f"{len(idx)} lms, center {L.center_um}, "
              f"radius {L.radius_um:.0f} µm, fall {L.falloff_um:.0f} µm, "
              f"SVD {S}", flush=True)
        lr_name_edit.clear()
        _refresh_lr_list()
        sample_layer.data = _resample()
        viewer.status = (f"added local refinement {name!r} "
                          f"({len(idx)} landmarks); preview updated")
    lr_add_btn.clicked.connect(_add_local_refinement)

    def _remove_local_refinement():
        rows = sorted({i.row() for i in lr_list_w.selectedIndexes()},
                       reverse=True)
        if not rows:
            viewer.status = "select a local refinement row to remove"
            return
        for r in rows:
            if 0 <= r < len(box["local_refinements"]):
                removed = box["local_refinements"].pop(r)
                print(f"[landmarks] removed local refinement "
                      f"{removed.name!r}", flush=True)
        _refresh_lr_list()
        sample_layer.data = _resample()
    lr_remove_btn.clicked.connect(_remove_local_refinement)

    # Revert: undo the last fit (pops one snapshot off the stack)
    revert_btn = QPushButton("Revert last fit")
    def _revert():
        if not box["undo_stack"]:
            viewer.status = "nothing to revert (no fit applied this session)"
            return
        snap_tr, snap_aff = box["undo_stack"].pop()
        (rz, ry, rx, tz, ty, tx, c_um) = snap_tr
        box["transform"] = RigidTransform(
            rz_deg=float(rz), ry_deg=float(ry), rx_deg=float(rx),
            tz_um=float(tz), ty_um=float(ty), tx_um=float(tx),
            center_um=tuple(c_um),
        )
        box["affine_matrix"] = snap_aff
        sample_layer.data = _resample()
        _refresh_sample_lm_display()
        depth = len(box["undo_stack"])
        viewer.status = (f"reverted ({depth} undo step{'s' if depth != 1 else ''} left)")
        print(f"[landmarks] reverted last fit ({depth} undo step{'s' if depth != 1 else ''} left)",
              flush=True)
    revert_btn.clicked.connect(_revert)
    ctrl_v.addWidget(revert_btn)

    # --------- Auto MI registration buttons ---------------------------
    # Run ANTs MI (rigid/affine, intensity/shape) WITH current state +
    # landmarks already in place. Updates state.transform / state.affine
    # via step_mi.run, then reloads & refreshes the preview.
    from qtpy.QtWidgets import QCheckBox
    ctrl_v.addWidget(QLabel("<b>Auto MI registration</b>"))
    mi_opts_row = QHBoxLayout()
    cb_shape  = QCheckBox("SHAPE (SDT)")
    cb_affine = QCheckBox("affine (12-DOF)")
    cb_mask   = QCheckBox("ANTs mask")
    mi_opts_row.addWidget(cb_shape); mi_opts_row.addWidget(cb_affine)
    mi_opts_row.addWidget(cb_mask)
    mw = QWidget(); mw.setLayout(mi_opts_row); ctrl_v.addWidget(mw)

    def _build_in_memory_state():
        """Snapshot current GUI state into an in-memory State object.
        Does NOT touch state.json on disk. The MI/JOINT functions take
        this and return a modified State; the GUI then updates box from
        the return value. state.json only changes when the user clicks
        an explicit Save button.
        """
        box["transform"].center_um = box["center_um"]
        s = state              # reuse the loaded State (mutates it in-memory)
        s.transform = box["transform"].to_dict()
        s.affine = (box["affine_matrix"].tolist()
                    if box.get("affine_matrix") is not None else None)
        s.landmarks = {
            "sample_um": [list(p) for p in landmarks_sample],
            "ccf_um":    [list(p) for p in landmarks_ccf],
        }
        return s

    def _apply_returned_state(new_state):
        """Pull transform/affine from a returned State into box. No disk I/O."""
        new_tr = new_state.transform
        new_aff = new_state.affine
        saved_c = (tuple(new_tr.get("center_um"))
                    if new_tr.get("center_um") is not None
                    else box["center_um"])
        box["transform"] = RigidTransform(
            **{k: new_tr[k] for k in
               ["rz_deg","ry_deg","rx_deg","tz_um","ty_um","tx_um"]},
            center_um=saved_c,
        )
        box["center_um"] = saved_c
        box["affine_matrix"] = (np.asarray(new_aff, dtype=float)
                                 if new_aff is not None else None)

    run_mi_btn = QPushButton("Run MI now")
    def _run_mi():
        _snapshot()
        in_state = _build_in_memory_state()
        from .step_mi import run as run_mi
        viewer.status = "Running ANTs MI... (see terminal)"
        print("[landmarks] launching MI in-memory: "
              f"shape={cb_shape.isChecked()}, affine={cb_affine.isChecked()}, "
              f"mask={cb_mask.isChecked()} (state.json untouched)", flush=True)
        try:
            new_state = run_mi(in_state,
                               shape=cb_shape.isChecked(),
                               affine=cb_affine.isChecked(),
                               mask=cb_mask.isChecked(),
                               skip_view=True,
                               check_mask=False)
        except SystemExit as e:
            print(f"[landmarks] MI aborted: {e}", flush=True)
            viewer.status = f"MI aborted: {e}"
            return
        _apply_returned_state(new_state)
        sample_layer.data = _resample()
        _refresh_sample_lm_display()
        viewer.status = "MI done — preview updated. Click Revert (or Save to keep)."
        print("[landmarks] MI done. state.json unchanged until you Save.",
              flush=True)
    run_mi_btn.clicked.connect(_run_mi)
    ctrl_v.addWidget(run_mi_btn)

    # JOINT button — Mattes MI + landmark PSE in a single
    # antsRegistration optimization. Uses the SHAPE/affine/mask
    # checkboxes above + a landmark weight λ.
    from qtpy.QtWidgets import QDoubleSpinBox
    joint_row = QHBoxLayout()
    joint_row.addWidget(QLabel("λ (landmark weight):"))
    lam_sb = QDoubleSpinBox(); lam_sb.setRange(0.0, 1000.0)
    lam_sb.setDecimals(1); lam_sb.setSingleStep(1.0); lam_sb.setValue(10.0)
    joint_row.addWidget(lam_sb)
    jw = QWidget(); jw.setLayout(joint_row); ctrl_v.addWidget(jw)

    run_joint_btn = QPushButton("Run JOINT MI + landmarks (single opt)")
    def _run_joint():
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 4:
            viewer.status = f"joint needs ≥4 landmark pairs (have {n})"
            return
        _snapshot()
        in_state = _build_in_memory_state()
        from .step_mi import run_joint
        viewer.status = "Running JOINT MI+landmarks... (see terminal)"
        print(f"[landmarks] launching JOINT in-memory: "
              f"shape={cb_shape.isChecked()}, affine={cb_affine.isChecked()}, "
              f"mask={cb_mask.isChecked()}, λ={lam_sb.value()}, n_lm={n} "
              f"(state.json untouched)", flush=True)
        try:
            new_state = run_joint(in_state,
                                  shape=cb_shape.isChecked(),
                                  affine=cb_affine.isChecked(),
                                  mask=cb_mask.isChecked(),
                                  landmark_weight=float(lam_sb.value()))
        except SystemExit as e:
            print(f"[landmarks] JOINT aborted: {e}", flush=True)
            viewer.status = f"JOINT aborted: {e}"
            return
        _apply_returned_state(new_state)
        sample_layer.data = _resample()
        _refresh_sample_lm_display()
        viewer.status = "JOINT done — preview updated. Click Revert (or Save to keep)."
        print("[landmarks] JOINT done. state.json unchanged until you Save.",
              flush=True)
    run_joint_btn.clicked.connect(_run_joint)
    ctrl_v.addWidget(run_joint_btn)

    # ITERATIVE joint button: alternates regularized landmark LSQ and
    # ANTs MI; iteration N's landmark fit is pulled toward iteration
    # (N-1)'s MI result by λ.
    from qtpy.QtWidgets import QSpinBox
    iter_row = QHBoxLayout()
    iter_row.addWidget(QLabel("max iter:"))
    iter_sb = QSpinBox(); iter_sb.setRange(1, 50); iter_sb.setValue(5)
    iter_row.addWidget(iter_sb)
    iw = QWidget(); iw.setLayout(iter_row); ctrl_v.addWidget(iw)

    run_joint_iter_btn = QPushButton(
        "Run JOINT iterative (block-coord descent)")
    def _run_joint_iter():
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n < 4:
            viewer.status = f"iterative needs ≥4 landmark pairs (have {n})"
            return
        _snapshot()
        in_state = _build_in_memory_state()
        from .step_mi import run_joint_iter
        viewer.status = "Running JOINT iterative... (see terminal)"
        print(f"[landmarks] launching JOINT iterative in-memory: "
              f"affine={cb_affine.isChecked()}, shape={cb_shape.isChecked()}, "
              f"mask={cb_mask.isChecked()}, λ={lam_sb.value()}, "
              f"max_iter={iter_sb.value()} (state.json untouched)", flush=True)
        try:
            new_state = run_joint_iter(in_state,
                                       affine=cb_affine.isChecked(),
                                       shape=cb_shape.isChecked(),
                                       mask=cb_mask.isChecked(),
                                       landmark_weight=float(lam_sb.value()),
                                       max_iter=int(iter_sb.value()))
        except SystemExit as e:
            print(f"[landmarks] JOINT iter aborted: {e}", flush=True)
            viewer.status = f"JOINT iter aborted: {e}"
            return
        _apply_returned_state(new_state)
        sample_layer.data = _resample()
        _refresh_sample_lm_display()
        viewer.status = "JOINT iter done — preview updated (Save to keep)."
        print("[landmarks] JOINT iter done. state.json unchanged until you Save.",
              flush=True)
    run_joint_iter_btn.clicked.connect(_run_joint_iter)
    ctrl_v.addWidget(run_joint_iter_btn)

    # Save without exit — keep the napari window open for more work
    save_only_btn = QPushButton("Save state.json (keep window open)")
    def _save_only():
        box["transform"].center_um = box["center_um"]
        state.transform = box["transform"].to_dict()
        state.affine = (box["affine_matrix"].tolist()
                        if box.get("affine_matrix") is not None else None)
        state.landmarks = {
            "sample_um": [list(p) for p in landmarks_sample],
            "ccf_um":    [list(p) for p in landmarks_ccf],
        }
        state.local_refinements = [L.to_dict() for L
                                    in box.get("local_refinements", [])]
        has_affine = box.get("affine_matrix") is not None
        n_lr = len(state.local_refinements)
        state.add_history(
            "landmarks",
            f"{len(landmarks_sample)} sample, {len(landmarks_ccf)} CCF landmarks"
            + (f", affine fitted" if has_affine else "")
            + (f", {n_lr} local refinement(s)" if n_lr else "")
            + " (intermediate save)",
        )
        save_state(state, state_path)
        viewer.status = f"saved -> {state_path}"
        print(f"[landmarks] saved -> {state_path} (window still open)", flush=True)
    save_only_btn.clicked.connect(_save_only)
    ctrl_v.addWidget(save_only_btn)

    clear_btn = QPushButton("Clear ALL landmarks")
    def _clear_all():
        landmarks_sample.clear(); landmarks_ccf.clear()
        _refresh_sample_lm_display(); _refresh_ccf_lm_display(); _refresh_lists()
        if box.get("tps_enabled"):
            sample_layer.data = _resample()
    clear_btn.clicked.connect(_clear_all)
    ctrl_v.addWidget(clear_btn)

    # --------- import / export landmarks ------------------------------
    # Round-trip pairs through external files (CSV BigWarp-format or
    # vol2atlas JSON). Coordinates are PHYSICAL µm — atlas-resolution
    # invariant. Useful for sharing, version control, or pre-picking
    # landmarks in another tool.
    from qtpy.QtWidgets import QFileDialog
    io_row = QHBoxLayout()
    import_btn = QPushButton("Import landmarks…")
    export_btn = QPushButton("Export landmarks…")
    io_row.addWidget(import_btn); io_row.addWidget(export_btn)
    iow = QWidget(); iow.setLayout(io_row); ctrl_v.addWidget(iow)

    def _do_import():
        from ..transform_io import read_landmarks_csv
        import json as _json
        path, _ = QFileDialog.getOpenFileName(
            None, "Import landmarks", "",
            "Landmarks (*.csv *.json);;CSV (*.csv);;JSON (*.json)")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() == ".csv":
            smp, ccf = read_landmarks_csv(p)
        elif p.suffix.lower() == ".json":
            d = _json.loads(p.read_text())
            smp = [tuple(pt) for pt in d.get("sample_um", [])]
            ccf = [tuple(pt) for pt in d.get("ccf_um", [])]
        else:
            viewer.status = f"unknown format {p.suffix}; use .csv or .json"
            return
        n = min(len(smp), len(ccf))
        if n == 0:
            viewer.status = f"no usable pairs in {p.name}"
            return
        # Replace; append is rare and ambiguous in GUI.
        landmarks_sample.clear(); landmarks_ccf.clear()
        for s_pt in smp[:n]:
            landmarks_sample.append(tuple(float(v) for v in s_pt))
        for c_pt in ccf[:n]:
            landmarks_ccf.append(tuple(float(v) for v in c_pt))
        _refresh_sample_lm_display(); _refresh_ccf_lm_display()
        _refresh_lists()
        if box.get("tps_enabled"):
            sample_layer.data = _resample()
        viewer.status = f"imported {n} pairs from {p.name}"
        print(f"[landmarks] imported {n} pairs from {p}", flush=True)
    import_btn.clicked.connect(_do_import)

    def _do_export():
        from ..transform_io import write_landmarks_csv
        import json as _json
        n = min(len(landmarks_sample), len(landmarks_ccf))
        if n == 0:
            viewer.status = "no landmarks to export"
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "Export landmarks",
            str(state_path.parent / "landmarks.csv"),
            "CSV BigWarp (*.csv);;JSON (*.json)")
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() == ".json":
            p.write_text(_json.dumps({
                "sample_um": [list(pt) for pt in landmarks_sample[:n]],
                "ccf_um":    [list(pt) for pt in landmarks_ccf[:n]],
            }, indent=2))
        else:
            # Default to CSV for anything else (incl. extensionless)
            if p.suffix.lower() != ".csv":
                p = p.with_suffix(".csv")
            write_landmarks_csv(
                landmarks_sample[:n], landmarks_ccf[:n], p)
        viewer.status = f"exported {n} pairs → {p.name}"
        print(f"[landmarks] exported {n} pairs → {p}", flush=True)
    export_btn.clicked.connect(_do_export)

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
        if box.get("affine_matrix") is not None:
            state.affine = box["affine_matrix"].tolist()
        else:
            state.affine = None
        state.landmarks = {
            "sample_um": [list(p) for p in landmarks_sample],
            "ccf_um":    [list(p) for p in landmarks_ccf],
        }
        state.local_refinements = [L.to_dict() for L
                                    in box.get("local_refinements", [])]
        has_affine = box.get("affine_matrix") is not None
        n_lr = len(state.local_refinements)
        state.add_history(
            "landmarks",
            f"{len(landmarks_sample)} sample, {len(landmarks_ccf)} CCF landmarks"
            + (f", affine fitted" if has_affine else "")
            + (f", {n_lr} local refinement(s)" if n_lr else ""),
        )
        save_state(state, state_path)
        print(f"[step3] saved -> {state_path}", flush=True)
        viewer.close()
    save_btn.clicked.connect(_save_and_exit)
    ctrl_v.addWidget(save_btn)

    dock = viewer.window.add_dock_widget(ctrl, name="landmarks", area="right")
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
