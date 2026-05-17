"""Interactive aligner: napari overlay of CCF + downsampled sample with sliders."""
from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np

from .atlas import CCFReference, load_ccf
from .io import MultiscaleVolume, open_multiscale
from .transform import RigidTransform, save_transform


class FloatRangeSlider:
    """A QSlider (int ticks) + QDoubleSpinBox combo, GUARANTEED to start at 0
    in the middle of the track. Replaces magicgui FloatSlider, which is buggy
    at wide ranges (handle stays stuck at min)."""

    def __init__(self, half_range: float, step: float, label: str):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (QWidget, QHBoxLayout, QSlider,
                                    QDoubleSpinBox, QLabel)

        self._step = float(step)
        self._ticks = max(2, int(round(half_range / step)))
        self._callbacks: list = []
        self._suppress = False

        self.widget = QWidget()
        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(-self._ticks)
        self._slider.setMaximum(self._ticks)
        self._slider.setValue(0)
        self._slider.setPageStep(max(1, self._ticks // 100))
        self._slider.setSingleStep(max(1, self._ticks // 1000))
        layout.addWidget(self._slider, 4)

        self._spin = QDoubleSpinBox()
        self._spin.setMinimum(-half_range)
        self._spin.setMaximum(half_range)
        self._spin.setValue(0.0)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(2)
        self._spin.setMinimumWidth(90)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, t: int):
        if self._suppress:
            return
        v = t * self._step
        self._suppress = True
        self._spin.setValue(v)
        self._suppress = False
        self._fire(v)

    def _on_spin(self, v: float):
        if self._suppress:
            return
        t = int(round(v / self._step))
        self._suppress = True
        self._slider.setValue(t)
        self._suppress = False
        self._fire(v)

    def _fire(self, v: float):
        for cb in self._callbacks:
            cb(v)

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, v: float):
        self._suppress = True
        self._spin.setValue(v)
        self._slider.setValue(int(round(v / self._step)))
        self._suppress = False

    def connect(self, callback):
        self._callbacks.append(callback)


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def align_interactive(
    sample: str | Path | MultiscaleVolume,
    level: int = 5,
    channel: int | None = 0,
    atlas_name: str = "allen_mouse_25um",
    ccf: CCFReference | None = None,
    initial: RigidTransform | None = None,
    save_to: str | Path | None = None,
    preview_max_voxels: int = 256 * 256 * 256,   # ~16M voxels => ~32 MB uint16
    ndisplay: int = 2,
    voxel_um_override: tuple[float, float, float] | None = None,
) -> RigidTransform:
    """Open a napari window to align `sample` (at pyramid `level`) onto the CCF.

    Returns the final RigidTransform once the window is closed.
    Optionally writes it to `save_to` (JSON).
    """
    import napari
    from magicgui import magicgui

    print("[zrot] align_interactive: starting", flush=True)
    # ---- load data (lazy) -------------------------------------------------
    ms = sample if isinstance(sample, MultiscaleVolume) else open_multiscale(sample)
    if not 0 <= level < ms.n_levels():
        raise ValueError(
            f"level {level} not available — pyramid has {ms.n_levels()} levels "
            f"(valid: 0..{ms.n_levels() - 1}).\n\n{ms.summary()}\n\n"
            f"Hint: pick the largest level whose voxel size is close to your "
            f"chosen atlas resolution (e.g. ~25 µm for 'allen_mouse_25um')."
        )
    arr = ms.level(level)
    if "c" in ms.axes and channel is not None:
        c_axis = ms.axes.index("c")
        if not 0 <= channel < arr.shape[c_axis]:
            raise ValueError(
                f"channel {channel} out of range; level {level} has "
                f"{arr.shape[c_axis]} channels"
            )
        arr = da.take(arr, channel, axis=c_axis)
    sample_um = ms.spacing(level)
    if voxel_um_override is not None:
        print(f"[zrot] voxel size override: {sample_um} → {voxel_um_override} µm",
              flush=True)
        sample_um = tuple(float(v) for v in voxel_um_override)

    # ---- downsample into a RAM preview (avoids dask MIP hangs) -----------
    spatial = arr.shape[-3:]
    n_vox = int(np.prod(spatial))
    if n_vox > preview_max_voxels:
        factor = max(1, int(np.ceil((n_vox / preview_max_voxels) ** (1 / 3))))
        out_vox = int(np.prod([s // factor for s in spatial]))
        print(
            f"[zrot] level {level} is {spatial} "
            f"({_human(n_vox * arr.dtype.itemsize)}); "
            f"strided x{factor} for preview "
            f"(output ~{_human(out_vox * arr.dtype.itemsize)})...",
            flush=True,
        )
        arr = arr[..., ::factor, ::factor, ::factor]
        sample_um = tuple(s * factor for s in sample_um)
    else:
        print(f"[zrot] loading level {level} {spatial} into RAM...", flush=True)

    import time as _time
    _t0 = _time.time()
    sample_np = np.ascontiguousarray(arr.compute())
    _dt = _time.time() - _t0
    print(f"[zrot] preview ready in {_dt:.1f}s: shape={sample_np.shape} "
          f"dtype={sample_np.dtype} (~{_human(sample_np.nbytes)})", flush=True)

    # Robust contrast limits from a percentile (cheap, fixes all-black render)
    flat = sample_np.ravel()
    if flat.size > 1_000_000:
        flat = flat[:: flat.size // 1_000_000]
    lo, hi = np.percentile(flat, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    sample_clim = (float(lo), float(hi))

    if ccf is None:
        ccf = load_ccf(atlas_name)

    # ---- initial transform centered on sample volume ----------------------
    spatial_shape = sample_np.shape
    sample_center_um = tuple(
        (s - 1) * v / 2.0 for s, v in zip(spatial_shape, sample_um)
    )
    ccf_center_um = tuple(
        (s - 1) * v / 2.0 for s, v in zip(ccf.reference.shape, ccf.voxel_um)
    )
    if initial is None:
        # seed translation so sample center lands on CCF center
        initial = RigidTransform(
            tz_um=ccf_center_um[0] - sample_center_um[0],
            ty_um=ccf_center_um[1] - sample_center_um[1],
            tx_um=ccf_center_um[2] - sample_center_um[2],
            center_um=sample_center_um,
        )
    else:
        initial.center_um = sample_center_um

    state = {"transform": initial, "muted": False}

    # ---- viewer setup -----------------------------------------------------
    # Default to 2D (slice view) — fast, immediate. Toggle to 3D from the
    # napari toolbar when you want a volume view.
    viewer = napari.Viewer(ndisplay=ndisplay)

    # CCF contrast — cheap, atlas is small
    ccf_flat = ccf.reference.ravel()
    if ccf_flat.size > 1_000_000:
        ccf_flat = ccf_flat[:: ccf_flat.size // 1_000_000]
    ccf_lo, ccf_hi = np.percentile(ccf_flat, [1, 99.5])

    viewer.add_image(
        ccf.reference,
        name=f"CCF ({ccf.name})",
        scale=ccf.voxel_um,
        colormap="gray",
        rendering="mip",
        contrast_limits=(float(ccf_lo), float(max(ccf_hi, ccf_lo + 1))),
        opacity=0.6,
        blending="additive",
    )

    # ---- landmark layers (always visible, in world µm) -----------------
    # SAMPLE landmarks are stored in sample-µm internally; their displayed
    # position is updated to current transform on every slider change.
    # CCF landmarks live in CCF-µm and never move.
    # Points-layer `size` is in DATA units, which is per-layer:
    #   - sample_lm_layer has scale=(1,1,1) → data unit = world µm. Size in µm.
    #   - ccf_lm_layer    has scale=ccf.voxel_um → data unit = voxels.
    # Both should look the same physical size on screen (~150 µm).
    _PHYS_LM_UM = 150.0
    sample_lm_layer = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=(1.0, 1.0, 1.0),
        name="sample landmarks (cyan)", face_color="cyan",
        border_color="white", size=_PHYS_LM_UM,
    )
    ccf_lm_layer = viewer.add_points(
        np.empty((0, 3)), ndim=3, scale=ccf.voxel_um,
        name="ccf landmarks (yellow)", face_color="yellow",
        border_color="white", size=_PHYS_LM_UM / ccf.voxel_um[0],
    )
    landmarks_sample_um: list[tuple[float, float, float]] = []
    landmarks_ccf_um: list[tuple[float, float, float]] = []
    # ---- WYSIWYG truth layer ---------------------------------------------
    # The ONLY sample layer is the scipy resample onto the CCF voxel grid.
    # No fake-affine layer that could confuse you by appearing in the layer
    # list. Slider change → debounced (200 ms) recompute → display.
    from scipy.ndimage import affine_transform as _scipy_affine
    from qtpy.QtCore import QTimer

    state["prewarp_layer"] = None
    state["prewarp_timer"] = QTimer()
    state["prewarp_timer"].setSingleShot(True)

    def _compute_prewarp():
        M = state["transform"].for_voxel_grid(sample_um, ccf.voxel_um)
        Minv = np.linalg.inv(M)
        return _scipy_affine(
            sample_np, Minv[:3, :3], offset=Minv[:3, 3],
            output_shape=ccf.reference.shape,
            order=0, mode="constant", cval=0,
        )

    def _refresh_prewarp_now():
        data = _compute_prewarp()
        if state["prewarp_layer"] is None:
            state["prewarp_layer"] = viewer.add_image(
                data, name="sample (live = saved = applied)",
                scale=ccf.voxel_um, colormap="magenta",
                contrast_limits=sample_clim, opacity=0.6,
                blending="additive",
                interpolation2d="nearest", interpolation3d="nearest",
            )
        else:
            state["prewarp_layer"].data = data

    state["prewarp_timer"].timeout.connect(_refresh_prewarp_now)

    def _apply():
        # Schedule the truthful resample, and update sample-landmark positions
        # (they're stored in sample µm; their displayed world coord moves with
        # the transform).
        state["prewarp_timer"].start(200)   # debounce, restarts on each call
        try: _refresh_sample_display()
        except NameError: pass   # before lm panel exists, on first _apply()

    # Eagerly create the layer so shift+drag etc. can attach callbacks.
    _refresh_prewarp_now()
    sample_layer = state["prewarp_layer"]   # alias for backward-compat below

    # ---- DELTA sliders: 0 = current pose; ± nudges from there ------------
    # Slider value is a *delta* from `baseline`. The applied transform is
    # baseline + delta. Click "Commit" to fold the delta into baseline and
    # snap the sliders back to 0.
    baseline = {
        "rz_deg": initial.rz_deg, "ry_deg": initial.ry_deg, "rx_deg": initial.rx_deg,
        "tz_um": initial.tz_um, "ty_um": initial.ty_um, "tx_um": initial.tx_um,
    }
    flips = {"flip_z": initial.flip_z, "flip_y": initial.flip_y, "flip_x": initial.flip_x}
    bbox_um = max(
        spatial_shape[i] * sample_um[i] for i in range(3)
    ) + max(ccf.reference.shape[i] * ccf.voxel_um[i] for i in range(3))
    DELTA_T_RANGE = float(bbox_um)   # µm — full reach
    DELTA_R_RANGE = 180.0            # deg — full circle (±180°)

    # Custom Qt widget (FloatRangeSlider) — guaranteed to start at 0/middle.
    sliders = {
        "rz_deg": FloatRangeSlider(DELTA_R_RANGE, 0.5, "Δrz (deg)"),
        "ry_deg": FloatRangeSlider(DELTA_R_RANGE, 0.5, "Δry (deg)"),
        "rx_deg": FloatRangeSlider(DELTA_R_RANGE, 0.5, "Δrx (deg)"),
        "tz_um":  FloatRangeSlider(DELTA_T_RANGE, 25.0, "Δtz (µm)"),
        "ty_um":  FloatRangeSlider(DELTA_T_RANGE, 25.0, "Δty (µm)"),
        "tx_um":  FloatRangeSlider(DELTA_T_RANGE, 25.0, "Δtx (µm)"),
    }

    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                                QLabel)
    controls_widget = QWidget()
    _v = QVBoxLayout(controls_widget)
    _v.setContentsMargins(4, 4, 4, 4)
    for _n in ("rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"):
        _v.addWidget(sliders[_n].widget)

    # ---- flip checkboxes -------------------------------------------------
    _flip_row = QWidget()
    _flip_layout = QHBoxLayout(_flip_row)
    _flip_layout.setContentsMargins(2, 6, 2, 2)
    _flip_layout.addWidget(QLabel("flip:"))
    flip_chks = {}
    for axis in ("z", "y", "x"):
        cb = QCheckBox(axis)
        cb.setChecked(flips[f"flip_{axis}"])
        flip_chks[f"flip_{axis}"] = cb
        _flip_layout.addWidget(cb)
    _flip_layout.addStretch(1)
    _v.addWidget(_flip_row)

    def _apply_from_sliders(*_):
        if state["muted"]:
            return
        for axis in ("z", "y", "x"):
            flips[f"flip_{axis}"] = flip_chks[f"flip_{axis}"].isChecked()
        state["transform"] = RigidTransform(
            rz_deg=baseline["rz_deg"] + sliders["rz_deg"].value(),
            ry_deg=baseline["ry_deg"] + sliders["ry_deg"].value(),
            rx_deg=baseline["rx_deg"] + sliders["rx_deg"].value(),
            tz_um=baseline["tz_um"] + sliders["tz_um"].value(),
            ty_um=baseline["ty_um"] + sliders["ty_um"].value(),
            tx_um=baseline["tx_um"] + sliders["tx_um"].value(),
            flip_z=flips["flip_z"], flip_y=flips["flip_y"], flip_x=flips["flip_x"],
            center_um=sample_center_um,
        )
        _refresh_abs_label()
        _apply()

    for _cb in flip_chks.values():
        _cb.stateChanged.connect(_apply_from_sliders)

    for _s in sliders.values():
        _s.connect(_apply_from_sliders)

    # Tiny shim so the rest of the file (jog, keyboard, drag, commit, reset)
    # can keep using `controls.<name>.value` semantics.
    class _ControlsShim:
        def __getattr__(self, name):
            if name in sliders:
                class _SlotProxy:
                    def __init__(self, slot): self._slot = slot
                    @property
                    def value(self): return self._slot.value()
                    @value.setter
                    def value(self, v): self._slot.setValue(v); _apply_from_sliders()
                return _SlotProxy(sliders[name])
            raise AttributeError(name)
    controls = _ControlsShim()

    # Sliders constructed at value=0 already — kick off the first redraw.
    _apply()

    def _set(rz=None, ry=None, rx=None, tz=None, ty=None, tx=None):
        if rz is not None: controls.rz_deg.value = rz
        if ry is not None: controls.ry_deg.value = ry
        if rx is not None: controls.rx_deg.value = rx
        if tz is not None: controls.tz_um.value = tz
        if ty is not None: controls.ty_um.value = ty
        if tx is not None: controls.tx_um.value = tx

    # ---- absolute-value display + commit/reset for delta model ------------
    from magicgui.widgets import Container, PushButton, ComboBox, Label

    abs_label = Label(value="")
    def _refresh_abs_label():
        t = state["transform"]
        flip_str = "".join(ax for ax, on in
                           zip("zyx", (t.flip_z, t.flip_y, t.flip_x)) if on) or "—"
        abs_label.value = (
            f"abs: tz={t.tz_um:+8.1f}  ty={t.ty_um:+8.1f}  tx={t.tx_um:+8.1f} µm\n"
            f"     rz={t.rz_deg:+7.2f}  ry={t.ry_deg:+7.2f}  rx={t.rx_deg:+7.2f} deg\n"
            f"     flips: {flip_str}"
        )

    step_translate = ComboBox(
        label="t step (µm)", value=100.0,
        choices=[("1 µm", 1.0), ("10 µm", 10.0), ("100 µm", 100.0),
                 ("500 µm", 500.0), ("2000 µm", 2000.0)],
    )
    step_rotate = ComboBox(
        label="r step (deg)", value=1.0,
        choices=[("0.1°", 0.1), ("1°", 1.0), ("5°", 5.0), ("10°", 10.0)],
    )

    def _bump(name, sign):
        t_step = float(step_translate.value)
        r_step = float(step_rotate.value)
        step = r_step if name.startswith("r") else t_step
        widget = getattr(controls, name)
        widget.value = widget.value + sign * step

    def _row(label, neg_name, pos_name, target):
        neg = PushButton(text=f"− {label}")
        pos = PushButton(text=f"+ {label}")
        neg.clicked.connect(lambda: _bump(target, -1))
        pos.clicked.connect(lambda: _bump(target, +1))
        return Container(layout="horizontal", widgets=[neg, Label(value=label), pos])

    jog = Container(
        layout="vertical",
        widgets=[
            step_translate,
            _row("tx", "tx-", "tx+", "tx_um"),
            _row("ty", "ty-", "ty+", "ty_um"),
            _row("tz", "tz-", "tz+", "tz_um"),
            step_rotate,
            _row("rz", "rz-", "rz+", "rz_deg"),
            _row("ry", "ry-", "ry+", "ry_deg"),
            _row("rx", "rx-", "rx+", "rx_deg"),
        ],
    )

    # ---- keyboard shortcuts on the viewer ---------------------------------
    # Arrows: nudge in the displayed plane. Shift = 10x, Ctrl = 0.1x.
    def _kb_step(event, base):
        if "Shift" in event.modifiers:
            return base * 10
        if "Control" in event.modifiers:
            return base * 0.1
        return base

    @viewer.bind_key("Left", overwrite=True)
    def _left(viewer):
        s = _kb_step(viewer.mouse_drag_callbacks, float(step_translate.value))
        controls.tx_um.value -= s
    @viewer.bind_key("Right", overwrite=True)
    def _right(viewer):
        controls.tx_um.value += float(step_translate.value)
    @viewer.bind_key("Up", overwrite=True)
    def _up(viewer):
        controls.ty_um.value -= float(step_translate.value)
    @viewer.bind_key("Down", overwrite=True)
    def _down(viewer):
        controls.ty_um.value += float(step_translate.value)
    @viewer.bind_key("PageUp", overwrite=True)
    def _pgup(viewer):
        controls.tz_um.value -= float(step_translate.value)
    @viewer.bind_key("PageDown", overwrite=True)
    def _pgdn(viewer):
        controls.tz_um.value += float(step_translate.value)
    @viewer.bind_key("q", overwrite=True)
    def _qk(viewer):
        controls.rz_deg.value -= float(step_rotate.value)
    @viewer.bind_key("e", overwrite=True)
    def _ek(viewer):
        controls.rz_deg.value += float(step_rotate.value)

    # ---- shift+drag on the sample layer to translate -----------------------
    @sample_layer.mouse_drag_callbacks.append
    def _drag(layer, event):
        if "Shift" not in event.modifiers:
            return
        start_pos = np.array(event.position, dtype=float)
        start_t = (state["transform"].tz_um,
                   state["transform"].ty_um,
                   state["transform"].tx_um)
        yield
        while event.type == "mouse_move":
            cur = np.array(event.position, dtype=float)
            d = cur - start_pos
            # event.position is in world coords (z, y, x in µm); 2D mode gives (y, x).
            if d.size == 3:
                _set(tz=start_t[0] + d[0], ty=start_t[1] + d[1], tx=start_t[2] + d[2])
            elif d.size == 2:
                _set(ty=start_t[1] + d[0], tx=start_t[2] + d[1])
            yield

    @magicgui(call_button="Save transform")
    def save_btn():
        if save_to is not None:
            save_transform(state["transform"], save_to)
            msg = f"[zrot] saved transform -> {Path(save_to).resolve()}"
            print(msg)
            try:
                viewer.status = msg
            except Exception:
                pass
        else:
            print("[zrot] no save path provided; use save_to=... to persist")

    @magicgui(call_button="Reset Δ to 0")
    def reset_btn():
        # Zero the deltas — baseline stays. Use this to "park" the sliders
        # in the middle without changing the actual pose.
        for n in ("rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"):
            getattr(controls, n).value = 0.0

    @magicgui(call_button="Commit Δ -> baseline")
    def commit_btn():
        # Fold the current delta into baseline and zero the sliders.
        # Useful when you've used most of the slider range and want to keep going.
        for n in ("rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"):
            baseline[n] += float(getattr(controls, n).value)
            getattr(controls, n).value = 0.0
        msg = (f"[zrot] committed. baseline now: "
               f"tz={baseline['tz_um']:.1f} ty={baseline['ty_um']:.1f} "
               f"tx={baseline['tx_um']:.1f} µm | "
               f"rz={baseline['rz_deg']:.2f} ry={baseline['ry_deg']:.2f} "
               f"rx={baseline['rx_deg']:.2f} deg")
        print(msg)
        try:
            viewer.status = msg
        except Exception:
            pass

    # ---- landmark panel ------------------------------------------------
    # Two lists (sample / CCF), plus "Add sample" / "Add CCF" buttons that
    # arm the next canvas click. Pairing is by row index.
    from qtpy.QtWidgets import (QListWidget, QListWidgetItem, QSplitter,
                                 QGroupBox, QPushButton)

    state["pending_lm_type"] = None   # None | 'sample' | 'ccf'

    def _refresh_sample_display():
        if not landmarks_sample_um:
            sample_lm_layer.data = np.empty((0, 3))
            return
        M = state["transform"].matrix()
        pts = np.asarray(landmarks_sample_um)
        world = (M[:3, :3] @ pts.T).T + M[:3, 3]
        sample_lm_layer.data = world

    def _refresh_ccf_display():
        if not landmarks_ccf_um:
            ccf_lm_layer.data = np.empty((0, 3))
            return
        # ccf_lm_layer has scale=ccf.voxel_um, so data = world / scale = voxel idx
        pts = np.asarray(landmarks_ccf_um)
        ccf_lm_layer.data = pts / np.asarray(ccf.voxel_um)

    def _refresh_lists():
        sample_list.clear()
        for i, (z, y, x) in enumerate(landmarks_sample_um):
            sample_list.addItem(f"#{i}  z={z:+8.1f}  y={y:+8.1f}  x={x:+8.1f} µm")
        ccf_list.clear()
        for i, (z, y, x) in enumerate(landmarks_ccf_um):
            ccf_list.addItem(f"#{i}  z={z:+8.1f}  y={y:+8.1f}  x={x:+8.1f} µm")

    def _add_lm_after_click(kind: str):
        state["pending_lm_type"] = kind
        viewer.status = (f"click on a {kind.upper()} feature in the canvas "
                         f"(one click adds it, then stops)")
        if kind == "sample":
            add_sample_btn.setStyleSheet("background-color: #225588;")
        else:
            add_ccf_btn.setStyleSheet("background-color: #886600;")

    @viewer.mouse_drag_callbacks.append
    def _capture_click(_viewer, event):
        if state["pending_lm_type"] is None:
            return
        # Robust 3D world position from the click: take the current dims.point
        # (which has the slice positions for non-displayed axes) and override
        # the DISPLAYED axes with the cursor coordinates from event.position.
        slice_pos = list(viewer.dims.point)
        displayed = viewer.dims.displayed
        cursor = np.asarray(event.position, dtype=float)
        if cursor.size == len(displayed):
            for i, ax in enumerate(displayed):
                slice_pos[ax] = float(cursor[i])
        elif cursor.size == len(slice_pos):
            slice_pos = [float(c) for c in cursor]
        world = np.asarray(slice_pos)

        kind = state["pending_lm_type"]
        state["pending_lm_type"] = None
        add_sample_btn.setStyleSheet("")
        add_ccf_btn.setStyleSheet("")
        if kind == "sample":
            M = state["transform"].matrix()
            M_inv = np.linalg.inv(M)
            sample_pos = (M_inv[:3, :3] @ world) + M_inv[:3, 3]
            landmarks_sample_um.append(tuple(map(float, sample_pos)))
            _refresh_sample_display()
            i = len(landmarks_sample_um) - 1
            msg = (f"added SAMPLE landmark #{i}: world={tuple(round(w,1) for w in world)} "
                   f"→ sample_µm={tuple(round(p,1) for p in sample_pos)}")
            print(msg, flush=True); viewer.status = msg
        else:
            landmarks_ccf_um.append(tuple(map(float, world)))
            _refresh_ccf_display()
            i = len(landmarks_ccf_um) - 1
            msg = f"added CCF landmark #{i}: ccf_µm={tuple(round(w,1) for w in world)}"
            print(msg, flush=True); viewer.status = msg
        _refresh_lists()

    # Build the panel
    lm_panel = QWidget()
    lm_v = QVBoxLayout(lm_panel)
    lm_v.setContentsMargins(4, 4, 4, 4)

    btn_row = QHBoxLayout()
    add_sample_btn = QPushButton("+ SAMPLE landmark\n(next click)")
    add_ccf_btn = QPushButton("+ CCF landmark\n(next click)")
    add_sample_btn.clicked.connect(lambda: _add_lm_after_click("sample"))
    add_ccf_btn.clicked.connect(lambda: _add_lm_after_click("ccf"))
    btn_row.addWidget(add_sample_btn)
    btn_row.addWidget(add_ccf_btn)
    row_wrap = QWidget(); row_wrap.setLayout(btn_row); lm_v.addWidget(row_wrap)

    lists_row = QHBoxLayout()
    s_box = QGroupBox("sample (µm)")
    s_box_v = QVBoxLayout(s_box); s_box_v.setContentsMargins(2, 2, 2, 2)
    sample_list = QListWidget(); sample_list.setMaximumHeight(150)
    s_box_v.addWidget(sample_list)
    s_del = QPushButton("Delete selected")
    s_del.clicked.connect(lambda: (_delete_selected("sample")))
    s_box_v.addWidget(s_del)
    lists_row.addWidget(s_box)

    c_box = QGroupBox("CCF (µm)")
    c_box_v = QVBoxLayout(c_box); c_box_v.setContentsMargins(2, 2, 2, 2)
    ccf_list = QListWidget(); ccf_list.setMaximumHeight(150)
    c_box_v.addWidget(ccf_list)
    c_del = QPushButton("Delete selected")
    c_del.clicked.connect(lambda: (_delete_selected("ccf")))
    c_box_v.addWidget(c_del)
    lists_row.addWidget(c_box)
    lr_wrap = QWidget(); lr_wrap.setLayout(lists_row); lm_v.addWidget(lr_wrap)

    fit_btn = QPushButton("Fit rigid from landmarks (≥3 pairs)")
    clear_btn = QPushButton("Clear ALL landmarks")
    hint = QLabel(
        "<i>tip: select a row in the list above, click 'Delete selected'. "
        "Or in the layer-controls toolbar (top-left), pick the points layer "
        "and press the '−' icon to delete by clicking points on the canvas.</i>"
    )
    hint.setWordWrap(True); hint.setStyleSheet("color: #888; padding: 4px;")
    lm_v.addWidget(fit_btn); lm_v.addWidget(clear_btn); lm_v.addWidget(hint)

    def _delete_selected(which: str):
        if which == "sample":
            rows = sorted({i.row() for i in sample_list.selectedIndexes()}, reverse=True)
            for r in rows:
                if 0 <= r < len(landmarks_sample_um):
                    landmarks_sample_um.pop(r)
            _refresh_sample_display()
        else:
            rows = sorted({i.row() for i in ccf_list.selectedIndexes()}, reverse=True)
            for r in rows:
                if 0 <= r < len(landmarks_ccf_um):
                    landmarks_ccf_um.pop(r)
            _refresh_ccf_display()
        _refresh_lists()

    def _clear_all():
        landmarks_sample_um.clear()
        landmarks_ccf_um.clear()
        _refresh_sample_display(); _refresh_ccf_display(); _refresh_lists()
        viewer.status = "all landmarks cleared"
    clear_btn.clicked.connect(_clear_all)

    def _fit_rigid_from_landmarks():
        n = min(len(landmarks_sample_um), len(landmarks_ccf_um))
        if n < 3:
            msg = (f"need ≥3 pairs (sample={len(landmarks_sample_um)}, "
                   f"ccf={len(landmarks_ccf_um)})")
            print(msg, flush=True); viewer.status = msg; return
        if len(landmarks_sample_um) != len(landmarks_ccf_um):
            msg = (f"count mismatch: sample={len(landmarks_sample_um)}, "
                   f"ccf={len(landmarks_ccf_um)} (fitting first {n})")
            print(msg, flush=True); viewer.status = msg

        sp = np.asarray(landmarks_sample_um[:n])
        cp = np.asarray(landmarks_ccf_um[:n])

        # Apply current flips to the source side: if flips are on, our actual
        # transform is R @ F. We want to fit R given F. Pre-apply F to sources.
        F_diag = np.array([
            -1.0 if flips["flip_z"] else 1.0,
            -1.0 if flips["flip_y"] else 1.0,
            -1.0 if flips["flip_x"] else 1.0,
        ])
        c = np.asarray(sample_center_um)
        sp_F = (sp - c) * F_diag + c

        # Sanity check: landmarks should not be coplanar
        spread = (sp_F - sp_F.mean(0)).std(0)
        if spread.min() < 1e-3 * spread.max():
            msg = (f"WARNING: landmarks are nearly coplanar (spread {spread}). "
                   f"Add points in the under-constrained axis or the fit will "
                   f"be ill-conditioned.")
            print(msg, flush=True); viewer.status = msg

        # Procrustes / Kabsch: R @ sp_F + t ≈ cp
        src_c = sp_F.mean(axis=0); tgt_c = cp.mean(axis=0)
        H = (sp_F - src_c).T @ (cp - tgt_c)
        U, S, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = tgt_c - R @ src_c
        rms = float(np.sqrt(np.mean(np.sum(((R @ sp_F.T).T + t - cp) ** 2, axis=1))))
        per_pt = np.sqrt(np.sum(((R @ sp_F.T).T + t - cp) ** 2, axis=1))

        from scipy.spatial.transform import Rotation
        rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)

        state["muted"] = True
        baseline["rz_deg"] = float(rz)
        baseline["ry_deg"] = float(ry)
        baseline["rx_deg"] = float(rx)
        offset = R @ c - c
        baseline["tz_um"] = float(t[0] + offset[0])
        baseline["ty_um"] = float(t[1] + offset[1])
        baseline["tx_um"] = float(t[2] + offset[2])
        for nm in ("rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"):
            getattr(controls, nm).value = 0.0
        state["muted"] = False

        # Build the new transform NOW (not via the slider path) so the prewarp
        # redraws immediately, not in 200 ms.
        state["transform"] = RigidTransform(
            rz_deg=baseline["rz_deg"], ry_deg=baseline["ry_deg"], rx_deg=baseline["rx_deg"],
            tz_um=baseline["tz_um"],   ty_um=baseline["ty_um"],   tx_um=baseline["tx_um"],
            flip_z=flips["flip_z"], flip_y=flips["flip_y"], flip_x=flips["flip_x"],
            center_um=sample_center_um,
        )
        _refresh_abs_label()
        _refresh_prewarp_now()         # immediate, not debounced
        _refresh_sample_display()

        print(f"\n[zrot.landmark_fit]")
        print(f"  pairs: {n}    RMS = {rms:.1f} µm")
        print(f"  flips active: "
              f"{[a for a, f in zip('zyx', F_diag) if f < 0] or 'none'}")
        print(f"  rotation:    rz={rz:+7.2f}  ry={ry:+7.2f}  rx={rx:+7.2f} deg")
        print(f"  translation: tz={baseline['tz_um']:+9.1f}  "
              f"ty={baseline['ty_um']:+9.1f}  tx={baseline['tx_um']:+9.1f} µm")
        print(f"  per-pair residuals (µm):")
        for i, r in enumerate(per_pt):
            print(f"    pair {i:2d}: {r:7.1f}{'   ← outlier?' if r > 3*rms else ''}")
        viewer.status = f"fit: {n} pairs, RMS {rms:.0f} µm. See terminal for details."
    fit_btn.clicked.connect(_fit_rigid_from_landmarks)

    # ---- view-axis switcher (verify in all three orthogonal planes) -----
    # napari's `dims.order` controls which axis is the slice axis. For data
    # in (z, y, x), order=(0,1,2) means "scroll z, show y-x plane" = coronal-ish.
    # Switching axes lets you check whether the alignment holds in every plane.
    from qtpy.QtWidgets import QPushButton as _QPB
    view_panel = QWidget()
    view_layout = QHBoxLayout(view_panel)
    view_layout.setContentsMargins(2, 2, 2, 2)
    def _set_view(order):
        viewer.dims.order = order
    cor = _QPB("coronal\n(scroll z)")
    axi = _QPB("axial\n(scroll y)")
    sag = _QPB("sagittal\n(scroll x)")
    nd3 = _QPB("3D MIP")
    cor.clicked.connect(lambda: (_set_view((0, 1, 2)), setattr(viewer.dims, "ndisplay", 2)))
    axi.clicked.connect(lambda: (_set_view((1, 0, 2)), setattr(viewer.dims, "ndisplay", 2)))
    sag.clicked.connect(lambda: (_set_view((2, 0, 1)), setattr(viewer.dims, "ndisplay", 2)))
    nd3.clicked.connect(lambda: setattr(viewer.dims, "ndisplay", 3))
    for b in (cor, axi, sag, nd3):
        view_layout.addWidget(b)

    # All controls live in a single TabWidget on the right so the napari
    # canvas + bottom z-slider stay visible at any screen size.
    from qtpy.QtWidgets import QTabWidget
    tabs = QTabWidget()

    actions_widget = QWidget()
    actions_layout = QVBoxLayout(actions_widget)
    actions_layout.setContentsMargins(4, 4, 4, 4)
    for w in (save_btn, commit_btn, reset_btn):
        actions_layout.addWidget(w.native if hasattr(w, "native") else w)
    actions_layout.addStretch(1)

    def _q(w):
        # magicgui Container/Widget → underlying QWidget
        return w.native if hasattr(w, "native") else w

    tabs.addTab(_q(controls_widget), "Δ sliders")
    tabs.addTab(_q(jog),              "jog")
    tabs.addTab(_q(view_panel),       "view")
    tabs.addTab(_q(lm_panel),         "landmarks")
    tabs.addTab(_q(actions_widget),   "save/commit/reset")

    # `pose` stays out of the tabs — it's always-visible info above them.
    right_root = QWidget()
    right_v = QVBoxLayout(right_root)
    right_v.setContentsMargins(2, 2, 2, 2)
    right_v.addWidget(abs_label.native if hasattr(abs_label, "native") else abs_label)
    right_v.addWidget(tabs, 1)
    viewer.window.add_dock_widget(right_root, name="zrot", area="right")
    _refresh_abs_label()

    print("[zrot] keyboard: Arrows = nudge tx/ty, PgUp/PgDn = tz, q/e = rz "
          "(by current 'jog' step). Shift+drag on the magenta layer = translate.")

    napari.run()

    if save_to is not None:
        save_transform(state["transform"], save_to)
    return state["transform"]
