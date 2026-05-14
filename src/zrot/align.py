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
    # interpolation 'nearest' is much faster for non-orthogonal slicing of
    # large previews. The dragging stays responsive; when you let go you can
    # toggle to 'linear' from the layer controls panel for a final visual check.
    sample_layer = viewer.add_image(
        sample_np,
        name=f"sample (level {level})",
        scale=sample_um,
        colormap="magenta",
        rendering="mip",
        contrast_limits=sample_clim,
        opacity=0.6,
        blending="additive",
        interpolation2d="nearest",
        interpolation3d="nearest",
    )

    # ---- WYSIWYG truth layer ---------------------------------------------
    # The only "live" view IS the scipy resample. There is no fake affine
    # render anymore. Slider change → debounced (200 ms) recompute → display.
    # Slower per tick but always honest: what you see = what gets saved.
    from scipy.ndimage import affine_transform as _scipy_affine
    from qtpy.QtCore import QTimer

    sample_layer.visible = False    # hide the lying affine layer permanently
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
        # No more cosmetic affine; schedule the truthful recompute.
        state["prewarp_timer"].start(200)   # debounce, restarts on each call

    _apply()

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

    # No prewarp toggle — the only view IS the truthful resample.

    viewer.window.add_dock_widget(abs_label, name="absolute pose", area="right")
    viewer.window.add_dock_widget(controls_widget, name="Δ from baseline", area="right")
    viewer.window.add_dock_widget(jog, name="jog Δ", area="right")
    viewer.window.add_dock_widget(commit_btn, name="commit", area="right")
    viewer.window.add_dock_widget(save_btn, name="save", area="right")
    viewer.window.add_dock_widget(reset_btn, name="reset Δ", area="right")
    _refresh_abs_label()

    print("[zrot] keyboard: Arrows = nudge tx/ty, PgUp/PgDn = tz, q/e = rz "
          "(by current 'jog' step). Shift+drag on the magenta layer = translate.")

    napari.run()

    if save_to is not None:
        save_transform(state["transform"], save_to)
    return state["transform"]
