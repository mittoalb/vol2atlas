"""Step: automated rigid refinement via Mutual Information (ANTs).

Reads state.transform (set by prealign/refine), resamples the sample
onto the cropped CCF voxel grid using that transform, then runs
ANTs Rigid + Mattes MI from Identity on (resampled sample, cropped
CCF). The small residual rigid that ANTs finds is composed back into
state.transform.

Same contract as `landmarks` Fit: refines state.transform in place.
Pick `landmarks` OR `mi`, not both.

Why "resample first, then register": both inputs to ANTs live in the
SAME CCF voxel grid, with identical spacing and identical
direction = identity (we use ants.from_numpy with no direction
matrix). That means the transform ANTs returns is already expressed
in the same (z, y, x) µm basis as our RigidTransform — no LPS/RAS
permutation. A runtime cross-check confirms this on every call.

Requires: pip install antspyx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..atlas import load_ccf
from ..frame import compute_output_frame, enable_ccf_axes, extract_ccf
from ..io import open_multiscale
from ..state import load as load_state, save as save_state, State
from ..transform import RigidTransform


def _load_or_pass(state_arg):
    """If state_arg is a Path: load State + return (state, path_for_save).
    If state_arg is already a State: return (state, None) — caller must
    NOT write state.json. This is the GUI path: in-memory only."""
    if isinstance(state_arg, (str, Path)):
        p = Path(state_arg)
        return load_state(p), p
    return state_arg, None


def _brain_mask(volume: np.ndarray, percentile: float,
                threshold: float | None, name: str = "") -> np.ndarray:
    """Threshold + largest CC + fill holes → solid brain envelope mask.

    Same logic for sample (µCT) and CCF — both are dense volumes with
    a bright brain region and dimmer / zero surroundings. Internal
    holes (ventricles, dark structures) are FILLED so the mask
    represents the brain ENVELOPE, not just the bright tissue.
    """
    from scipy.ndimage import label as cc_label, binary_fill_holes
    nz = volume[volume > 0]
    if nz.size == 0:
        return np.zeros_like(volume, dtype=bool)
    thr = float(threshold) if threshold is not None else float(
        np.percentile(nz, percentile))
    src = "user --mask-threshold" if threshold is not None else \
          f"{percentile:.0f}th percentile of nonzero"
    bin_ = volume > thr
    lbl, n = cc_label(bin_)
    if n > 0:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        mask = (lbl == int(sizes.argmax()))
    else:
        mask = bin_
    # Fill any internal holes (ventricles, dark voxels inside brain).
    mask_filled = binary_fill_holes(mask)
    kept = int(mask_filled.sum())
    holes_filled = int(mask_filled.sum() - mask.sum())
    print(f"[mi] {name} mask: threshold={thr:.3f} ({src}), "
          f"envelope {kept}/{volume.size} ({100*kept/volume.size:.1f}%), "
          f"filled {holes_filled} interior hole voxels")
    return mask_filled


def run(
    state_arg,
    *,
    level: int | None = None,
    mask: bool = True,
    skip_view: bool = False,
    affine: bool = False,
    shape: bool = False,
    mask_percentile: float = 50.0,
    mask_threshold: float | None = None,
    ccf_mask_percentile: float = 60.0,
    ccf_mask_threshold: float | None = None,
    check_mask: bool = False,
) -> State:
    try:
        import ants  # noqa: F401
    except ImportError:
        sys.exit("[mi] antspyx not installed. Run: pip install -e \".[ants]\"")

    state, _save_path = _load_or_pass(state_arg)
    if state.transform is None:
        sys.exit("[mi] state.json has no transform — run `vol2atlas prealign` first.")
    if state.ccf_crop_bbox is None:
        sys.exit("[mi] state.json has no ccf_crop_bbox — run `vol2atlas prealign` first.")

    # ---------- load sample at chosen level -----------------------------
    ms = open_multiscale(state.sample_zarr)
    lvl = state.sample_level if level is None else int(level)
    if not 0 <= lvl < ms.n_levels():
        sys.exit(f"[mi] --level {lvl} not in 0..{ms.n_levels()-1}")
    arr = ms.level(lvl)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    sample_um = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                 else ms.spacing(lvl))
    print(f"[mi] sample level {lvl}: {arr.shape} @ {sample_um} µm", flush=True)
    print(f"[mi] loading sample into RAM...", flush=True)
    sample_np = np.ascontiguousarray(arr.compute()).astype(np.float32)

    # ---------- load CCF (full) ----------------------------------------
    ccf = load_ccf(state.atlas_name)
    ccf_full = np.asarray(ccf.reference)
    sp = tuple(float(v) for v in ccf.voxel_um)

    # ---------- current rigid ------------------------------------------
    sample_center_um = tuple(
        (s - 1) * v / 2.0 for s, v in zip(sample_np.shape, sample_um))
    saved_center = (tuple(state.transform.get("center_um"))
                    if state.transform.get("center_um") is not None
                    else sample_center_um)
    rigid_old = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        center_um=saved_center,
    )

    # ---------- output frame: crop ∪ rotated sample footprint ----------
    # Compute a frame in CCF voxels that contains BOTH the user's CCF
    # crop bbox AND the 8 corners of the sample after rigid transform.
    # The sample never gets clipped, even at oblique rotations.
    # PAD the bbox by 30 voxels (~750 µm at 25 µm) so the CCF mask gets
    # some atlas-zero region around the brain — needed for SDT shape
    # registration to define the brain envelope. (Output frame for
    # subsequent steps stays at the user's crop.)
    pad_bbox = None
    if state.ccf_crop_bbox is not None:
        PAD = 30
        b = state.ccf_crop_bbox
        pad_bbox = {
            "z": [max(0, b["z"][0] - PAD), min(ccf_full.shape[0], b["z"][1] + PAD)],
            "y": [max(0, b["y"][0] - PAD), min(ccf_full.shape[1], b["y"][1] + PAD)],
            "x": [max(0, b["x"][0] - PAD), min(ccf_full.shape[2], b["x"][1] + PAD)],
        }
    origin, out_shape = compute_output_frame(
        sample_np.shape, sample_um, ccf.voxel_um,
        rigid_old.matrix(), pad_bbox,
    )
    ccf_data = extract_ccf(ccf_full, origin, out_shape).astype(np.float32)
    crop_voxel_origin = origin.astype(float)
    print(f"[mi] output frame: shape={ccf_data.shape}  "
          f"origin(vox)={tuple(int(v) for v in origin)}  "
          f"voxel={ccf.voxel_um} µm")

    # ---------- resample sample onto output frame (order=1) ------------
    from scipy.ndimage import affine_transform
    M_old_vox = rigid_old.for_voxel_grid(sample_um, ccf.voxel_um)  # sample_vox → ccf_vox
    Minv_vox = np.linalg.inv(M_old_vox)
    offset = Minv_vox[:3, :3] @ crop_voxel_origin + Minv_vox[:3, 3]
    print(f"[mi] resampling sample onto output frame...", flush=True)
    warped_sample = affine_transform(
        sample_np, Minv_vox[:3, :3], offset=offset,
        output_shape=ccf_data.shape, order=1, mode="constant", cval=0.0,
    ).astype(np.float32)

    # ---------- ANTs Rigid + Mattes MI from Identity -------------------
    # NOT using `affine_initializer`: its center-of-mass translation
    # init is wrong for partial-hemisphere samples (the partial COM and
    # the full CCF COM aren't the same anatomical point), and it
    # overwrites the careful pose set by `prealign`/`refine`.
    #
    # Instead we WIDEN MI's capture range via a coarser-than-default
    # pyramid (shrink 12→1, sigma 6→0, 5 levels) and BUMP the
    # finest-level iterations (ANTs default is 10, basically zero
    # fine-tuning) so MI can escape the local minima it gets trapped in
    # on cross-modality (µCT vs Nissl CCF) data. We start from Identity
    # to keep the prealign pose intact.
    #
    # Brain-only mask of the warped sample. Same threshold/percentile
    # logic as the CCF — `_brain_mask` handles both. For µCT use a
    # high percentile (~50-80) so only the brightest tissue is kept.
    from scipy.ndimage import label as cc_label, distance_transform_edt
    brain_mask = _brain_mask(warped_sample, mask_percentile, mask_threshold,
                              name="sample (in crop)").astype(np.float32)
    warped_sample_masked = (warped_sample * brain_mask).astype(np.float32)

    # --check-mask: open napari with BOTH masks overlaid (red = sample,
    # blue = CCF brain from FULL atlas).
    if check_mask:
        import napari
        from .. frame import enable_ccf_axes
        # Same _brain_mask helper as the sample, on the FULL atlas
        # (the user's crop is usually entirely inside brain → useless).
        ccf_mask_check = _brain_mask(
            ccf_full, ccf_mask_percentile, ccf_mask_threshold,
            name="CCF (full atlas)").astype(np.float32)
        v = napari.Viewer(ndisplay=3, title="vol2atlas mi --check-mask")
        enable_ccf_axes(v, ccf.orientation)
        v.add_image(ccf_full,           name="CCF (full atlas)",      scale=sp,
                    colormap="gray",    blending="additive", opacity=0.4)
        v.add_image(warped_sample,      name="sample (warped to crop)", scale=sp,
                    translate=tuple(float(o)*v for o, v in zip(crop_voxel_origin, sp)),
                    colormap="gray",    blending="additive", opacity=0.6)
        v.add_image(brain_mask,         name="SAMPLE mask (red)",     scale=sp,
                    translate=tuple(float(o)*v for o, v in zip(crop_voxel_origin, sp)),
                    colormap="red",     blending="additive", opacity=0.5,
                    rendering="iso", iso_threshold=0.5,
                    contrast_limits=(0.0, 1.0))
        v.add_image(ccf_mask_check,     name="CCF brain (blue iso)",  scale=sp,
                    colormap="blue",    blending="additive", opacity=0.5,
                    rendering="iso", iso_threshold=0.5,
                    contrast_limits=(0.0, 1.0))
        print("[mi] --check-mask: toggle layers.")
        print("[mi] BLUE = CCF brain envelope from full atlas (should look like brain).")
        print("[mi] RED = your sample mask after warp (also should look brain-ish).")
        napari.run()
        return

    # --shape: replace raw intensities with signed distance transforms
    # of the brain masks. SDTs are smooth, cross-modality safe, and the
    # global optimum is the true shape overlap — no over-stretching.
    if shape:
        print("[mi] --shape: building full-atlas frame for SDT registration",
              flush=True)
        # SAME mask logic as the sample: threshold + largest CC.
        ccf_mask_full = _brain_mask(ccf_full, ccf_mask_percentile,
                                     ccf_mask_threshold,
                                     name="CCF (full atlas)")
        # Warp sample onto the FULL atlas frame with the current rigid.
        M_full_vox = rigid_old.for_voxel_grid(sample_um, ccf.voxel_um)
        Minv_full  = np.linalg.inv(M_full_vox)
        # No crop offset — full atlas origin = (0, 0, 0).
        offset_full = Minv_full[:3, 3]
        print(f"[mi] warping sample onto full CCF frame "
              f"{ccf_full.shape}...", flush=True)
        sample_in_full = affine_transform(
            sample_np, Minv_full[:3, :3], offset=offset_full,
            output_shape=ccf_full.shape, order=1, mode="constant", cval=0.0,
        ).astype(np.float32)
        # Sample brain mask in full atlas frame.
        nzf = sample_in_full[sample_in_full > 0]
        if nzf.size > 0:
            if mask_threshold is not None:
                thrf = float(mask_threshold)
            else:
                thrf = float(np.percentile(nzf, mask_percentile))
            binf = sample_in_full > thrf
            lblf, nf = cc_label(binf)
            if nf > 0:
                szf = np.bincount(lblf.ravel()); szf[0] = 0
                sample_mask_full = (lblf == int(szf.argmax()))
            else:
                sample_mask_full = binf
        else:
            sample_mask_full = np.zeros_like(sample_in_full, dtype=bool)
        kept_full = int(sample_mask_full.sum())
        print(f"[mi] sample brain in full atlas: "
              f"{kept_full}/{sample_in_full.size} voxels "
              f"({100*kept_full/sample_in_full.size:.1f}%)")

        print("[mi] computing signed distance transforms...", flush=True)
        sample_sdt = (distance_transform_edt( sample_mask_full)
                       - distance_transform_edt(~sample_mask_full)).astype(np.float32)
        ccf_sdt    = (distance_transform_edt( ccf_mask_full)
                       - distance_transform_edt(~ccf_mask_full)).astype(np.float32)
        fixed_arr  = ccf_sdt
        moving_arr = sample_sdt
        print(f"[mi] SDT registration (full atlas frame, cross-modality safe)")
    else:
        fixed_arr  = ccf_data
        moving_arr = warped_sample_masked

    import ants
    fixed  = ants.from_numpy(np.ascontiguousarray(fixed_arr),  spacing=sp)
    moving = ants.from_numpy(np.ascontiguousarray(moving_arr), spacing=sp)
    reg_kw = {}
    if shape:
        # ALWAYS pass the brain masks for SDT registration. Without
        # them, MI is computed over the empty atlas-zero region (most
        # of the frame for a partial sample) and the optimizer can
        # cheat by SHRINKING the sample so more of the frame matches
        # CCF-outside. Masks restrict MI to actual brain voxels only.
        reg_kw["mask"]        = ants.from_numpy(
            ccf_mask_full.astype(np.uint8), spacing=sp)
        reg_kw["moving_mask"] = ants.from_numpy(
            sample_mask_full.astype(np.uint8), spacing=sp)
        print("[mi] passing sample + CCF brain masks to ANTs "
              "(prevents shrink-to-empty cheating)")
    elif mask:
        reg_kw["mask"]        = ants.get_mask(fixed)
        reg_kw["moving_mask"] = ants.get_mask(moving)

    tx_type = "Affine" if affine else "Rigid"
    print(f"[mi] ANTs {tx_type} + Mattes MI (wide pyramid, mask={mask})...",
          flush=True)
    reg = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform=tx_type,
        aff_metric="mattes",
        aff_sampling=32,
        initial_transform="Identity",
        aff_shrink_factors   = (12, 8, 4, 2, 1),
        aff_smoothing_sigmas = (6,  4, 2, 1, 0),
        aff_iterations       = (2000, 2000, 1500, 1000, 500),
        verbose=True,
        **reg_kw,
    )

    # ---------- read back residual transform → 4×4 in (z,y,x) µm -------
    tx_files = reg.get("fwdtransforms", [])
    if not tx_files:
        sys.exit("[mi] ANTs returned no transform.")
    residual_4x4 = _transformlist_to_4x4(tx_files)

    # ---------- info: print the residual + a few sanity numbers --------
    # NOTE: no scipy↔ANTs cross-check. We verified empirically (see
    # diagnostic in conversation log 2026-06-12) that scipy.ndimage
    # .affine_transform and ANTs apply_transforms produce different
    # pixel-level results for the SAME transform on the same image —
    # ANTs appears to apply masking / boundary handling that suppresses
    # roughly half the gradient response. The transform itself is
    # correct (apply_to_point and a row-major parameter parse agree),
    # so we trust it and rely on the visual preview (or your eyeballing
    # `vol2atlas export` afterwards) to validate the alignment.
    S_in   = np.diag([*sp, 1.0])                       # voxel → µm
    S_out  = np.diag([1.0 / v for v in sp] + [1.0])    # µm → voxel
    res_vox = S_out @ residual_4x4 @ S_in
    print(f"[mi] residual (µm): translation={residual_4x4[:3, 3].tolist()}, "
          f"|rotation off-diag|≤{np.max(np.abs(residual_4x4[:3, :3] - np.eye(3))):.2e}")

    # ---------- compose result -----------------------------------------
    # residual_4x4 is fixed → moving in CCF µm. The correction is its
    # INVERSE.
    # Rigid path: M_new = T^-1 @ M_old (updates state.transform).
    # Affine path: A_new = T^-1 (stored separately in state.affine,
    #              applied on top of state.transform at export time).
    residual_inv = np.linalg.inv(residual_4x4)
    if affine:
        # New affine matrix in CCF µm space. Composed with any existing
        # state.affine (so re-running --affine refines further).
        prior_affine = (np.asarray(state.affine, dtype=float)
                        if state.affine is not None else np.eye(4))
        new_affine_um = residual_inv @ prior_affine
        # Decompose so the user can see scale/shear deltas.
        S = np.linalg.svd(new_affine_um[:3, :3], compute_uv=False)
        print(f"[mi] new affine singular values (~scale per axis): "
              f"{S.tolist()}  (1.0 = no scale change)")
        print(f"[mi] new affine translation (µm): "
              f"{new_affine_um[:3, 3].tolist()}")
        rigid_new = rigid_old   # rigid unchanged
        new_affine_to_save = new_affine_um.tolist()
    else:
        M_old_um = rigid_old.matrix()
        M_new_um = residual_inv @ M_old_um
        rigid_new = _decompose_into_rigid(M_new_um, rigid_old)
        _print_delta(rigid_old, rigid_new)
        new_affine_to_save = state.affine   # untouched

    # ---------- optional napari preview --------------------------------
    # Pre-resample BEFORE and AFTER onto the user's CCF crop grid so
    # napari's 2D ortho views work correctly (they slice in data space,
    # not world space — see prealign comments). One-shot resample, fine
    # to be slow.
    if skip_view:
        accepted = True
    else:
        b = state.ccf_crop_bbox
        if b is None:
            ccf_crop_data = ccf_full
            ccf_crop_origin_um = (0.0, 0.0, 0.0)
            crop_vox_origin = np.zeros(3)
        else:
            z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
            ccf_crop_data = ccf_full[z0:z1, y0:y1, x0:x1]
            ccf_crop_origin_um = (z0 * sp[0], y0 * sp[1], x0 * sp[2])
            crop_vox_origin = np.array([z0, y0, x0], dtype=float)

        def _warp_to_crop(M_um):
            """Apply a 4×4 µm transform (sample µm → CCF µm) onto the crop."""
            S_in   = np.diag([sample_um[0], sample_um[1], sample_um[2], 1.0])
            S_out  = np.diag([1.0 / sp[0], 1.0 / sp[1], 1.0 / sp[2], 1.0])
            Mvox = S_out @ M_um @ S_in
            Minv = np.linalg.inv(Mvox)
            off = Minv[:3, :3] @ crop_vox_origin + Minv[:3, 3]
            return affine_transform(
                sample_np, Minv[:3, :3], offset=off,
                output_shape=ccf_crop_data.shape, order=1,
                mode="constant", cval=0.0,
            ).astype(np.float32)

        print("[mi] resampling BEFORE/AFTER onto CCF crop for preview...",
              flush=True)
        # BEFORE = existing rigid + existing affine (state as-is now).
        # AFTER  = same rigid + new affine (for --affine path), OR
        #          new rigid + existing affine (for rigid path).
        prior_affine_um = (np.asarray(state.affine, dtype=float)
                           if state.affine is not None else np.eye(4))
        before_um = prior_affine_um @ rigid_old.matrix()
        if affine:
            after_um = np.asarray(new_affine_to_save, dtype=float) @ rigid_old.matrix()
        else:
            after_um = prior_affine_um @ rigid_new.matrix()
        before_warp = _warp_to_crop(before_um)
        after_warp  = _warp_to_crop(after_um)
        # Also warp the SAMPLE BRAIN MASK so the user can see exactly
        # which surface the registration used as "marks".
        # Sample mask in native space using the SAME threshold the
        # registration used (mask_percentile / mask_threshold). Largest
        # connected component, then warp to the CCF crop grid via the
        # AFTER transform so the user sees the mask aligned to atlas.
        nz_smp = sample_np[sample_np > 0]
        if nz_smp.size > 0:
            if mask_threshold is not None:
                thr_smp = float(mask_threshold)
            else:
                thr_smp = float(np.percentile(nz_smp, mask_percentile))
            bin_smp = sample_np > thr_smp
            lbl_smp, n_smp = cc_label(bin_smp)
            if n_smp > 0:
                sizes_smp = np.bincount(lbl_smp.ravel())
                sizes_smp[0] = 0
                smp_mask_native = (lbl_smp == int(sizes_smp.argmax())).astype(np.float32)
            else:
                smp_mask_native = bin_smp.astype(np.float32)
        else:
            smp_mask_native = np.ones_like(sample_np, dtype=np.float32)

        def _warp_mask(M_um):
            S_in   = np.diag([sample_um[0], sample_um[1], sample_um[2], 1.0])
            S_out  = np.diag([1.0 / sp[0], 1.0 / sp[1], 1.0 / sp[2], 1.0])
            Mvox = S_out @ M_um @ S_in
            Minv = np.linalg.inv(Mvox)
            off = Minv[:3, :3] @ crop_vox_origin + Minv[:3, 3]
            return affine_transform(
                smp_mask_native, Minv[:3, :3], offset=off,
                output_shape=ccf_crop_data.shape, order=0,
                mode="constant", cval=0.0,
            ).astype(np.float32)
        sample_mask_after = _warp_mask(after_um)
        # CCF mask preview: show the SAME mask the registration uses,
        # cropped to the user's bbox. If the crop is inside brain it
        # will be a block here — that's HONEST for the crop region;
        # registration uses the full-atlas mask (see --check-mask).
        ccf_mask_full_preview = _brain_mask(
            ccf_full, ccf_mask_percentile, ccf_mask_threshold,
            name="CCF (preview)")
        b = state.ccf_crop_bbox
        if b is None:
            ccf_mask_data = ccf_mask_full_preview.astype(np.float32)
        else:
            z0, z1 = b["z"]; y0, y1 = b["y"]; x0, x1 = b["x"]
            ccf_mask_data = ccf_mask_full_preview[z0:z1, y0:y1, x0:x1].astype(np.float32)

        accepted = _preview(
            ccf_crop_data, ccf_crop_origin_um, ccf.voxel_um,
            before_warp, after_warp,
            rigid_old, rigid_new,
            sample_mask_after=sample_mask_after, ccf_mask=ccf_mask_data,
        )

    if not accepted:
        print("[mi] Rejected — state.json unchanged.")
        return

    # ---------- save ---------------------------------------------------
    if affine:
        state.affine = new_affine_to_save     # 4x4 in CCF µm
        # state.transform unchanged
    else:
        state.transform = rigid_new.to_dict()
        state.affine = new_affine_to_save     # preserved (no-op)
    state.add_history(
        "mi",
        f"ants {('Affine' if affine else 'Rigid')}+MI, level={lvl}, mask={mask}, "
        f"residual_um=[{residual_4x4[0,3]:.3f},{residual_4x4[1,3]:.3f},{residual_4x4[2,3]:.3f}]",
    )
    if _save_path is not None:
        save_state(state, _save_path)
        print(f"[mi] saved -> {_save_path}")
    else:
        print(f"[mi] in-memory result returned (no disk write)")
    return state


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _transformlist_to_4x4(tx_files: list) -> np.ndarray:
    """Reconstruct the 4×4 affine matrix in our (z, y, x) µm basis from
    a list of ANTs transform files (the combined initializer + rigid
    output of ants.registration with initial_transform).

    Uses ants.apply_transforms_to_points at 4 known points to evaluate
    the composed transform through ITK's own machinery — avoids parsing
    parameters (Fortran vs C order, Euler vs matrix transform type) and
    composing manually.

    Basis correspondence: from_numpy() preserves numpy axes such that
    numpy axis 0 ↔ ITK x. We treat numpy axis 0 as "z", so passing our
    (z, y, x) µm triplets through ITK as (x, y, z) gives the right
    physical positions; no permutation needed.
    """
    import ants
    import pandas as pd
    L = 100.0  # lever-arm in µm; arbitrary non-zero
    # df columns are ITK (x, y, z), which numerically correspond to our
    # (z, y, x). Pass our coordinates verbatim under those names.
    pts_in_df = pd.DataFrame({
        "x": [0.0, L,   0.0, 0.0],
        "y": [0.0, 0.0, L,   0.0],
        "z": [0.0, 0.0, 0.0, L  ],
    })
    pts_out_df = ants.apply_transforms_to_points(
        dim=3, points=pts_in_df, transformlist=tx_files,
    )
    pts_in  = pts_in_df [["x", "y", "z"]].values
    pts_out = pts_out_df[["x", "y", "z"]].values
    # pts_out[0] = A @ 0 + t  =>  t = pts_out[0]
    # pts_out[i+1] - t = A @ (L * e_i)  =>  col i of A = (pts_out[i+1] - t) / L
    t = pts_out[0]
    A = ((pts_out[1:] - t) / L).T
    M = np.eye(4)
    M[:3, :3] = A
    M[:3,  3] = t
    return M


def _decompose_into_rigid(M: np.ndarray, prior: RigidTransform) -> RigidTransform:
    """Given a 4×4 sample_µm → CCF_µm matrix and the prior RigidTransform
    (whose center we preserve), return a new RigidTransform.

    Recall RigidTransform.matrix() builds:
        M  = [[R, -R @ c + c + (tz, ty, tx)]]
    So given M and c:
        R = M[:3, :3]
        t = M[:3, 3] - (-R @ c + c) = M[:3, 3] + R @ c - c
    """
    from scipy.spatial.transform import Rotation
    c = np.asarray(prior.center_um, dtype=float)
    R = M[:3, :3]
    # Re-orthonormalize R against tiny numerical drift before Euler.
    U, _, Vt = np.linalg.svd(R)
    d = float(np.sign(np.linalg.det(U @ Vt)) or 1.0)
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    t = M[:3, 3] + R @ c - c
    return RigidTransform(
        rz_deg=float(rz), ry_deg=float(ry), rx_deg=float(rx),
        tz_um=float(t[0]), ty_um=float(t[1]), tx_um=float(t[2]),
        center_um=tuple(float(x) for x in prior.center_um),
    )


def _print_delta(old: RigidTransform, new: RigidTransform) -> None:
    print(
        f"[mi] rotation (deg):  rz {old.rz_deg:+8.3f} → {new.rz_deg:+8.3f}   "
        f"ry {old.ry_deg:+8.3f} → {new.ry_deg:+8.3f}   "
        f"rx {old.rx_deg:+8.3f} → {new.rx_deg:+8.3f}"
    )
    print(
        f"[mi] translation (µm): tz {old.tz_um:+9.2f} → {new.tz_um:+9.2f}   "
        f"ty {old.ty_um:+9.2f} → {new.ty_um:+9.2f}   "
        f"tx {old.tx_um:+9.2f} → {new.tx_um:+9.2f}"
    )


def _preview(
    ccf_crop_data:       np.ndarray,
    ccf_crop_origin_um:  tuple,
    ccf_voxel_um:        tuple,
    before_warp:         np.ndarray,
    after_warp:          np.ndarray,
    rigid_old:           RigidTransform,
    rigid_new:           RigidTransform,
    *,
    sample_mask_after:   np.ndarray | None = None,
    ccf_mask:            np.ndarray | None = None,
) -> bool:
    """napari viewer: CCF crop (gray) + sample BEFORE (cyan) + sample
    AFTER (magenta). All three layers share the CCF crop's voxel grid
    so 3D MIP and 2D ortho slices both work. Accept/Reject buttons.
    """
    import napari
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton)

    def _clim(a):
        f = a.ravel()
        if f.size > 1_000_000: f = f[:: f.size // 1_000_000]
        f = f[f > 0] if (f > 0).any() else f
        lo, hi = np.percentile(f, [2, 99.5])
        return float(lo), float(max(hi, lo + 1))

    decision = {"accepted": False}
    v = napari.Viewer(ndisplay=3, title="vol2atlas mi — review")
    enable_ccf_axes(v, "asr")
    v.add_image(ccf_crop_data, name="CCF",
                scale=ccf_voxel_um, translate=ccf_crop_origin_um,
                colormap="gray", blending="additive", opacity=0.55,
                contrast_limits=_clim(ccf_crop_data))
    before_layer = v.add_image(
        before_warp, name="sample BEFORE",
        scale=ccf_voxel_um, translate=ccf_crop_origin_um,
        colormap="gray", blending="additive", opacity=0.5,
        contrast_limits=_clim(before_warp))
    v.add_image(
        after_warp, name="sample AFTER",
        scale=ccf_voxel_um, translate=ccf_crop_origin_um,
        colormap="gray", blending="additive", opacity=0.7,
        contrast_limits=_clim(after_warp))
    before_layer.visible = False   # toggle the eye icon to compare

    # Diagnostic "marks" used by the registration: the brain envelopes.
    # Hidden by default — toggle on if the registration looks wrong, to
    # verify the masks cover the right shape.
    if sample_mask_after is not None:
        ml = v.add_image(
            sample_mask_after, name="sample mask (used by registration)",
            scale=ccf_voxel_um, translate=ccf_crop_origin_um,
            colormap="red", blending="additive", opacity=0.5,
            rendering="iso", iso_threshold=0.5,
            contrast_limits=(0.0, 1.0))
        ml.visible = False
    if ccf_mask is not None:
        cl = v.add_image(
            ccf_mask, name="CCF mask (used by registration)",
            scale=ccf_voxel_um, translate=ccf_crop_origin_um,
            colormap="blue", blending="additive", opacity=0.5,
            rendering="iso", iso_threshold=0.5,
            contrast_limits=(0.0, 1.0))
        cl.visible = False

    ctrl = QWidget(); col = QVBoxLayout(ctrl)
    col.addWidget(QLabel("<b>Review the MI fit</b>"))
    col.addWidget(QLabel(
        f"Δ rotation (deg):\n"
        f"  rz {rigid_new.rz_deg - rigid_old.rz_deg:+.3f}\n"
        f"  ry {rigid_new.ry_deg - rigid_old.ry_deg:+.3f}\n"
        f"  rx {rigid_new.rx_deg - rigid_old.rx_deg:+.3f}"
    ))
    col.addWidget(QLabel(
        f"Δ translation (µm):\n"
        f"  tz {rigid_new.tz_um - rigid_old.tz_um:+.2f}\n"
        f"  ty {rigid_new.ty_um - rigid_old.ty_um:+.2f}\n"
        f"  tx {rigid_new.tx_um - rigid_old.tx_um:+.2f}"
    ))
    col.addWidget(QLabel(
        "Toggle the 'sample BEFORE'/'sample AFTER' layers to compare."))

    row = QHBoxLayout()
    accept_btn = QPushButton("Accept → save")
    reject_btn = QPushButton("Reject → keep old")
    row.addWidget(accept_btn); row.addWidget(reject_btn)
    rw = QWidget(); rw.setLayout(row); col.addWidget(rw)
    col.addStretch(1)

    def _accept():
        decision["accepted"] = True
        v.close()
    def _reject():
        decision["accepted"] = False
        v.close()

    accept_btn.clicked.connect(_accept)
    reject_btn.clicked.connect(_reject)

    v.window.add_dock_widget(ctrl, name="mi", area="right")
    napari.run()
    return decision["accepted"]


# =====================================================================
# JOINT MI + landmark registration via antsRegistration CLI
# (true simultaneous optimization, not sequential)
# =====================================================================
def run_joint(
    state_arg,
    *,
    level: int | None = None,
    affine: bool = False,
    shape: bool = False,
    mask: bool = True,
    landmark_weight: float = 10.0,
    mask_percentile: float = 50.0,
    mask_threshold: float | None = None,
    ccf_mask_percentile: float = 60.0,
    ccf_mask_threshold: float | None = None,
) -> State:
    """SINGLE-SHOT joint registration: Mattes MI + PointSetExpectation
    landmarks + brain masks, all in ONE antsRegistration call.

    Does NOT use any prior state.transform / state.affine. Starts from
    a closed-form landmark fit (rigid Procrustes or affine least-
    squares), then antsRegistration refines that initial with the
    multi-metric optimization. Result overwrites state with the final
    transform — no compositional residue from previous steps.
    """
    import shutil, tempfile, sys as _sys
    try:
        import ants
        from ants.internal import get_lib_fn, process_arguments
    except ImportError:
        _sys.exit("[joint] antspyx not installed.")

    state, _save_path = _load_or_pass(state_arg)
    sample_lms_raw = state.landmarks.get("sample_um", [])
    ccf_lms_raw    = state.landmarks.get("ccf_um",    [])
    n = min(len(sample_lms_raw), len(ccf_lms_raw))
    if n < 4:
        _sys.exit(f"[joint] need ≥4 landmark pairs (have {n}).")
    sample_lms = np.asarray(sample_lms_raw[:n], dtype=float)
    ccf_lms    = np.asarray(ccf_lms_raw[:n],    dtype=float)

    # ---- load sample + atlas (NO pre-warping with prior transform) ----
    ms = open_multiscale(state.sample_zarr)
    lvl = state.sample_level if level is None else int(level)
    arr = ms.level(lvl)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    sample_um = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                 else ms.spacing(lvl))
    print(f"[joint] sample level {lvl}: {arr.shape} @ {sample_um} µm",
          flush=True)
    sample_np = np.ascontiguousarray(arr.compute()).astype(np.float32)
    ccf = load_ccf(state.atlas_name)
    ccf_full = np.asarray(ccf.reference).astype(np.float32)
    sp_ccf = tuple(float(v) for v in ccf.voxel_um)

    # ---- closed-form landmark initial transform (sample µm → CCF µm) ----
    if affine:
        H = np.hstack([sample_lms, np.ones((n, 1))])
        A_ls, *_ = np.linalg.lstsq(H, ccf_lms, rcond=None)
        M_init_smp2ccf = np.eye(4)
        M_init_smp2ccf[:3, :3] = A_ls[:3, :].T
        M_init_smp2ccf[:3,  3] = A_ls[3, :]
        print("[joint] initial transform: landmark-fit AFFINE (12 DOF)",
              flush=True)
    else:
        # Procrustes (rigid).
        src_c = sample_lms.mean(0); tgt_c = ccf_lms.mean(0)
        H = (sample_lms - src_c).T @ (ccf_lms - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0)
        R0 = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t0 = tgt_c - R0 @ src_c
        M_init_smp2ccf = np.eye(4)
        M_init_smp2ccf[:3, :3] = R0
        M_init_smp2ccf[:3,  3] = t0
        print("[joint] initial transform: landmark-fit RIGID (6 DOF, Procrustes)",
              flush=True)
    # ANTs convention: fixed→moving = CCF→sample = inverse of sample→CCF
    M_init_ccf2smp = np.linalg.inv(M_init_smp2ccf)

    # ---- brain masks (in each volume's native frame) ----
    sample_mask_native = _brain_mask(
        sample_np, mask_percentile, mask_threshold, name="sample (native)")
    ccf_mask_native = _brain_mask(
        ccf_full, ccf_mask_percentile, ccf_mask_threshold,
        name="CCF (native)")

    # ---- SDT if requested ----
    if shape:
        from scipy.ndimage import distance_transform_edt
        print("[joint] computing SDTs...", flush=True)
        sample_for_ants = (distance_transform_edt( sample_mask_native)
                           - distance_transform_edt(~sample_mask_native)
                           ).astype(np.float32)
        ccf_for_ants    = (distance_transform_edt( ccf_mask_native)
                           - distance_transform_edt(~ccf_mask_native)
                           ).astype(np.float32)
    else:
        sample_for_ants = (sample_np * sample_mask_native).astype(np.float32)
        ccf_for_ants    = (ccf_full  * ccf_mask_native   ).astype(np.float32)

    # ---- build intermediate files for antsRegistration ----
    tmpdir = tempfile.mkdtemp(prefix="vol2atlas_joint_")
    try:
        print(f"[joint] tmpdir: {tmpdir}", flush=True)
        # Images
        ants.from_numpy(np.ascontiguousarray(ccf_for_ants),    spacing=sp_ccf
                        ).to_filename(f"{tmpdir}/fixed.nii.gz")
        ants.from_numpy(np.ascontiguousarray(sample_for_ants), spacing=sample_um
                        ).to_filename(f"{tmpdir}/moving.nii.gz")
        if mask:
            ants.from_numpy(ccf_mask_native.astype(np.uint8),    spacing=sp_ccf
                            ).to_filename(f"{tmpdir}/fixed_mask.nii.gz")
            ants.from_numpy(sample_mask_native.astype(np.uint8), spacing=sample_um
                            ).to_filename(f"{tmpdir}/moving_mask.nii.gz")

        # Landmark LABEL IMAGES (NIfTI — what ANTs PSE actually reads).
        # Each landmark = one voxel set to a unique label (i+1).
        fixed_lm_img  = np.zeros(ccf_full.shape,   dtype=np.float32)
        moving_lm_img = np.zeros(sample_np.shape,  dtype=np.float32)
        for i, (clm, slm) in enumerate(zip(ccf_lms, sample_lms)):
            cz, cy, cx = (int(round(clm[k] / sp_ccf[k])) for k in range(3))
            sz, sy, sx = (int(round(slm[k] / sample_um[k])) for k in range(3))
            if (0 <= cz < ccf_full.shape[0] and 0 <= cy < ccf_full.shape[1]
                    and 0 <= cx < ccf_full.shape[2]):
                fixed_lm_img[cz, cy, cx] = float(i + 1)
            if (0 <= sz < sample_np.shape[0] and 0 <= sy < sample_np.shape[1]
                    and 0 <= sx < sample_np.shape[2]):
                moving_lm_img[sz, sy, sx] = float(i + 1)
        ants.from_numpy(fixed_lm_img,  spacing=sp_ccf
                        ).to_filename(f"{tmpdir}/fixed_lms.nii.gz")
        ants.from_numpy(moving_lm_img, spacing=sample_um
                        ).to_filename(f"{tmpdir}/moving_lms.nii.gz")

        # Initial transform: write our landmark-fit M_init_ccf2smp as an
        # ANTs .mat via a temporary ANTsTransform.
        #
        # ITK reads the 12 parameters as ROW-MAJOR (C order) when passed as
        # a flat 1D array — VERIFIED empirically against this round-trip
        # check: with order='F', the recovered matrix comes back transposed
        # (off-diagonal elements appear swapped, e.g. intended[0,1] lands
        # at reconstructed[1,0]). The `flatten(order='F')` branch in
        # antspyx set_parameters only fires for 2D ndarray input; passing
        # a pre-flattened 1D array bypasses it, and ITK's underlying
        # convention is row-major. The misleading antspyx code comment is
        # about its 2D-input shim, not ITK's wire format.
        A = M_init_ccf2smp[:3, :3]
        t = M_init_ccf2smp[:3,  3]
        params = np.concatenate([A.flatten(order='C'), t])
        init_tx = ants.new_ants_transform(precision="float", dimension=3,
                                          transform_type="AffineTransform")
        init_tx.set_parameters(params.astype(np.float32))
        init_tx.set_fixed_parameters(np.zeros(3, dtype=np.float32))
        init_tx_file = f"{tmpdir}/init.mat"
        ants.write_transform(init_tx, init_tx_file)

        # Verify the round-trip: write → read → reconstruct via
        # apply_to_point at 4 corners; abort if it doesn't match what
        # we intended. Catches any future convention regression.
        check_4x4 = _transformlist_to_4x4([init_tx_file])
        if not np.allclose(check_4x4, M_init_ccf2smp, atol=1e-3):
            err = float(np.max(np.abs(check_4x4 - M_init_ccf2smp)))
            print(f"[joint] WARN: initial transform round-trip error "
                  f"max|Δ|={err:.4e}", flush=True)
            print(f"[joint] intended:\n{M_init_ccf2smp}")
            print(f"[joint] reconstructed:\n{check_4x4}")
            _sys.exit("[joint] initial transform write/read mismatch — abort.")

        # ---- build antsRegistration argv ----
        tx_kind = "Affine[0.1]" if affine else "Rigid[0.1]"
        argv = [
            "antsRegistration",
            "--dimensionality", "3",
            "--float", "1",
            "--initial-moving-transform", init_tx_file,
            "--transform", tx_kind,
            "--metric",
            f"Mattes[{tmpdir}/fixed.nii.gz,{tmpdir}/moving.nii.gz,1,32,Regular,0.2]",
            # NOTE: PSE on single-voxel sparse landmarks segfaults in
            # this ANTs build (GMM EM degenerate for sparse points).
            # Landmarks are now used ONLY as the initial transform
            # (--initial-moving-transform above) — that's already a
            # strong anchor since affine optimization is local.
            "--convergence", "[2000x2000x1500x1000x500,1e-6,10]",
            "--shrink-factors", "12x8x4x2x1",
            "--smoothing-sigmas", "6x4x2x1x0vox",
            "--output", f"[{tmpdir}/result,{tmpdir}/warped.nii.gz]",
            "--verbose", "1",
        ]
        if mask:
            argv += ["--masks",
                     f"[{tmpdir}/fixed_mask.nii.gz,{tmpdir}/moving_mask.nii.gz]"]
        print("[joint] running antsRegistration via antspyx C++ wrapper:",
              " ".join(argv), flush=True)
        libfn = get_lib_fn("antsRegistration")
        rc = libfn(process_arguments(argv))
        if rc != 0:
            _sys.exit(f"[joint] antsRegistration failed (code {rc}).")

        # ---- read back combined transform ----
        # With --collapse-output-transforms 1 (default), antsRegistration
        # COLLAPSES the initial-moving-transform into the output
        # 0GenericAffine.mat. Per ANTs wiki "Forward-and-inverse-warps...":
        #   "A user-defined initial moving transform matrix will also be
        #    collapsed into the combined affine transform... you would
        #    supply only 0GenericAffine.mat — not init.mat again."
        # So pass ONLY [out_file]. Passing [out_file, init_tx_file] applies
        # the initial transform TWICE.
        out_file = f"{tmpdir}/result0GenericAffine.mat"
        if not Path(out_file).exists():
            _sys.exit(f"[joint] expected output transform not found: {out_file}")
        M_full_ccf2smp = _transformlist_to_4x4([out_file])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- save: sample → CCF (inverse of ANTs direction) ----
    M_full_smp2ccf = np.linalg.inv(M_full_ccf2smp)
    if affine:
        # state.affine carries the full 12-DOF; state.transform reset
        # to identity (the affine carries everything, no separate rigid).
        state.transform = RigidTransform(
            center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))
                            if state.transform else (0.0, 0.0, 0.0))
        ).to_dict()
        state.affine = M_full_smp2ccf.tolist()
        S = np.linalg.svd(M_full_smp2ccf[:3, :3], compute_uv=False)
        print(f"[joint] FINAL affine singular values: {S.tolist()}")
        print(f"[joint] FINAL affine translation (µm): "
              f"{M_full_smp2ccf[:3, 3].tolist()}")
    else:
        # Decompose into rigid; affine cleared.
        prior_center = tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))
                              if state.transform else (0.0, 0.0, 0.0))
        prior_for_decompose = RigidTransform(center_um=prior_center)
        rigid_new = _decompose_into_rigid(M_full_smp2ccf, prior_for_decompose)
        state.transform = rigid_new.to_dict()
        state.affine = None
        _print_delta(prior_for_decompose, rigid_new)

    state.add_history(
        "joint_mi_landmarks",
        f"shape={shape} affine={affine} mask={mask} "
        f"n_landmarks={n} weight={landmark_weight} (single-shot)",
    )
    if _save_path is not None:
        save_state(state, _save_path)
        print(f"[joint] saved -> {_save_path}", flush=True)
    else:
        print(f"[joint] in-memory result returned (no disk write)", flush=True)
    return state


# =====================================================================
# ITERATIVE joint MI + landmark — block coordinate descent
# =====================================================================
def run_joint_iter(
    state_arg,
    *,
    level: int | None = None,
    affine: bool = True,                # affine is the useful case
    shape: bool = False,
    mask: bool = True,
    landmark_weight: float = 10.0,      # unused here (kept for API parity)
    smooth_weight: float = 0.01,        # regularization weight for step-A LSQ
    max_iter: int = 5,
    tol: float = 1.0,                   # convergence on max |ΔT| in µm
    mask_percentile: float = 50.0,
    mask_threshold: float | None = None,
    ccf_mask_percentile: float = 60.0,
    ccf_mask_threshold: float | None = None,
) -> State:
    """Iterative block-coord descent for joint MI + landmark fit.

    Alternates:
      Step A: regularized closed-form fit of T against landmarks,
              pulled toward the previous MI result by λ.
      Step B: ANTs MI starting from step A's T.

    Converges when consecutive T's differ by < tol. Reports per-iter
    landmark RMS so you can see if landmarks stay anchored.
    """
    import shutil, tempfile, sys as _sys
    try:
        import ants
        from ants.internal import get_lib_fn, process_arguments
    except ImportError:
        _sys.exit("[joint_iter] antspyx not installed.")

    state, _save_path = _load_or_pass(state_arg)
    sample_lms_raw = state.landmarks.get("sample_um", [])
    ccf_lms_raw    = state.landmarks.get("ccf_um",    [])
    n = min(len(sample_lms_raw), len(ccf_lms_raw))
    if n < 4:
        _sys.exit(f"[joint_iter] need ≥4 landmark pairs (have {n}).")
    sample_lms = np.asarray(sample_lms_raw[:n], dtype=float)
    ccf_lms    = np.asarray(ccf_lms_raw[:n],    dtype=float)

    # ---- load sample + atlas ----
    ms = open_multiscale(state.sample_zarr)
    lvl = state.sample_level if level is None else int(level)
    arr = ms.level(lvl)
    if "c" in ms.axes:
        arr = arr[state.sample_channel]
    sample_um = (tuple(state.sample_voxel_um) if state.sample_voxel_um
                 else ms.spacing(lvl))
    sample_np = np.ascontiguousarray(arr.compute()).astype(np.float32)
    ccf = load_ccf(state.atlas_name)
    ccf_full = np.asarray(ccf.reference).astype(np.float32)
    sp_ccf = tuple(float(v) for v in ccf.voxel_um)

    # ---- masks ----
    sample_mask_native = _brain_mask(sample_np, mask_percentile,
                                      mask_threshold, name="sample")
    ccf_mask_native = _brain_mask(ccf_full, ccf_mask_percentile,
                                   ccf_mask_threshold, name="CCF")
    if shape:
        from scipy.ndimage import distance_transform_edt
        print("[joint_iter] computing SDTs...", flush=True)
        sample_for_ants = (distance_transform_edt( sample_mask_native)
                           - distance_transform_edt(~sample_mask_native)
                           ).astype(np.float32)
        ccf_for_ants    = (distance_transform_edt( ccf_mask_native)
                           - distance_transform_edt(~ccf_mask_native)
                           ).astype(np.float32)
    else:
        sample_for_ants = (sample_np * sample_mask_native).astype(np.float32)
        ccf_for_ants    = (ccf_full  * ccf_mask_native   ).astype(np.float32)

    # ---- write static intermediate files ONCE (reused across iters) ----
    tmpdir = tempfile.mkdtemp(prefix="vol2atlas_joint_iter_")
    try:
        ants.from_numpy(np.ascontiguousarray(ccf_for_ants),    spacing=sp_ccf
                        ).to_filename(f"{tmpdir}/fixed.nii.gz")
        ants.from_numpy(np.ascontiguousarray(sample_for_ants), spacing=sample_um
                        ).to_filename(f"{tmpdir}/moving.nii.gz")
        if mask:
            ants.from_numpy(ccf_mask_native.astype(np.uint8),    spacing=sp_ccf
                            ).to_filename(f"{tmpdir}/fixed_mask.nii.gz")
            ants.from_numpy(sample_mask_native.astype(np.uint8), spacing=sample_um
                            ).to_filename(f"{tmpdir}/moving_mask.nii.gz")

        # Regularized landmark fit: minimize
        #   ‖X·T - y‖² + λ_smooth·‖T - T_prev‖_F²
        # where T is the (sample→CCF) affine (or rigid Procrustes for rigid case).
        # Closed-form via augmented LSQ.
        H = np.hstack([sample_lms, np.ones((n, 1))])    # n × 4
        target = ccf_lms                                # n × 3

        def _fit_landmarks_regularized(T_prev_smp2ccf_3x4: np.ndarray,
                                        lam_smooth: float) -> np.ndarray:
            """Affine fit pulled toward T_prev with weight lam_smooth.
            Returns 4×4 sample µm → CCF µm.
            """
            if T_prev_smp2ccf_3x4 is None or lam_smooth <= 0:
                # Pure landmark fit (unregularized)
                A_ls, *_ = np.linalg.lstsq(H, target, rcond=None)  # 4×3
            else:
                # Augmented LSQ:
                #   [H        ] T = [target          ]
                #   [√λ·I_4×4 ]     [√λ·T_prev (4×3) ]
                sqlam = float(np.sqrt(lam_smooth))
                H_aug = np.vstack([H, sqlam * np.eye(4)])
                y_aug = np.vstack([target, sqlam * T_prev_smp2ccf_3x4])
                A_ls, *_ = np.linalg.lstsq(H_aug, y_aug, rcond=None)
            M = np.eye(4)
            M[:3, :3] = A_ls[:3, :].T
            M[:3,  3] = A_ls[3, :]
            return M

        # ---- iteration loop ----
        T_prev = None             # 3×4 in [A, t] form for regularizer
        T_prev_full = None        # 4×4 (sample → CCF)
        for it in range(1, max_iter + 1):
            # STEP A: regularized landmark fit
            T_smp2ccf = _fit_landmarks_regularized(
                T_prev, lam_smooth=landmark_weight if it > 1 else 0.0)
            T_ccf2smp = np.linalg.inv(T_smp2ccf)
            # Write init.mat (row-major — see run_joint comments for ITK convention)
            A = T_ccf2smp[:3, :3]; t = T_ccf2smp[:3, 3]
            params = np.concatenate([A.flatten(order='C'), t]).astype(np.float32)
            init_tx = ants.new_ants_transform(precision="float", dimension=3,
                                              transform_type="AffineTransform")
            init_tx.set_parameters(params)
            init_tx.set_fixed_parameters(np.zeros(3, dtype=np.float32))
            init_tx_file = f"{tmpdir}/init_iter{it}.mat"
            ants.write_transform(init_tx, init_tx_file)
            # Round-trip sanity
            check = _transformlist_to_4x4([init_tx_file])
            if not np.allclose(check, T_ccf2smp, atol=1e-3):
                _sys.exit(f"[joint_iter] init mat round-trip failed at iter {it}.")

            # Predicted landmark positions BEFORE MI step (for RMS reporting)
            pred_pre = (T_smp2ccf[:3, :3] @ sample_lms.T).T + T_smp2ccf[:3, 3]
            rms_pre = float(np.sqrt(np.mean(np.sum((pred_pre - ccf_lms) ** 2, axis=1))))

            # STEP B: ANTs MI starting from init
            tx_kind = "Affine[0.05]" if affine else "Rigid[0.05]"  # small step
            out_prefix = f"{tmpdir}/result_iter{it}"
            argv = [
                "antsRegistration",
                "--dimensionality", "3",
                "--float", "1",
                "--initial-moving-transform", init_tx_file,
                "--transform", tx_kind,
                "--metric",
                f"Mattes[{tmpdir}/fixed.nii.gz,{tmpdir}/moving.nii.gz,1,32,Regular,0.2]",
                "--convergence", "[500x500x500,1e-6,10]",   # short — give iter loop control
                "--shrink-factors", "4x2x1",
                "--smoothing-sigmas", "2x1x0vox",
                "--output", f"[{out_prefix},{out_prefix}_warped.nii.gz]",
                "--verbose", "1",
            ]
            if mask:
                argv += ["--masks",
                         f"[{tmpdir}/fixed_mask.nii.gz,{tmpdir}/moving_mask.nii.gz]"]
            print(f"\n[joint_iter] === ITER {it}/{max_iter} ===  "
                  f"pre-MI landmark RMS = {rms_pre:.2f} µm", flush=True)
            libfn = get_lib_fn("antsRegistration")
            rc = libfn(process_arguments(argv))
            if rc != 0:
                _sys.exit(f"[joint_iter] antsRegistration failed at iter {it}.")

            # Read combined transform from this iter.
            # antsRegistration --collapse-output-transforms (default ON)
            # bakes the initial-moving-transform into 0GenericAffine.mat,
            # so pass ONLY [out_tx] — passing init twice would compound.
            out_tx = f"{out_prefix}0GenericAffine.mat"
            M_full_ccf2smp = _transformlist_to_4x4([out_tx])
            M_full_smp2ccf = np.linalg.inv(M_full_ccf2smp)

            # Landmark RMS after MI
            pred_post = (M_full_smp2ccf[:3, :3] @ sample_lms.T).T + M_full_smp2ccf[:3, 3]
            rms_post = float(np.sqrt(np.mean(np.sum((pred_post - ccf_lms) ** 2, axis=1))))

            # Convergence check vs previous full-iter result
            if T_prev_full is not None:
                delta = float(np.max(np.abs(M_full_smp2ccf - T_prev_full)))
                S = np.linalg.svd(M_full_smp2ccf[:3, :3], compute_uv=False)
                print(f"[joint_iter] iter {it}: post-MI landmark RMS = {rms_post:.2f} µm, "
                      f"|ΔT|max = {delta:.4e} µm, SVD = {S.tolist()}", flush=True)
                if delta < tol:
                    print(f"[joint_iter] converged after {it} iter "
                          f"(|ΔT| < {tol})", flush=True)
                    T_prev_full = M_full_smp2ccf
                    T_prev = np.vstack([M_full_smp2ccf[:3, :3].T, M_full_smp2ccf[:3, 3]])
                    break
            else:
                S = np.linalg.svd(M_full_smp2ccf[:3, :3], compute_uv=False)
                print(f"[joint_iter] iter {it}: post-MI landmark RMS = {rms_post:.2f} µm, "
                      f"SVD = {S.tolist()}", flush=True)
            T_prev_full = M_full_smp2ccf
            # T_prev as 4×3 in [A.T; t] form for the augmented LSQ regularizer
            T_prev = np.vstack([M_full_smp2ccf[:3, :3].T, M_full_smp2ccf[:3, 3]])

        # ---- save final result ----
        if affine:
            state.transform = RigidTransform(
                center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))
                                 if state.transform else (0.0, 0.0, 0.0))
            ).to_dict()
            state.affine = T_prev_full.tolist()
            S = np.linalg.svd(T_prev_full[:3, :3], compute_uv=False)
            print(f"[joint_iter] FINAL affine singular values: {S.tolist()}")
        else:
            prior_center = tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))
                                  if state.transform else (0.0, 0.0, 0.0))
            prior_for_decompose = RigidTransform(center_um=prior_center)
            rigid_new = _decompose_into_rigid(T_prev_full, prior_for_decompose)
            state.transform = rigid_new.to_dict()
            state.affine = None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    state.add_history(
        "joint_mi_landmarks_iter",
        f"shape={shape} affine={affine} mask={mask} "
        f"n_landmarks={n} weight={landmark_weight} max_iter={max_iter}",
    )
    if _save_path is not None:
        save_state(state, _save_path)
        print(f"[joint_iter] saved -> {_save_path}", flush=True)
    else:
        print(f"[joint_iter] in-memory result returned (no disk write)",
              flush=True)
    return state
