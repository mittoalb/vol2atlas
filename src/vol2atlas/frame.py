"""Output-frame computation: contain the rotated sample without loss.

The user's `state.ccf_crop_bbox` is a REGION OF INTEREST in CCF voxel
coordinates, not a hard output limit. If the rigid transform rotates
the sample so its footprint extends beyond that bbox (e.g., a 45°
rotation pushes corners outside), naively using the bbox as the
output frame CLIPS the sample — silently losing voxels in every
output (interactive preview, export, alignFull).

This module computes the union of (user's bbox) ∪ (8-corner footprint
of the rotated sample) and returns that as the output frame. The
helper zero-pads the CCF reference if the union extends past CCF.

All shapes/origins/spacings are (z, y, x), µm where applicable.
"""
from __future__ import annotations

import numpy as np


def compute_output_frame(
    sample_shape: tuple,           # voxels (z, y, x)
    sample_voxel_um: tuple,        # µm per voxel (z, y, x)
    ccf_voxel_um: tuple,           # µm per voxel (z, y, x)
    rigid_matrix_um: np.ndarray,   # 4×4 sample_µm → ccf_µm
    ccf_crop_bbox: dict | None,    # {"z":[z0,z1], ...} in CCF voxels, or None
    pad_voxels: int = 2,
) -> tuple[np.ndarray, tuple]:
    """Compute (origin, shape) in CCF voxel coords that contains both
    the user's `ccf_crop_bbox` (if set) AND the 8 corners of the
    sample volume after applying `rigid_matrix_um`.

    `origin` may be negative or extend past `ccf.reference.shape` —
    use `extract_ccf` to zero-pad the CCF when slicing.
    """
    Lz = (sample_shape[0] - 1) * sample_voxel_um[0]
    Ly = (sample_shape[1] - 1) * sample_voxel_um[1]
    Lx = (sample_shape[2] - 1) * sample_voxel_um[2]
    corners_sample = np.array([
        [0,  0,  0 ],
        [Lz, 0,  0 ],
        [0,  Ly, 0 ],
        [0,  0,  Lx],
        [Lz, Ly, 0 ],
        [Lz, 0,  Lx],
        [0,  Ly, Lx],
        [Lz, Ly, Lx],
    ], dtype=float)

    M = np.asarray(rigid_matrix_um, dtype=float)
    corners_ccf_um  = (M[:3, :3] @ corners_sample.T).T + M[:3, 3]
    corners_ccf_vox = corners_ccf_um / np.asarray(ccf_voxel_um, dtype=float)

    lo = corners_ccf_vox.min(axis=0)
    hi = corners_ccf_vox.max(axis=0)

    if ccf_crop_bbox is not None:
        b = ccf_crop_bbox
        lo = np.minimum(lo, np.array([b["z"][0], b["y"][0], b["x"][0]], dtype=float))
        hi = np.maximum(hi, np.array([b["z"][1], b["y"][1], b["x"][1]], dtype=float))

    origin = np.floor(lo - pad_voxels).astype(np.int64)
    shape  = tuple(int(s) for s in np.ceil(hi + pad_voxels - origin).astype(np.int64))
    return origin, shape


def enable_ccf_axes(viewer, orientation: str = "asr") -> None:
    """Turn on napari's axis overlay and label each axis with the
    anatomical direction implied by the CCF `orientation` string
    (e.g. "asr" → Anterior, Superior, Right).

    Each character of `orientation` is the direction of POSITIVE motion
    along that axis (numpy axes 0, 1, 2 = our z, y, x):
      A=Anterior  P=Posterior  S=Superior  I=Inferior  L=Left  R=Right
    """
    DIR_NAMES = {"A": "Anterior",  "P": "Posterior",
                 "S": "Superior",  "I": "Inferior",
                 "L": "Left",      "R": "Right"}
    DIR_OPP   = {"A": "P", "P": "A", "S": "I", "I": "S", "L": "R", "R": "L"}
    code = orientation.upper()
    if len(code) != 3 or any(c not in DIR_NAMES for c in code):
        labels = ("z", "y", "x")
    else:
        labels = tuple(
            f"{ax} ({DIR_OPP[c]}→{c})"
            for ax, c in zip(("z", "y", "x"), code)
        )
    try:
        viewer.axes.visible = True
        viewer.axes.labels  = True
        viewer.axes.colored = True
        viewer.dims.axis_labels = labels
    except Exception:
        pass


def extract_ccf(
    ccf_full: np.ndarray,
    origin: np.ndarray,    # CCF voxel coords, possibly negative
    shape: tuple,          # CCF voxels
) -> np.ndarray:
    """Slice `ccf_full` over [origin, origin+shape], zero-padding any
    region that falls outside the CCF (negative origin or beyond
    ccf_full.shape)."""
    out = np.zeros(shape, dtype=ccf_full.dtype)
    src_lo = np.maximum(origin, 0)
    src_hi = np.minimum(origin + np.asarray(shape), np.asarray(ccf_full.shape))
    if np.any(src_hi <= src_lo):
        return out
    dst_lo = src_lo - origin
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        ccf_full[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out
