"""Local refinement: spherical-mask + sigmoid-falloff blend between
a global affine and one or more local affines.

Use case: after a global rigid+affine registration to the CCF, the
user notices one region (e.g., a single hemisphere lobe) is slightly
off. They want to apply EXTRA local landmarks ONLY in that region and
have those landmarks ONLY locally bend the transform, without
disturbing the rest of the volume.

Implementation:

  - Each local refinement defines a sphere in SAMPLE physical µm
    (center_um, radius_um) and a sigmoid falloff width (falloff_um).
  - Inside the sphere (distance to center < radius - falloff/2): mask
    weight w → 1, full local affine applies.
  - Outside the sphere + falloff (distance > radius + falloff/2):
    w → 0, only the global transform applies.
  - In the falloff zone: w smoothly interpolates via sigmoid.

  - For each ATLAS voxel Q (in CCF µm), compute:
       global_source = M_global⁻¹ @ Q                 (sample µm)
       w             = weight(global_source)
       local_source  = M_local⁻¹  @ Q                 (sample µm)
       blended       = (1 - w) * global_source + w * local_source
    Then sample-grid lookup at `blended / sample_voxel_um`.

This vectorizes cleanly: build a coordinate grid for the output chunk,
apply both matrices, blend, and pass to scipy.ndimage.map_coordinates.

Multiple refinements: weights stack additively but capped at 1, with
each local's contribution proportionally trimmed if the cap is hit
(handled in `_blended_sources`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class LocalRefinement:
    name: str
    center_um: tuple          # in SAMPLE µm
    radius_um: float
    falloff_um: float
    affine: np.ndarray        # 4x4, sample µm → CCF µm
    landmark_indices: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "LocalRefinement":
        return cls(
            name=str(d["name"]),
            center_um=tuple(float(v) for v in d["center_um"]),
            radius_um=float(d["radius_um"]),
            falloff_um=float(d["falloff_um"]),
            affine=np.asarray(d["affine_4x4"], dtype=float),
            landmark_indices=list(d.get("landmark_indices", [])),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "center_um": list(float(v) for v in self.center_um),
            "radius_um": float(self.radius_um),
            "falloff_um": float(self.falloff_um),
            "affine_4x4": self.affine.tolist(),
            "landmark_indices": list(self.landmark_indices),
        }


def fit_from_landmarks(
    sample_lms_subset: np.ndarray,    # (n, 3) sample µm
    ccf_lms_subset: np.ndarray,       # (n, 3) CCF µm
    *,
    name: str,
    falloff_um: float = 300.0,
    radius_pad_um: float = 200.0,
    landmark_indices: list | None = None,
) -> LocalRefinement:
    """Fit a local affine from a SUBSET of landmark pairs and derive
    the spherical mask from their centroid + max distance.

    Center = mean of sample-side landmarks.
    Radius = max(distance to center) + radius_pad_um (so all the
    landmarks are well inside the sphere, with margin).
    """
    n = sample_lms_subset.shape[0]
    if n < 4:
        raise ValueError(
            f"local refinement needs ≥4 landmark pairs (have {n})"
        )
    # Closed-form affine LSQ: M @ [sample; 1] = ccf
    H = np.hstack([sample_lms_subset, np.ones((n, 1))])
    A_ls, *_ = np.linalg.lstsq(H, ccf_lms_subset, rcond=None)
    M = np.eye(4)
    M[:3, :3] = A_ls[:3, :].T
    M[:3, 3] = A_ls[3, :]
    center = sample_lms_subset.mean(axis=0)
    dists = np.linalg.norm(sample_lms_subset - center, axis=1)
    radius = float(dists.max()) + float(radius_pad_um)
    return LocalRefinement(
        name=name,
        center_um=tuple(float(v) for v in center),
        radius_um=radius,
        falloff_um=float(falloff_um),
        affine=M,
        landmark_indices=list(landmark_indices or []),
    )


def _sphere_weight(
    sample_pts: np.ndarray,           # (..., 3) sample µm
    center: np.ndarray,                # (3,) sample µm
    radius: float,
    falloff: float,
) -> np.ndarray:
    """Sigmoid weight: ~1 inside the sphere, ~0 outside, smoothly
    interpolated in the falloff band straddling the boundary.

    Specifically: w = sigmoid((radius - dist) / (falloff/4)). At
    dist = radius this gives w = 0.5; at dist = radius - falloff
    it's near 1; at dist = radius + falloff it's near 0.
    """
    d = np.linalg.norm(sample_pts - center, axis=-1)
    # Avoid division by zero; clip to a tiny floor.
    fall = max(float(falloff), 1e-3)
    x = (radius - d) / (fall / 4.0)
    # Stable sigmoid
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def blended_inverse_sample_coords(
    out_voxel_coords: np.ndarray,     # (3, ...) output voxel indices (within crop)
    M_global_um: np.ndarray,          # 4x4 sample µm → atlas µm (global)
    sample_voxel_um: tuple,            # (z, y, x) µm per sample voxel
    out_voxel_um: tuple,               # (z, y, x) µm per OUTPUT voxel
    crop_origin_um: tuple,             # (z, y, x) crop origin in atlas µm
    local_refinements: Iterable[LocalRefinement] = (),
) -> np.ndarray:
    """For a chunk of OUTPUT voxel coordinates, return the corresponding
    SAMPLE voxel coordinates after applying the global transform PLUS
    any blended local refinements.

    Input shape conventions:
      out_voxel_coords: shape (3, ...) — first axis is (z, y, x).
      The output voxel size may differ from the atlas voxel size
      (e.g., alignFull writes at sample_um while atlas is at 25 µm).
      crop_origin_um is given in atlas µm (where the crop starts in
      atlas space).

    Returns shape (3, ...) — sample voxel coordinates suitable for
    `scipy.ndimage.map_coordinates(sample, returned, ...)`.

    The math: for each output voxel Q (output voxel index), compute
    atlas_um = Q * out_voxel_um + crop_origin_um. Then:
      global_source_um = M_global_inv @ atlas_um
      For each local refinement L:
        w = _sphere_weight(global_source_um, L.center, L.radius, L.falloff)
        local_source_um = L.affine_inv @ atlas_um
        global_source_um += w * (local_source_um - global_source_um)
    Finally divide by sample_voxel_um to get sample voxel coords.
    """
    out_voxel_um_arr = np.asarray(out_voxel_um, dtype=float).reshape(3, *([1] * (out_voxel_coords.ndim - 1)))
    crop_origin_um_arr = np.asarray(crop_origin_um, dtype=float).reshape(3, *([1] * (out_voxel_coords.ndim - 1)))
    sample_voxel_um_arr = np.asarray(sample_voxel_um, dtype=float).reshape(3, *([1] * (out_voxel_coords.ndim - 1)))

    # output voxel → atlas µm  (includes crop origin offset)
    ccf_um = out_voxel_coords * out_voxel_um_arr + crop_origin_um_arr   # (3, ...)

    M_global_inv = np.linalg.inv(M_global_um)
    A_g = M_global_inv[:3, :3]
    t_g = M_global_inv[:3, 3]

    # global_source = A_g @ ccf_um + t_g
    # tensordot to handle arbitrary trailing shape
    global_source = np.tensordot(A_g, ccf_um, axes=(1, 0)) + t_g.reshape(3, *([1] * (out_voxel_coords.ndim - 1)))

    if local_refinements:
        # Vectorized over each refinement. We move the (3, ...)→(...,3)
        # axis once for weight computation.
        global_source_pts = np.moveaxis(global_source, 0, -1)   # (..., 3)
        for L in local_refinements:
            w = _sphere_weight(
                global_source_pts,
                np.asarray(L.center_um, dtype=float),
                L.radius_um,
                L.falloff_um,
            )                                                    # (...,)
            M_local_inv = np.linalg.inv(L.affine)
            A_l = M_local_inv[:3, :3]
            t_l = M_local_inv[:3, 3]
            local_source = np.tensordot(A_l, ccf_um, axes=(1, 0)) + \
                            t_l.reshape(3, *([1] * (out_voxel_coords.ndim - 1)))
            # blended = (1-w) * global + w * local
            global_source = global_source + w[None, ...] * (local_source - global_source)
            # Update the (...,3) view so the next refinement uses the
            # already-blended position for its own weight.
            global_source_pts = np.moveaxis(global_source, 0, -1)

    # sample µm → sample voxel
    return global_source / sample_voxel_um_arr
