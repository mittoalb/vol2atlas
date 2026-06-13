"""Anatomical orientation helpers.

A 3-letter orientation code (BrainGlobe / ITK convention) describes
which anatomical direction each numpy axis INCREASES TOWARD:

  letter  direction        unit vector in RAS (Right, Anterior, Superior)
  ------  ---------------- ---------------------
    R     +Right           (+1,  0,  0)
    L     -Right (Left)    (-1,  0,  0)
    A     +Anterior        ( 0, +1,  0)
    P     -Anterior (Post) ( 0, -1,  0)
    S     +Superior        ( 0,  0, +1)
    I     -Superior (Inf)  ( 0,  0, -1)

So "asr" means: numpy axis 0 increases toward Anterior, axis 1 toward
Superior, axis 2 toward Right. Allen mouse CCF is "asr" by default.

For two orientations to map onto each other with a PROPER rotation
(det = +1), both must be right-handed coordinate systems. Otherwise
the mapping requires a reflection, which we do not support.
"""
from __future__ import annotations

import numpy as np


_LETTER_TO_VEC = {
    "R": np.array([+1.0, 0.0, 0.0]),
    "L": np.array([-1.0, 0.0, 0.0]),
    "A": np.array([0.0, +1.0, 0.0]),
    "P": np.array([0.0, -1.0, 0.0]),
    "S": np.array([0.0, 0.0, +1.0]),
    "I": np.array([0.0, 0.0, -1.0]),
}

_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


def parse_orientation(code: str) -> np.ndarray:
    """Return a 3×3 direction matrix M where row i is the anatomical
    unit vector that numpy axis i points along, in RAS basis.

    Raises ValueError if the code is malformed (not 3 letters, repeats
    an axis, or uses an invalid letter).
    """
    c = code.upper().strip()
    if len(c) != 3:
        raise ValueError(f"orientation must be 3 letters; got {code!r}")
    seen_axes: set[str] = set()
    rows = []
    for ch in c:
        if ch not in _LETTER_TO_VEC:
            raise ValueError(
                f"invalid letter {ch!r} in {code!r}; "
                f"choose from R, L, A, P, S, I"
            )
        # Canonicalize to the +direction letter to detect duplicate axes
        axis_letter = ch if ch in "RAS" else _OPPOSITE[ch]
        if axis_letter in seen_axes:
            raise ValueError(
                f"orientation {code!r} uses the {axis_letter}/{_OPPOSITE[axis_letter]} "
                f"axis twice (each anatomical axis must appear exactly once)"
            )
        seen_axes.add(axis_letter)
        rows.append(_LETTER_TO_VEC[ch])
    return np.vstack(rows)


def rotation_between(sample_orient: str, atlas_orient: str) -> np.ndarray:
    """3×3 rotation R that maps a vector expressed in SAMPLE numpy-axes
    to the same physical vector expressed in ATLAS numpy-axes.

    Derivation: if D is the sample direction matrix (rows = anatomical
    unit vectors for sample numpy axes 0/1/2) and E is the atlas
    direction matrix, then for a vector v in sample µm space the same
    physical vector in atlas µm space is w = E @ D.T @ v. Both D and E
    are orthonormal (their rows are unit vectors along ±cardinal
    directions), so E.T^-1 == E.

    Raises ValueError if R has negative determinant (the two orientations
    differ by a reflection / handedness change — that requires axis
    flips, which we do not currently support in state.transform).
    """
    D = parse_orientation(sample_orient)
    E = parse_orientation(atlas_orient)
    R = E @ D.T
    det = float(np.linalg.det(R))
    if not np.isclose(abs(det), 1.0, atol=1e-6):
        raise ValueError(
            f"rotation matrix from {sample_orient!r} to {atlas_orient!r} "
            f"is not orthonormal (det={det}). This is a bug — please "
            f"report."
        )
    if det < 0:
        raise ValueError(
            f"orientation {sample_orient!r} → {atlas_orient!r} requires "
            f"an axis FLIP (det(R) = {det:.0f}, i.e. the two orientations "
            f"are different chirality). vol2atlas does not support flips. "
            f"Either re-label your sample with a right-handed orientation, "
            f"or pre-flip the sample data along one axis before init."
        )
    return R


def euler_zyx_degrees_from_matrix(R: np.ndarray) -> tuple:
    """Decompose a 3×3 proper rotation into intrinsic ZYX Euler angles
    (degrees), matching the convention used by RigidTransform."""
    from scipy.spatial.transform import Rotation
    rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return float(rz), float(ry), float(rx)
