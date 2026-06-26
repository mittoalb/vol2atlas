"""Portable transform I/O.

Writes the registration transform out as standalone files (ITK
GenericAffine `.mat`, OME-NGFF coordinateTransformations `.json`, plus
landmarks `.csv`) so it can travel with the export and be reapplied to
other images / volumes / tools (ANTs, Slicer, BigWarp, custom scripts)
without needing the full vol2atlas state.

Convention reminder:
  - All vol2atlas matrices map SAMPLE physical µm → ATLAS physical µm.
  - vol2atlas's numpy axis order is (z, y, x).
  - ITK / ANTs files are written in the same numerical basis since
    ANTs' `from_numpy` reverses array axes internally (numpy axis 0
    becomes ITK axis 2); the matrix elements are identical when ITK
    reads them in row-major (ROW-MAJOR is what the underlying ITK
    binding expects for the 12 affine parameters — see comment in
    step_mi.py for the empirical justification).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


# =====================================================================
# ITK GenericAffine .mat — readable by ANTs antsApplyTransforms etc.
# =====================================================================
def write_itk_mat(M_smp2atlas: np.ndarray, path: Path) -> None:
    """Write a 4×4 sample→atlas matrix to ITK GenericAffine `.mat`
    format (the same format `antsRegistration` emits as
    `<prefix>0GenericAffine.mat`).

    Uses antspyx as the file-format codec. Falls back to a clear error
    if antspyx isn't installed — install via `pip install -e .[ants]`.
    """
    try:
        import ants
    except ImportError as e:
        raise RuntimeError(
            f"writing ITK .mat needs antspyx. Install: "
            f"pip install -e '.[ants]'  ({e})"
        )
    # ANTs convention for --initial-moving-transform is fixed→moving =
    # atlas→sample. Our matrix is sample→atlas. Invert so the saved
    # file matches the ANTs convention readers expect.
    M_atlas2smp = np.linalg.inv(M_smp2atlas)
    A = M_atlas2smp[:3, :3]
    t = M_atlas2smp[:3, 3]
    # ITK reads 12 params as ROW-MAJOR; see step_mi.py for the empirical
    # round-trip verification.
    params = np.concatenate([A.flatten(order='C'), t]).astype(np.float32)
    tx = ants.new_ants_transform(precision="float", dimension=3,
                                  transform_type="AffineTransform")
    tx.set_parameters(params)
    tx.set_fixed_parameters(np.zeros(3, dtype=np.float32))
    ants.write_transform(tx, str(path))


def read_itk_mat(path: Path) -> np.ndarray:
    """Read an ITK GenericAffine `.mat` and return the 4×4 SAMPLE→ATLAS
    matrix in our (z, y, x) µm basis.

    Inverts back from the on-disk ANTs (atlas→sample) convention.
    """
    try:
        import ants
    except ImportError as e:
        raise RuntimeError(
            f"reading ITK .mat needs antspyx. Install: "
            f"pip install -e '.[ants]'  ({e})"
        )
    # Use apply_transforms_to_points at 4 known points to recover the
    # matrix through ITK's own evaluator — same trick as
    # step_mi._transformlist_to_4x4. Avoids any parameter-order
    # ambiguity at the read step.
    import pandas as pd
    L = 100.0
    pts_in = pd.DataFrame({
        "x": [0.0, L,   0.0, 0.0],
        "y": [0.0, 0.0, L,   0.0],
        "z": [0.0, 0.0, 0.0, L  ],
    })
    pts_out = ants.apply_transforms_to_points(
        dim=3, points=pts_in, transformlist=[str(path)],
    )
    pts_out_arr = pts_out[["x", "y", "z"]].values
    pts_in_arr = pts_in[["x", "y", "z"]].values
    t = pts_out_arr[0]
    A = ((pts_out_arr[1:] - t) / L).T
    M_atlas2smp = np.eye(4)
    M_atlas2smp[:3, :3] = A
    M_atlas2smp[:3, 3] = t
    return np.linalg.inv(M_atlas2smp)


# =====================================================================
# OME-NGFF coordinateTransformations
# =====================================================================
def write_ngff_transform(
    M_smp2atlas: np.ndarray,
    atlas_voxel_um: tuple,
    path: Path,
    *,
    atlas_name: str = "",
    local_refinements: list | None = None,
) -> None:
    """Write the transform in OME-NGFF coordinateTransformations
    format. Output shape:

        {
          "type": "affine",
          "affine": [[..3×4..]],
          "input": {"axes": ["z","y","x"], "unit": "micrometer"},
          "output": {"axes": ["z","y","x"], "unit": "micrometer",
                     "voxel_um": [25.0, 25.0, 25.0],
                     "atlas": "allen_mouse_25um"}
        }

    Not the full OME-NGFF v0.4 spec embed (no parent multiscale linkage)
    — a standalone file suitable for `apply-transform` and
    BigWarp-adjacent tools.
    """
    payload = {
        "type": "affine",
        "affine": M_smp2atlas[:3, :4].tolist(),
        "input": {"axes": ["z", "y", "x"], "unit": "micrometer"},
        "output": {
            "axes": ["z", "y", "x"], "unit": "micrometer",
            "voxel_um": [float(v) for v in atlas_voxel_um],
            "atlas": atlas_name,
        },
        "local_refinements": list(local_refinements or []),
        "notes": (
            "4x4 homogeneous affine, sample µm → atlas µm. Bottom row "
            "(0,0,0,1) is implicit and omitted (3x4 only). To apply to "
            "a point (z,y,x) in sample µm: out = affine[:,:3] @ p + "
            "affine[:,3]. Compose with the OME-Zarr multiscale "
            "coordinateTransformations of the source dataset for end-"
            "to-end voxel→atlas voxel pipelines."
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def read_ngff_transform(path: Path) -> np.ndarray:
    """Read an OME-NGFF-style transform JSON written by
    write_ngff_transform and return the 4×4 sample→atlas matrix."""
    d = json.loads(Path(path).read_text())
    if d.get("type") != "affine":
        raise ValueError(f"unsupported transform type {d.get('type')!r}")
    a34 = np.asarray(d["affine"], dtype=float)
    if a34.shape != (3, 4):
        raise ValueError(f"expected 3×4 affine, got {a34.shape}")
    M = np.eye(4)
    M[:3, :4] = a34
    return M


# =====================================================================
# Landmarks CSV (compatible with BigWarp's CSV convention)
# =====================================================================
def write_landmarks_csv(
    sample_um_pts: list,
    ccf_um_pts: list,
    path: Path,
) -> None:
    """Write landmark pairs to CSV, one row per pair.

    Columns (BigWarp-compatible):
      "Pt-i,True,moving_x,moving_y,moving_z,fixed_x,fixed_y,fixed_z"
    where moving = sample, fixed = atlas (CCF). NOTE the column order
    is (x, y, z) — BigWarp's convention — while vol2atlas stores tuples
    as (z, y, x). We swap on write so the file is BigWarp-readable.
    """
    n = min(len(sample_um_pts), len(ccf_um_pts))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(n):
            sz, sy, sx = sample_um_pts[i]
            cz, cy, cx = ccf_um_pts[i]
            w.writerow([
                f"Pt-{i}", "True",
                f"{sx:.6f}", f"{sy:.6f}", f"{sz:.6f}",
                f"{cx:.6f}", f"{cy:.6f}", f"{cz:.6f}",
            ])


def read_landmarks_csv(path: Path) -> tuple:
    """Read a BigWarp-format landmarks CSV; return (sample_um_pts,
    ccf_um_pts) as lists of (z, y, x) tuples in vol2atlas convention.
    Rows where flag != 'True' are skipped."""
    sample_pts, ccf_pts = [], []
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            if row[1].strip() != "True":
                continue
            sx, sy, sz = float(row[2]), float(row[3]), float(row[4])
            cx, cy, cz = float(row[5]), float(row[6]), float(row[7])
            sample_pts.append((sz, sy, sx))   # back to (z, y, x)
            ccf_pts.append((cz, cy, cx))
    return sample_pts, ccf_pts
