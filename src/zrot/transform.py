"""Rigid transform stored in physical (micrometer) space.

Convention: T maps SAMPLE physical coordinates -> CCF physical coordinates.
Coords are (z, y, x) in µm.

Because the transform lives in physical space, it is independent of which
pyramid level you align on — the same matrix applies to every resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RigidTransform:
    # Euler angles (degrees) about z, y, x intrinsic axes — rotation order "ZYX"
    rz_deg: float = 0.0
    ry_deg: float = 0.0
    rx_deg: float = 0.0
    # Translation in micrometers, applied AFTER rotation around `center_um`
    tz_um: float = 0.0
    ty_um: float = 0.0
    tx_um: float = 0.0
    # Axis flips (mirror about `center_um`) — applied BEFORE rotation.
    flip_z: bool = False
    flip_y: bool = False
    flip_x: bool = False
    # Center of rotation/flip (µm), in SAMPLE physical space — typically volume center
    center_um: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> np.ndarray:
        """Return a 4x4 homogeneous matrix mapping sample µm -> CCF µm.

        Order:  T_translate * T_to_origin_inv * R * F * T_to_origin
        i.e. flip about `center_um`, then rotate, then translate.
        """
        c = np.asarray(self.center_um, dtype=float)
        R = Rotation.from_euler(
            "ZYX",
            [self.rz_deg, self.ry_deg, self.rx_deg],
            degrees=True,
        ).as_matrix()
        F = np.diag([
            -1.0 if self.flip_z else 1.0,
            -1.0 if self.flip_y else 1.0,
            -1.0 if self.flip_x else 1.0,
        ])
        RF = R @ F
        M = np.eye(4)
        M[:3, :3] = RF
        M[:3, 3] = -RF @ c + c + np.array([self.tz_um, self.ty_um, self.tx_um])
        return M

    def inverse_matrix(self) -> np.ndarray:
        M = self.matrix()
        Minv = np.eye(4)
        Rt = M[:3, :3].T
        Minv[:3, :3] = Rt
        Minv[:3, 3] = -Rt @ M[:3, 3]
        return Minv

    def for_voxel_grid(
        self,
        sample_voxel_um: tuple[float, float, float],
        ccf_voxel_um: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """4x4 matrix mapping SAMPLE voxel indices -> CCF voxel indices (or µm).

        If `ccf_voxel_um` is None, output is in micrometers.
        Use this directly with napari `Layer.affine` (in voxel units) by
        passing both spacings.
        """
        S_in = np.diag([*sample_voxel_um, 1.0])             # voxel -> µm
        if ccf_voxel_um is None:
            return self.matrix() @ S_in
        S_out_inv = np.diag([1.0 / v for v in ccf_voxel_um] + [1.0])  # µm -> voxel
        return S_out_inv @ self.matrix() @ S_in

    def to_dict(self) -> dict:
        return {
            "kind": "rigid",
            "convention": "sample_um_to_ccf_um",
            "axes": ["z", "y", "x"],
            "rotation_order": "ZYX_intrinsic_degrees",
            "rz_deg": self.rz_deg,
            "ry_deg": self.ry_deg,
            "rx_deg": self.rx_deg,
            "tz_um": self.tz_um,
            "ty_um": self.ty_um,
            "tx_um": self.tx_um,
            "flip_z": bool(self.flip_z),
            "flip_y": bool(self.flip_y),
            "flip_x": bool(self.flip_x),
            "center_um": list(self.center_um),
            "matrix_4x4": self.matrix().tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigidTransform":
        return cls(
            rz_deg=d["rz_deg"], ry_deg=d["ry_deg"], rx_deg=d["rx_deg"],
            tz_um=d["tz_um"], ty_um=d["ty_um"], tx_um=d["tx_um"],
            flip_z=bool(d.get("flip_z", False)),
            flip_y=bool(d.get("flip_y", False)),
            flip_x=bool(d.get("flip_x", False)),
            center_um=tuple(d["center_um"]),
        )


def save_transform(t: RigidTransform, path: str | Path) -> None:
    Path(path).write_text(json.dumps(t.to_dict(), indent=2))


def load_transform(path: str | Path) -> RigidTransform:
    return RigidTransform.from_dict(json.loads(Path(path).read_text()))
