"""OME-Zarr multiscale I/O. Lazy by design — never reads pixel data eagerly."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import zarr


@dataclass
class MultiscaleVolume:
    """A pyramid of dask arrays plus per-level voxel spacing in micrometers.

    Axes are ordered (z, y, x). For 4D OME-Zarr (c, z, y, x) the channel axis
    is preserved as axis 0 of each level array; voxel spacing still describes
    the spatial axes only.
    """
    levels: list[da.Array]
    voxel_um: list[tuple[float, float, float]]  # per-level (z, y, x) µm
    axes: tuple[str, ...]                        # e.g. ("z", "y", "x") or ("c", "z", "y", "x")
    store_path: str
    group_path: str                              # path inside the zarr (often "")

    def level(self, i: int) -> da.Array:
        return self.levels[i]

    def spacing(self, i: int) -> tuple[float, float, float]:
        return self.voxel_um[i]

    def n_levels(self) -> int:
        return len(self.levels)

    def spatial_shape(self, i: int) -> tuple[int, int, int]:
        arr = self.levels[i]
        # last 3 axes are spatial
        return tuple(arr.shape[-3:])  # type: ignore[return-value]

    def summary(self) -> str:
        """Human-readable pyramid table."""
        lines = [f"OME-Zarr: {self.store_path}",
                 f"  axes: {self.axes}",
                 f"  levels: {self.n_levels()}",
                 "  idx | shape                | voxel (z,y,x) µm        | est. size"]
        for i, (arr, vx) in enumerate(zip(self.levels, self.voxel_um)):
            n_bytes = int(np.prod(arr.shape)) * arr.dtype.itemsize
            lines.append(
                f"  {i:3d} | {str(tuple(arr.shape)):20s} | "
                f"({vx[0]:7.2f},{vx[1]:7.2f},{vx[2]:7.2f}) | {_human_bytes(n_bytes)}"
            )
        return "\n".join(lines)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:6.1f} EB"


def open_multiscale(path: str | Path) -> MultiscaleVolume:
    """Open an OME-Zarr group and return a MultiscaleVolume.

    Reads only the multiscales metadata + array headers. No voxels are loaded.
    """
    path = str(path)
    root = zarr.open(path, mode="r")
    if not isinstance(root, zarr.hierarchy.Group):
        raise ValueError(f"{path} is not a zarr group")

    attrs = dict(root.attrs)
    if "multiscales" not in attrs:
        raise ValueError(
            f"{path} has no 'multiscales' metadata; not an OME-Zarr group"
        )
    ms = attrs["multiscales"][0]
    axes = tuple(a["name"] for a in ms["axes"])
    axis_units = {a["name"]: a.get("unit", "") for a in ms["axes"]}

    levels: list[da.Array] = []
    spacings: list[tuple[float, float, float]] = []
    for ds in ms["datasets"]:
        arr_path = ds["path"]
        z = root[arr_path]
        # Use dask to wrap the zarr array — chunked, lazy, parallelizable
        d = da.from_zarr(z)
        levels.append(d)

        # Compose coordinateTransformations (scale only, for OME-Zarr 0.4)
        scale = np.ones(len(axes), dtype=float)
        for ct in ds.get("coordinateTransformations", []):
            if ct["type"] == "scale":
                scale *= np.asarray(ct["scale"], dtype=float)
        # Apply group-level transformations too
        for ct in ms.get("coordinateTransformations", []):
            if ct["type"] == "scale":
                scale *= np.asarray(ct["scale"], dtype=float)

        # Convert each spatial axis to micrometers
        spatial = []
        for ax_name, s in zip(axes, scale):
            if ax_name in ("z", "y", "x"):
                spatial.append(_to_um(s, axis_units.get(ax_name, "")))
        if len(spatial) != 3:
            raise ValueError(f"Expected 3 spatial axes, got {spatial} from {axes}")
        spacings.append(tuple(spatial))  # type: ignore[arg-type]

    return MultiscaleVolume(
        levels=levels,
        voxel_um=spacings,
        axes=axes,
        store_path=path,
        group_path="",
    )


def _to_um(value: float, unit: str) -> float:
    """Normalize a coordinate scale value to micrometers."""
    u = unit.lower()
    factor = {
        "": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "micron": 1.0,
        "um": 1.0,
        "µm": 1.0,
        "millimeter": 1e3,
        "millimeters": 1e3,
        "mm": 1e3,
        "nanometer": 1e-3,
        "nanometers": 1e-3,
        "nm": 1e-3,
        "meter": 1e6,
        "m": 1e6,
    }.get(u, 1.0)
    return float(value) * factor
