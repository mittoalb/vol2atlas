"""Lazy resampling helpers used by `prealign_for_brainreg.py`.

The original elastix-based refine + inspect pipeline is gone; brainreg replaces
it. Only the resampling primitive is kept, since the prealignment script needs
it to write a CCF-grid NIfTI for brainreg to consume.
"""
from __future__ import annotations

import dask.array as da
import numpy as np

from .atlas import CCFReference
from .io import MultiscaleVolume
from .transform import RigidTransform


def _resample_to_ccf_grid(
    ms: MultiscaleVolume,
    t: RigidTransform,
    ccf: CCFReference,
    level: int,
    channel: int | None,
    oversample: float = 2.0,
) -> np.ndarray:
    """Pre-warp the chosen pyramid level into the CCF voxel grid (in RAM).

    If the chosen level is much finer than the CCF target, decimates further
    with dask block-mean (anti-aliased) so RAM stays small. `oversample` keeps
    the working resolution finer than the CCF by that factor (default 2x).
    """
    from scipy.ndimage import affine_transform

    arr = ms.level(level)
    if "c" in ms.axes and channel is not None:
        c_axis = ms.axes.index("c")
        arr = da.take(arr, channel, axis=c_axis)
    sample_um = ms.spacing(level)

    # Decimate with dask if level is finer than oversample x target spacing.
    target = tuple(v / float(oversample) for v in ccf.voxel_um)
    factors = tuple(max(1, int(round(t_um / s_um)))
                    for t_um, s_um in zip(target, sample_um))
    if any(f > 1 for f in factors):
        new_um = tuple(s * f for s, f in zip(sample_um, factors))
        new_shape = tuple(arr.shape[-3 + i] // factors[i] for i in range(3))
        n_bytes = int(np.prod(new_shape)) * arr.dtype.itemsize
        print(f"[zrot] level {level} is {arr.shape} @ {sample_um} µm; "
              f"decimating x{factors} -> {new_shape} @ {new_um} µm "
              f"(~{n_bytes / 1e9:.2f} GB) before RAM load")
        crop = tuple(slice(0, (arr.shape[-3 + i] // factors[i]) * factors[i])
                     for i in range(3))
        if arr.ndim == 4:
            crop = (slice(None),) + crop
        arr = arr[crop]
        coarsen_axes = {arr.ndim - 3: factors[0],
                        arr.ndim - 2: factors[1],
                        arr.ndim - 1: factors[2]}
        arr = da.coarsen(np.mean, arr, coarsen_axes, trim_excess=True).astype(arr.dtype)
        sample_um = new_um

    print(f"[zrot] loading {arr.shape} {arr.dtype} into RAM "
          f"(~{int(np.prod(arr.shape) * arr.dtype.itemsize) / 1e9:.2f} GB)...")
    arr_np = np.ascontiguousarray(arr.compute())

    M = t.for_voxel_grid(sample_um, ccf.voxel_um)
    Minv = np.linalg.inv(M)
    out = affine_transform(
        arr_np, Minv[:3, :3], offset=Minv[:3, 3],
        output_shape=ccf.reference.shape,
        order=1, mode="constant", cval=0,
    )
    return out
