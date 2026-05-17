"""Apply the saved rigid transform to the FULL OME-Zarr chunkwise.

For each requested input pyramid level, lazily reads the sample's level k,
applies the inverse rigid in voxel coords via dask_image, writes the
warped result as level k of a NEW multiscale OME-Zarr in CCF coordinates.
RAM stays bounded by the dask chunk size (~64³ in / out per worker).

Currently rigid-only (the transform stored in state.json). Composition
with ANTs/elastix deformable fields is straightforward to add as a
second pass once the rigid pipeline is verified — see TODO at bottom.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from ome_zarr.io import parse_url

from ..atlas import load_ccf
from ..io import open_multiscale
from ..state import load as load_state, save as save_state
from ..transform import RigidTransform


def run(
    state_path: Path,
    out_zarr: Path,
    levels: Optional[list[int]] = None,
    output_chunks: Optional[tuple[int, int, int]] = None,
) -> None:
    try:
        from dask_image.ndinterp import affine_transform as dask_affine
    except ImportError:
        sys.exit("[alignFull] dask-image not installed. "
                 "Run: pip install dask-image")

    state = load_state(Path(state_path))
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run `vol2atlas prealign` first.")

    ms  = open_multiscale(state.sample_zarr)
    ccf = load_ccf(state.atlas_name)

    # ---- crop in atlas voxels + µm (full atlas coords) ------------------
    atlas_voxel_um = np.asarray(ccf.voxel_um, dtype=float)
    b = state.ccf_crop_bbox
    if b is None:
        crop_origin_vox = np.zeros(3)
        crop_size_vox   = np.asarray(ccf.reference.shape, dtype=float)
    else:
        crop_origin_vox = np.asarray([b["z"][0], b["y"][0], b["x"][0]], dtype=float)
        crop_size_vox   = np.asarray([b["z"][1] - b["z"][0],
                                       b["y"][1] - b["y"][0],
                                       b["x"][1] - b["x"][0]], dtype=float)
    crop_origin_um = crop_origin_vox * atlas_voxel_um
    crop_size_um   = crop_size_vox   * atlas_voxel_um
    print(f"[alignFull] CCF crop  origin={tuple(crop_origin_um)} µm  "
          f"size={tuple(crop_size_um)} µm")

    # ---- rigid (sample µm → CCF µm); we need the inverse for pullback ---
    rigid = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        flip_z=bool(state.transform.get("flip_z", False)),
        flip_y=bool(state.transform.get("flip_y", False)),
        flip_x=bool(state.transform.get("flip_x", False)),
        center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))),
    )
    M_um    = rigid.matrix()
    Minv_um = np.linalg.inv(M_um)

    # ---- levels to process ----------------------------------------------
    avail = list(range(ms.n_levels()))
    target_levels = levels if levels else avail
    bad = [l for l in target_levels if l not in avail]
    if bad:
        raise ValueError(f"Requested levels {bad} not in available {avail}")

    # ---- output OME-Zarr group ------------------------------------------
    out_zarr = Path(out_zarr)
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.mkdir(parents=True, exist_ok=True)
    store = parse_url(str(out_zarr), mode="w").store
    root  = zarr.group(store=store)

    # Per-level voxel size: if state.json has an override at sample_level
    # (because the OME-Zarr's stored spacing was wrong), derive every level's
    # spacing from the level-to-level RATIO of ms.spacing — that ratio is
    # reliable even when the absolute is wrong.
    def voxel_um_at(level: int) -> np.ndarray:
        sp_lev = np.asarray(ms.spacing(level),               dtype=float)
        if state.sample_voxel_um is None:
            return sp_lev
        sp_anc = np.asarray(ms.spacing(state.sample_level),  dtype=float)
        anc_um = np.asarray(state.sample_voxel_um,           dtype=float)
        return anc_um * (sp_lev / sp_anc)

    scales: list[list[float]] = []
    for out_idx, level in enumerate(target_levels):
        sample_um = voxel_um_at(level)
        # Output level shape is the cropped CCF physical size, divided by
        # THIS pyramid level's voxel size (mirroring the input's scale).
        out_shape = tuple(int(np.ceil(crop_size_um[a] / sample_um[a]))
                          for a in range(3))

        # Voxel-grid pullback matrix for affine_transform:
        #   sample_voxel = D⁻¹ · Minv_um[:3,:3] · D · out_voxel
        #                + D⁻¹ · (Minv_um[:3,:3] · crop_origin_um + Minv_um[:3,3])
        # For isotropic voxels D commutes with the rotation and simplifies,
        # but the matrix form here works for anisotropic too.
        D     = np.diag(sample_um)
        D_inv = np.diag(1.0 / sample_um)
        M_v   = D_inv @ Minv_um[:3, :3] @ D
        off_v = D_inv @ (Minv_um[:3, :3] @ crop_origin_um + Minv_um[:3, 3])

        arr = ms.level(level)
        if "c" in ms.axes:
            arr = arr[state.sample_channel]
        # If user didn't pass --chunks, mirror the input zarr's chunks for
        # this level. Aligning output chunks with input chunks eliminates
        # the read amplification that kills throughput with mismatched
        # chunk sizes (output chunks 64³ pulling input chunks 256³ ≈ 64x
        # waste). Falls back to 128³ if the input is unchunked.
        chunks_for_level = output_chunks
        if chunks_for_level is None:
            try:
                in_chunks = arr.chunksize          # 3-tuple from dask
                chunks_for_level = tuple(int(c) for c in in_chunks[-3:])
            except Exception:
                chunks_for_level = (128, 128, 128)
        n_vox        = int(np.prod(out_shape))
        size_raw_gb  = n_vox * np.dtype(arr.dtype).itemsize / 1e9
        # Blosc compression on µCT brain volumes typically gives 2-3x,
        # so on-disk size is usually ~30-50% of raw.
        print(f"[alignFull] level {level} → out '{out_idx}'  "
              f"shape={out_shape}  voxel={tuple(sample_um)} µm  "
              f"chunks={chunks_for_level}  ~{size_raw_gb:.2f} GB raw "
              f"(~{size_raw_gb*0.4:.2f} GB compressed expected)",
              flush=True)

        warped = dask_affine(
            arr,
            matrix=M_v,
            offset=off_v,
            output_shape=out_shape,
            order=1,
            mode="constant",
            cval=0,
            output_chunks=chunks_for_level,
        ).astype(arr.dtype)

        # Native dask `.to_zarr()` so the scheduler can fuse tasks across
        # chunks and reuse decompressed input — orders of magnitude faster
        # than per-chunk `.compute()` loops on a rotated transform.
        # Progress shown via dask's ProgressBar (task-level, not chunk-level
        # but moves continuously through the whole graph).
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            warped.to_zarr(str(out_zarr / str(out_idx)),
                            overwrite=True, compute=True)
        scales.append([float(sample_um[0]),
                       float(sample_um[1]),
                       float(sample_um[2])])
        print(f"[alignFull] wrote {out_zarr.name}/{out_idx}")

    # ---- OME-Zarr v0.4 multiscales metadata -----------------------------
    root.attrs["multiscales"] = [{
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {"path": str(i),
             "coordinateTransformations": [
                {"type": "scale", "scale": s}
             ]}
            for i, s in enumerate(scales)
        ],
        "version": "0.4",
    }]

    state.add_history("alignFull",
                       f"levels={target_levels} chunks={output_chunks}")
    save_state(state, Path(state_path))
    print(f"[alignFull] done. multiscale OME-Zarr at {out_zarr}")


def _write_chunks_tqdm(
    warped,
    zarr_path: Path,
    output_chunks: tuple[int, int, int],
    desc: str = "chunks",
    max_workers: int | None = None,
) -> None:
    """Allocate a zarr array of `warped.shape`/`warped.dtype` at zarr_path,
    then compute and write each dask chunk in parallel, with a per-chunk
    tqdm progress bar showing count / rate / ETA.

    Parallelism is bounded by a ThreadPoolExecutor so RAM usage is
    `max_workers × (chunk + a few input chunks)`. CPU-bound work inside
    each chunk (scipy.ndimage.map_coordinates) still releases the GIL
    where possible.
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    import zarr
    from tqdm import tqdm

    if max_workers is None:
        max_workers = int(os.environ.get("DASK_NUM_WORKERS",
                                          os.cpu_count() or 4))

    z = zarr.open(
        str(zarr_path), mode="w",
        shape=warped.shape, chunks=output_chunks,
        dtype=warped.dtype,
    )

    # Enumerate every dask block; each is a (z_block, y_block, x_block) tuple.
    block_indices = list(itertools.product(*[range(n) for n in warped.numblocks]))

    def _one(idx):
        block = warped.blocks[idx].compute()
        # Compute the destination slice (last block may be smaller than chunk).
        starts = tuple(b * c for b, c in zip(idx, output_chunks))
        slices = tuple(slice(s, s + bs) for s, bs in zip(starts, block.shape))
        z[slices] = block
        return idx

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, idx) for idx in block_indices]
        for fut in tqdm(as_completed(futures),
                         total=len(block_indices),
                         desc=desc, unit="chk"):
            fut.result()   # propagate any chunk-level exception


# TODO: --transform <dir> for ANTs/elastix deformable composition.
# Two reasonable implementations once needed:
#   - ANTs:   per-chunk wrap of ants.apply_transforms with the saved
#             ants_transforms/ files, composed with the rigid above.
#   - elastix: per-chunk transformix call with the saved
#              elastix_transforms/TransformParameters.*.txt files.
# Both are chunk-safe natively; the wrapper just needs to walk
# dask blocks instead of relying on dask_image.ndinterp.
