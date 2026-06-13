"""Apply the saved rigid transform to the FULL OME-Zarr chunkwise.

For each requested input pyramid level, iterates the OUTPUT chunks one
at a time. For each output chunk it computes the corresponding source
bounding box via the inverse rigid, reads ONLY that bounding box
directly from the zarr array (no dask graph), applies
scipy.ndimage.affine_transform locally, writes the output chunk to the
result zarr. RAM stays bounded by `chunk × ~rotation-overhead` per
worker.

Why manual instead of dask_image.ndinterp:
For TB-scale inputs chunked at 64³, the dask task graph from
dask_image.ndinterp.affine_transform has millions of nodes, and the
upfront graph-optimization phase takes hours before any compute starts.
The manual loop avoids the graph entirely; output progress is per
chunk via tqdm.

Currently rigid-only. ANTs / elastix composition: TODO.
"""
from __future__ import annotations

import itertools
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from ome_zarr.io import parse_url

from ..atlas import load_ccf
from ..frame import compute_output_frame
from ..io import open_multiscale
from ..state import load as load_state, save as save_state
from ..transform import RigidTransform


def run(
    state_path: Path,
    out_zarr: Path,
    levels: Optional[list[int]] = None,
    output_chunks: Optional[tuple[int, int, int]] = None,
    max_workers: Optional[int] = None,
) -> None:
    from scipy.ndimage import affine_transform

    state = load_state(Path(state_path))
    if state.transform is None:
        raise RuntimeError("state.json has no transform — run `vol2atlas prealign` first.")

    ms  = open_multiscale(state.sample_zarr)
    ccf = load_ccf(state.atlas_name)

    atlas_voxel_um = np.asarray(ccf.voxel_um, dtype=float)

    rigid = RigidTransform(
        **{k: state.transform[k] for k in
           ["rz_deg", "ry_deg", "rx_deg", "tz_um", "ty_um", "tx_um"]},
        center_um=tuple(state.transform.get("center_um", (0.0, 0.0, 0.0))),
    )
    M_um = rigid.matrix()
    if state.affine is not None:
        A_um = np.asarray(state.affine, dtype=float)
        M_um = A_um @ M_um
        print(f"[alignFull] composing state.affine on top of rigid")
    Minv_um = np.linalg.inv(M_um)

    avail = list(range(ms.n_levels()))
    target_levels = levels if levels else avail
    bad = [l for l in target_levels if l not in avail]
    if bad:
        raise ValueError(f"Requested levels {bad} not in available {avail}")

    out_zarr = Path(out_zarr)
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    out_zarr.mkdir(parents=True, exist_ok=True)
    store = parse_url(str(out_zarr), mode="w").store
    root  = zarr.group(store=store)

    # Per-level voxel size honoring any state.json override.
    def voxel_um_at(level: int) -> np.ndarray:
        sp_lev = np.asarray(ms.spacing(level), dtype=float)
        if state.sample_voxel_um is None:
            return sp_lev
        sp_anc = np.asarray(ms.spacing(state.sample_level), dtype=float)
        anc_um = np.asarray(state.sample_voxel_um, dtype=float)
        return anc_um * (sp_lev / sp_anc)

    if max_workers is None:
        max_workers = int(os.environ.get("DASK_NUM_WORKERS",
                                          os.cpu_count() or 4))

    # Output frame = user's CCF crop bbox (no union with rotated sample).
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

    scales: list[list[float]] = []
    for out_idx, level in enumerate(target_levels):
        sample_um = voxel_um_at(level)

        # The dask-array level wraps the underlying zarr. Get the raw zarr
        # array for direct (chunk-aware) reads with no graph overhead.
        arr_dask = ms.level(level)
        if "c" in ms.axes:
            arr_dask = arr_dask[state.sample_channel]
        in_shape = tuple(int(s) for s in arr_dask.shape)
        in_dtype = arr_dask.dtype

        out_shape = tuple(int(np.ceil(crop_size_um[a] / sample_um[a]))
                          for a in range(3))

        D      = np.diag(sample_um)
        D_inv  = np.diag(1.0 / sample_um)
        M_v    = D_inv @ Minv_um[:3, :3] @ D
        off_v  = D_inv @ (Minv_um[:3, :3] @ crop_origin_um + Minv_um[:3, 3])

        chunks_out = (output_chunks if output_chunks is not None
                      else _pick_default_chunks(arr_dask))

        n_vox   = int(np.prod(out_shape))
        raw_gb  = n_vox * np.dtype(in_dtype).itemsize / 1e9
        n_blocks = tuple(int(np.ceil(out_shape[i] / chunks_out[i]))
                          for i in range(3))
        n_total  = int(np.prod(n_blocks))
        print(f"[alignFull] level {level} → out '{out_idx}'  "
              f"shape={out_shape}  voxel={tuple(sample_um)} µm  "
              f"chunks={chunks_out}  {n_total} output chunks  "
              f"~{raw_gb:.2f} GB raw (~{raw_gb*0.4:.2f} GB compressed expected)",
              flush=True)

        # Allocate the output zarr array.
        out_arr = zarr.open(
            str(out_zarr / str(out_idx)), mode="w",
            shape=out_shape, chunks=chunks_out, dtype=in_dtype,
        )

        # Bypass dask for input reads; go straight to the underlying zarr.
        in_zarr = _underlying_zarr(arr_dask)

        # Iterate output chunks in parallel, each as a small independent task.
        block_indices = list(itertools.product(*[range(n) for n in n_blocks]))

        def _process_block(idx_block):
            return _warp_one_chunk(
                idx_block, chunks_out, out_shape,
                M_v, off_v, in_zarr, in_shape, out_arr,
                affine_transform=affine_transform,
            )

        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_block, b) for b in block_indices]
            for fut in tqdm(as_completed(futures), total=n_total,
                             desc=f"L{level}→{out_idx}", unit="chk"):
                fut.result()

        scales.append([float(sample_um[0]),
                       float(sample_um[1]),
                       float(sample_um[2])])
        print(f"[alignFull] wrote {out_zarr.name}/{out_idx}")

    root.attrs["multiscales"] = [{
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {"path": str(i),
             "coordinateTransformations": [{"type": "scale", "scale": s}]}
            for i, s in enumerate(scales)
        ],
        "version": "0.4",
    }]

    state.add_history("alignFull",
                       f"levels={target_levels} chunks={output_chunks}")
    save_state(state, Path(state_path))
    print(f"[alignFull] done. multiscale OME-Zarr at {out_zarr}")


def _pick_default_chunks(arr_dask) -> tuple[int, int, int]:
    """Pick a sane default output chunk size. Larger than the input's chunk
    size on small-chunk inputs (so the output zarr isn't fragmented into
    millions of tiny files); equal to it otherwise."""
    try:
        in_chunks = tuple(int(c) for c in arr_dask.chunksize[-3:])
    except Exception:
        return (256, 256, 256)
    return tuple(max(c, 128) for c in in_chunks)


def _underlying_zarr(arr_dask):
    """Pull the raw zarr.Array out from under a dask array.

    open_multiscale wraps zarr levels in dask. For direct reads we want
    the underlying zarr array; that lets us slice arbitrary regions
    without building a dask graph.
    """
    # If it's already a zarr array, just return it.
    if isinstance(arr_dask, zarr.Array):
        return arr_dask
    # dask arrays from from_zarr have the zarr in .name's source — but the
    # robust way is to walk .dask graph and find the zarr layer. Easier:
    # just slice the whole thing through dask once. But that defeats the
    # purpose. Use the documented interface:
    try:
        # dask.array.from_zarr stores the underlying array as the leaf
        # value of the dask graph's first task.
        keys = list(arr_dask.__dask_keys__())
        layer = arr_dask.__dask_graph__().layers
        for lname, lyr in layer.items():
            # Look for the input store value
            for v in lyr.values():
                if isinstance(v, zarr.Array):
                    return v
    except Exception:
        pass
    # Fallback: wrap in an adapter that uses dask slicing for region reads.
    class _DaskWrapper:
        def __init__(self, da): self.da = da; self.shape = da.shape
        def __getitem__(self, sl): return np.asarray(self.da[sl])
    return _DaskWrapper(arr_dask)


def _warp_one_chunk(
    idx_block, chunks_out, out_shape,
    M_v, off_v, in_zarr, in_shape, out_arr,
    *, affine_transform,
):
    """Compute one output chunk and write it.

    For an output chunk starting at `out_origin` with shape `chunk_shape`:
      1. compute the inverse-mapped source region (axis-aligned bounding
         box of `M_v @ out_voxel + off_v` over the chunk corners),
      2. clip + pad to the source bounds,
      3. read that source sub-array directly from the zarr,
      4. run scipy.ndimage.affine_transform locally with the same M_v
         and the offset adjusted for the local sub-array origin,
      5. write the chunk into the output zarr.
    """
    # Output slab indices.
    z0 = idx_block[0] * chunks_out[0]; z1 = min(z0 + chunks_out[0], out_shape[0])
    y0 = idx_block[1] * chunks_out[1]; y1 = min(y0 + chunks_out[1], out_shape[1])
    x0 = idx_block[2] * chunks_out[2]; x1 = min(x0 + chunks_out[2], out_shape[2])
    out_origin = np.array([z0, y0, x0], dtype=float)
    out_shape_local = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=int)

    # Map the 8 corners of the output chunk back into source voxel space
    # to find the axis-aligned bounding box we need to read.
    corners = np.array(list(itertools.product(
        [0, out_shape_local[0] - 1],
        [0, out_shape_local[1] - 1],
        [0, out_shape_local[2] - 1],
    )), dtype=float)
    corners_out_global = corners + out_origin
    src_corners = (M_v @ corners_out_global.T).T + off_v[None, :]
    src_min = np.floor(src_corners.min(axis=0)).astype(int) - 2
    src_max = np.ceil (src_corners.max(axis=0)).astype(int) + 2

    # Clip to the source bounds.
    src_min_cl = np.maximum(src_min, 0)
    src_max_cl = np.minimum(src_max, np.array(in_shape))

    # If the source region is empty (output chunk is entirely outside the
    # sample), write zeros and bail.
    if np.any(src_max_cl <= src_min_cl):
        out_arr[z0:z1, y0:y1, x0:x1] = 0
        return

    # Read the source sub-array directly from zarr.
    src = in_zarr[
        src_min_cl[0]:src_max_cl[0],
        src_min_cl[1]:src_max_cl[1],
        src_min_cl[2]:src_max_cl[2],
    ]
    src = np.asarray(src)

    # The affine_transform we feed scipy maps OUTPUT-local index → SOURCE-local
    # index. With M_v, off_v acting on GLOBAL coords:
    #   src_global = M_v @ (out_global) + off_v
    # We want src_local = src_global - src_min_cl, and out_global = out_local + out_origin:
    #   src_local = M_v @ (out_local + out_origin) + off_v - src_min_cl
    #             = M_v @ out_local + (M_v @ out_origin + off_v - src_min_cl)
    off_local = M_v @ out_origin + off_v - src_min_cl

    warped = affine_transform(
        src, M_v, offset=off_local,
        output_shape=tuple(out_shape_local),
        order=1, mode="constant", cval=0,
    ).astype(out_arr.dtype)
    out_arr[z0:z1, y0:y1, x0:x1] = warped


# TODO: --transform <dir> for ANTs/elastix deformable composition.
# Per-chunk wrap of ants.apply_transforms or transformix, composed with
# the rigid above. Both tools are chunk-safe; integration just walks the
# same block iteration as _warp_one_chunk.
