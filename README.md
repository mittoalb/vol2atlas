# zrot

Hand-driven prealignment of large OME-Zarr volumes (X-ray µCT, light-sheet, etc.)
to the Allen Mouse Brain Common Coordinate Framework (CCF), paired with
[BrainGlobe brainreg](https://brainglobe.info/documentation/brainreg/index.html)
for the deformable refinement.

The interactive aligner solves the orientation/initial-pose problem (where
brainreg's auto-init typically fails on µCT data); brainreg solves the affine +
B-spline refinement (where rolling your own elastix pipeline typically blows up
on small joint masks). End result: a multiscale OME-Zarr in atlas space, ready
for Neuroglancer.

## What's in the box

```
src/zrot/
  align.py       interactive aligner (napari, scipy resample as live view)
  atlas.py       Allen CCF loader (BrainGlobe atlasapi)
  io.py          OME-Zarr multiscale reader
  transform.py   RigidTransform: rotations + translations + axis flips, in µm
  refine.py      one helper: _resample_to_ccf_grid (used by the prealign script)
  cli.py         `zrot-align align`, `zrot-align info`
scripts/
  prealign_for_brainreg.py   zrot transform -> CCF-coords NIfTI for brainreg
  tiff_to_ngff.py            brainreg's TIFF -> multiscale OME-Zarr
  compare_with_ccf.py        napari side-by-side QC
  view_atlas.py              browse atlas + parcellation in napari
```

## Install

```bash
conda create -n zrot -c conda-forge python=3.11 \
    napari pyqt zarr ome-zarr dask scipy numpy nibabel tifffile
conda activate zrot
pip install brainglobe-atlasapi typer magicgui brainreg
pip install -e .
```

## Workflow

```bash
# 1. (Optional) inspect the zarr's pyramid
zrot-align info /data/sample.zarr

# 2. Hand-align in napari. The displayed sample IS the scipy resample —
#    what you see IS what gets saved IS what brainreg gets.
zrot-align align /data/sample.zarr \
    --level 2 --channel 0 --save transform.json \
    --ndisplay 2 --preview-voxels $((192**3))

# 3. Resample the sample into CCF coordinates as a NIfTI for brainreg
python scripts/prealign_for_brainreg.py \
    /data/sample.zarr  transform.json  sample_prealigned.nii.gz \
    --atlas allen_mouse_25um

# 4. brainreg refines (affine + B-spline; orientation already correct)
brainreg sample_prealigned.nii.gz ./brainreg_out \
    -v 25 25 25 --orientation asr --atlas allen_mouse_25um \
    --brain_geometry hemisphere_r       # or full / hemisphere_l

# 5. Convert brainreg's TIFF output to multiscale OME-Zarr (for Neuroglancer)
python scripts/tiff_to_ngff.py \
    brainreg_out/downsampled_standard.tiff \
    out/sample_in_ccf.zarr  --voxel-um 25

# 6. QC overlay vs CCF + parcellation
python scripts/compare_with_ccf.py out/sample_in_ccf.zarr
```

## Serving to Neuroglancer

```bash
cd out && python -m http.server 8000 --bind 127.0.0.1
# In Neuroglancer:  zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

For remote-machine viewing, port-forward from your laptop:
`ssh -L 8000:127.0.0.1:8000 user@server`, then point Neuroglancer at
`http://127.0.0.1:8000/...` locally.

## The aligner — what's where in the napari window

Right dock panels:

- **absolute pose** — live readout of `tz/ty/tx (µm)`, `rz/ry/rx (deg)`, active flips.
- **Δ from baseline** — six sliders + spinboxes for rotation/translation, and a
  **flip: [z] [y] [x]** row of checkboxes for axis mirrors. Sliders represent
  *deltas* from the baseline pose (start at 0, in the middle of the track).
- **jog Δ** — pick a step size from the dropdown, click +/- buttons to nudge by
  exactly that amount. Useful for fine work.
- **commit** — fold the current Δ into the baseline; sliders snap back to 0.
  Use when you've moved a slider near its end and want more travel.
- **save** — write `transform.json`. Watch the terminal for confirmation.
- **reset Δ** — zero the sliders without changing the baseline.

Keyboard shortcuts (focus the canvas first by clicking on it):

- `← / →` nudge `tx` by current jog step
- `↑ / ↓` nudge `ty`
- `PgUp / PgDn` nudge `tz`
- `q / e` rotate around z by current jog step
- **Shift + drag** on the sample layer in the canvas → translate by mouse motion

Mode:

- `--ndisplay 2`: 2D slice view (default). Scroll z with the bottom slider.
- `--ndisplay 3`: 3D MIP. Mouse-drag to rotate the camera. Slower per redraw.

The displayed sample layer **is** a scipy resample of your data through the
current transform onto the CCF grid — recomputed (debounced 200 ms) on every
slider change. There is no fake affine layer to mislead you. Everything you see
is what `prealign_for_brainreg.py` will produce.

## RigidTransform conventions

`transform.json` holds a transform in **physical micrometres**, mapping
sample µm → CCF µm. It's scale-invariant: the same matrix is valid at every
pyramid level when composed with that level's voxel size.

Order of operations (around `center_um`, the sample's geometric center):

```
output_µm = R · F · (input_µm − center_µm) + center_µm + (tz, ty, tx)
```

i.e. **flip → rotate → translate**. `R` is intrinsic ZYX Euler from
`(rz_deg, ry_deg, rx_deg)`; `F` is `diag(±1, ±1, ±1)` from the three flip flags.

## Helper scripts

### `prealign_for_brainreg.py`
Reads your zarr + `transform.json`, applies the rigid transform via scipy onto
the CCF voxel grid, writes a NIfTI in atlas orientation. brainreg consumes
this with `--orientation asr` (no orientation guessing needed — the prewarp
already lives in atlas space).

### `tiff_to_ngff.py`
brainreg writes its registered output as a single-resolution TIFF. Neuroglancer
needs a multiscale pyramid. This script wraps the TIFF in a 5-level OME-Zarr.

### `compare_with_ccf.py`
napari window with three layers: gray CCF reference, magenta registered sample,
labels for atlas regions (hidden by default — toggle on for IDs). Hover any
voxel for the region ID. `--grid` puts the layers side-by-side; `--diff` adds
a z-scored absolute-difference layer to highlight misalignment.

### `view_atlas.py`
Standalone atlas browser. Searchable region panel on the right; click a region
to isolate it and jump napari to its centroid; hover for id/acronym/full name
in the status bar.

## File formats

| | Extension | Frame | Used by |
|---|---|---|---|
| OME-Zarr | `.zarr/` (directory) | µm, (z,y,x) | napari, neuroglancer, this repo |
| NIfTI | `.nii.gz` | mm, (x,y,z), RAS | brainreg, FSL, SPM |
| NRRD | `.nrrd` | mm, (x,y,z), LPS | ITK-SNAP, 3D Slicer |
| TIFF | `.tiff` | unitless / metadata | brainreg outputs, Fiji |

NIfTI ≠ NRRD: same kind of data, different headers and default anatomical
frames. brainreg's `load_any` accepts NIfTI and TIFF but **not** NRRD.

## Coordinate conventions (gotchas)

- **numpy / OME-Zarr**: array axis order `(z, y, x)`; voxel sizes in **µm**.
- **NIfTI / NRRD**: physical coordinates in **mm**, vector order `(x, y, z)`;
  default frames RAS (NIfTI) and LPS (NRRD).
- The aligner works in (z, y, x) µm. `prealign_for_brainreg.py` converts to
  (x, y, z) mm RAS at the boundary so brainreg sees a NIfTI in atlas
  orientation already — `--orientation asr` is the right flag for that pass.

## Performance / scaling

- The aligner reads one strided slice of one pyramid level (~30-100 MB into
  RAM, depending on `--preview-voxels`). Each slider change triggers a scipy
  resample of that preview, ~200-500 ms.
- `prealign_for_brainreg.py` resamples one pyramid level at atlas-relevant
  resolution (~hundreds of MB).
- brainreg downsamples internally; runs in 2-5 minutes on the prealigned NIfTI.
- For full-resolution warping of TB-scale data into atlas space, you'd need to
  apply brainreg's deformation field (the `deformation_field_*.tiff` files) to
  the original full-res zarr, chunk-by-chunk. That's not in this repo yet —
  most use cases are happy with the 25 µm output.

## Caveats

- `--orientation` for brainreg is set to `asr` because the input has already
  been put in atlas orientation by the prealignment. Don't change it.
- `--brain_geometry` (note underscore — brainreg argparse quirk) must match
  what your sample contains: `full`, `hemisphere_l`, or `hemisphere_r`. Wrong
  choice = wildly distorted output.
- The aligner shows the truthful scipy resample on every slider tick. If
  dragging a slider feels sluggish, drop `--preview-voxels` (e.g.
  `$((128**3))`).

## References

- BrainGlobe brainreg: https://brainglobe.info/documentation/brainreg/index.html
- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/index.html
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
