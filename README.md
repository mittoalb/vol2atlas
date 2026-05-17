# zrot

Interactive alignment of large OME-Zarr volumes (µCT, light-sheet, …) to
the Allen Mouse Brain Common Coordinate Framework (CCF), producing a
multiscale OME-Zarr in atlas space ready for Neuroglancer.

Hand-driven rigid alignment first (three steps), then any of:
- landmark-driven thin-plate-spline correction (`export --tps`),
- intensity-driven ANTs SyNOnly + masks (`zrot ants`),
- BrainGlobe brainreg (`zrot brainreg` — assumes a full brain/hemisphere).

All four paths write the same NIfTI + multiscale-OME-Zarr layout, so a
single QC viewer can flip between them on the same atlas.

## Install

```bash
conda create -n zrot -c conda-forge python=3.11 napari pyqt
conda activate zrot
pip install -e .
```

## Pipeline

Every step reads and writes a single `state.json`. Quit + resume any time.

```bash
zrot init  /data/sample.zarr  --level 2  --voxel-um 2.74  --atlas allen_mouse_25um
zrot step1 state.json   # 3D rough prealign + crop the CCF to your sample's extent
zrot step2 state.json   # fine refine in axial / coronal / sagittal views
zrot step3 state.json   # pick landmark pairs (optional Procrustes refit)
zrot export state.json -o out/aligned          # rigid-only
zrot export state.json -o out/aligned --tps    # rigid + TPS-on-residuals (needs ≥4 landmarks)

# optional deformable refinement of an `export` output (intensity-driven, no landmarks needed)
zrot ants     out/aligned -o out/ants          # ANTs SyNOnly + masks
zrot brainreg out/aligned -o out/brainreg      # BrainGlobe brainreg (NB: assumes full brain/hemisphere)
```

Outputs under `out/aligned/`:

```
sample_in_ccf.nii.gz     atlas-space NIfTI of the warped sample
atlas_cropped.nii.gz     the cropped CCF reference, same grid
sample_in_ccf.zarr/      multiscale OME-Zarr of the sample (5 levels)
atlas_cropped.zarr/      multiscale OME-Zarr of the atlas (5 levels)
```

QC overlay in napari (atlas vs. one method):

```bash
python scripts/qc_export.py out/aligned
```

QC across several methods at once (atlas as gray, each method as its
own toggleable colored layer):

```bash
python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/brainreg
```

## Serving to Neuroglancer

```bash
cd out/aligned && python -m http.server 8000 --bind 127.0.0.1
```

Then in Neuroglancer, add layers:

```
zarr://http://127.0.0.1:8000/atlas_cropped.zarr
zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

For remote work, port-forward: `ssh -L 8000:127.0.0.1:8000 user@server`.

## What each step does

### `zrot init <zarr>`
Records sample path, pyramid level, voxel size, channel, target atlas
into `state.json`. No napari window.

### `zrot step1 state.json` — rough prealign
3D MIP napari window. Six sliders (translation/rotation) + flip
checkboxes. The displayed sample IS a live scipy resample onto the CCF
voxel grid — what you see IS what gets saved. A second tab provides
six min/max sliders to crop the CCF to a bounding box around your
sample (so steps 2 / 3 / export work on the relevant subvolume only).

### `zrot step2 state.json` — fine refine
Single napari window with axial / coronal / sagittal switch buttons.
Tighter slider ranges (±3 mm, ±30°) for precise tuning.

### `zrot step3 state.json` — landmarks (+ optional Procrustes)
- `+ SAMPLE (next click)` → next click adds a landmark on the sample.
- `+ CCF (next click)` → next click adds the corresponding atlas landmark.
- Same row index = corresponding pair.
- `Fit` runs Procrustes/Kabsch on ≥3 pairs and **replaces** the saved
  rigid transform. **Skip Fit if your hand alignment from step1/step2
  is already good** — a Procrustes fit on a few noisy landmarks can be
  worse than the hand pose.
- Landmarks are saved to `state.json` regardless of whether you Fit.

### `zrot export state.json` — write outputs
Uses step3's exact resample to write the sample on the cropped CCF grid,
plus the cropped atlas, both via one shared writer with one identical
diagonal affine. Same code path for both ⇒ correct overlay.

With `--tps`: fits a thin-plate spline on the *residuals* between the
rigid prediction and your landmark clicks. Far from landmarks the
correction → 0 (rigid is preserved); at landmarks the sample lands
exactly on its CCF target.

`--tps-smoothing X`: relax exact-interpolation. Useful when landmarks
are noisy. Units of µm²; 0 = exact, 2500 ≈ tolerates ±50 µm click error.

`zrot export --help` for all flags.

### `zrot ants <export_dir>` — ANTs SyNOnly refinement
Intensity-driven deformable refinement on top of an `export` output.
Uses `type_of_transform='SyNOnly'` (the rigid+affine stages of plain
`SyN` would overwrite your prealignment) plus auto-built brain masks
(unmasked metrics on partial samples cause the field to run wild into
zero regions). Result lands in the same NIfTI + multiscale-zarr layout
as `export`, so `python scripts/qc_export.py <out>` Just Works.

Requires: `pip install antspyx`.

### `zrot brainreg <export_dir>` — BrainGlobe brainreg refinement
Same shape: takes an `export` output, runs brainreg, writes back in the
same layout. **brainreg assumes a full brain or a complete hemisphere**
— on a partial sample it will distort to fill the missing region.

Requires: `pip install brainreg`.

## TPS limitations

- Needs ≥4 landmark pairs.
- With few or clustered landmarks the spline has no support outside the
  cluster and extrapolates wildly. **Spread your landmarks across the
  whole volume** (front / back / top / bottom / left / right of the crop).
- If `--tps` makes the overlay worse than rigid-only, you don't have
  enough landmark coverage — add more, or stay rigid.

## Repo layout

```
src/zrot/
  cli.py                  Typer CLI (init / step1..3 / export / ants / brainreg / info)
  io.py                   OME-Zarr multiscale reader
  atlas.py                Allen CCF loader (brainglobe-atlasapi)
  state.py                State dataclass + load/save (state.json)
  transform.py            RigidTransform (rot + trans + flips, in µm)
  steps/
    step1_prealign.py     3D rough prealign + CCF crop
    step2_refine.py       fine refine in ortho views
    step3_landmarks.py    landmark pairs + Procrustes
    step_export.py        write NIfTI + multiscale OME-Zarr (rigid, + optional TPS-on-residuals)
    refine_ants.py        ANTs SyNOnly + masks on top of an export
    refine_brainreg.py    brainreg on top of an export
scripts/
  qc_export.py            napari overlay of atlas vs. one warped sample
  qc_compare_methods.py   napari overlay of atlas vs. several methods (one toggleable layer each)
```

## Coordinate conventions

- All math in (z, y, x) µm.
- Landmarks in `state.json`:
  - `sample_um[i]` = intrinsic sample µm (so it survives transform updates).
  - `ccf_um[i]`    = full-CCF µm (not cropped).
- RigidTransform: `output_µm = R · F · (input_µm − center_µm) + center_µm + (tz, ty, tx)`,
  R = intrinsic ZYX Euler, F = `diag(±1, ±1, ±1)` from the three flip flags.

## Comparing methods

`zrot export`, `zrot ants`, and `zrot brainreg` all write to whatever
`-o <dir>` you pass; nothing is overwritten and `state.json` is only
appended-to (history line). A typical compare-all run:

```bash
zrot export   state.json -o out/rigid                 # rigid baseline
zrot export   state.json -o out/tps --tps             # rigid + TPS-on-residuals
zrot ants     out/rigid  -o out/ants                  # ANTs SyNOnly + masks
zrot brainreg out/rigid  -o out/brainreg              # brainreg (likely poor on partial)

python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/brainreg
```

## Caveats

- Partial-hemisphere samples are fully supported by the rigid + TPS
  path (step1 crop + landmark pinning). `zrot brainreg` and the affine
  stage of plain `SyN` assume a full brain or hemisphere and will
  distort to fill the missing region — `zrot ants` defaults to
  `SyNOnly` + masks to avoid this.
- step3's `Fit` button **replaces** the saved rigid transform with the
  Procrustes solution from your landmarks. A fit on few or noisy
  landmarks can be worse than the step1/step2 hand pose; if your hand
  alignment already looks right, just Save and exit without clicking
  Fit.

## References

- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
