# zrot

Interactive alignment of large OME-Zarr volumes (µCT, light-sheet, …) to
the Allen Mouse Brain Common Coordinate Framework (CCF), producing a
multiscale OME-Zarr in atlas space ready for Neuroglancer.

Hand-driven rigid alignment first (three steps), optional landmark-driven
thin-plate-spline (TPS) refinement on top. No brainreg, no ANTs — both
struggle on partial samples and silently reorient on write.

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
```

Outputs under `out/aligned/`:

```
sample_in_ccf.nii.gz     atlas-space NIfTI of the warped sample
atlas_cropped.nii.gz     the cropped CCF reference, same grid
sample_in_ccf.zarr/      multiscale OME-Zarr of the sample (5 levels)
atlas_cropped.zarr/      multiscale OME-Zarr of the atlas (5 levels)
```

QC overlay in napari:

```bash
python scripts/qc_export.py out/aligned
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
  cli.py              Typer CLI (init / step1..3 / export / info)
  io.py               OME-Zarr multiscale reader
  atlas.py            Allen CCF loader (brainglobe-atlasapi)
  state.py            State dataclass + load/save (state.json)
  transform.py        RigidTransform (rot + trans + flips, in µm)
  steps/
    step1_prealign.py    3D rough prealign + CCF crop
    step2_refine.py      fine refine in ortho views
    step3_landmarks.py   landmark pairs + Procrustes
    step_export.py       write NIfTI + multiscale OME-Zarr
scripts/
  qc_export.py        napari overlay of atlas vs. warped sample
```

## Coordinate conventions

- All math in (z, y, x) µm.
- Landmarks in `state.json`:
  - `sample_um[i]` = intrinsic sample µm (so it survives transform updates).
  - `ccf_um[i]`    = full-CCF µm (not cropped).
- RigidTransform: `output_µm = R · F · (input_µm − center_µm) + center_µm + (tz, ty, tx)`,
  R = intrinsic ZYX Euler, F = `diag(±1, ±1, ±1)` from the three flip flags.

## Caveats

- Partial-hemisphere samples are fully supported (the cropping in step1
  is what makes that work). brainreg and ANTs SyN are *not* used by
  default — they reorient on save and/or assume whole-brain inputs.
- step3's `Fit` can overwrite a good hand alignment with a worse
  Procrustes solution if landmarks are few or noisy. Save without
  fitting if step1/step2 already look right.

## References

- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
