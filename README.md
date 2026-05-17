# vol2atlas

Interactive alignment of large OME-Zarr volumes (µCT, light-sheet, …) to
the Allen Mouse Brain Common Coordinate Framework (CCF), producing a
multiscale OME-Zarr in atlas space ready for Neuroglancer.

Hand-driven rigid alignment first (three interactive steps), then any of:

- landmark-driven thin-plate-spline correction on top of rigid (`export --tps`)
- intensity-driven ANTs `SyNOnly` with brain masks (`ants`)
- elastix B-spline FFD with Mattes MI (`elastix`)
- BrainGlobe brainreg / NiftyReg (`brainreg`, full brain or hemisphere only)
- chunked rigid apply to a TB-scale OME-Zarr at *all* pyramid levels (`alignFull`)

All output methods write the same NIfTI + multiscale-OME-Zarr layout, so a
single QC viewer can flip between them on the same atlas.

## Install

```bash
conda create -n vol2atlas -c conda-forge python=3.11 napari pyqt
conda activate vol2atlas
pip install -e .
```

Optional refinement engines (install only the ones you actually want):

```bash
pip install -e ".[ants]"        # antspyx
pip install -e ".[elastix]"     # itk-elastix
pip install -e ".[brainreg]"    # brainreg
pip install -e ".[slides]"      # python-pptx (regenerate methods_survey.pptx)
pip install -e ".[full]"        # everything above
```

## Pipeline

Every step reads and writes a single `state.json`. Quit + resume any time.

```bash
vol2atlas init  /data/sample.zarr  --level 2  --voxel-um 2.74  --atlas allen_mouse_25um
vol2atlas prealign state.json   # 3D rough prealign + CCF crop
vol2atlas refine state.json   # fine refine in axial / coronal / sagittal views
vol2atlas landmarks state.json   # pick landmark pairs (optional Procrustes refit)

# write a downsampled output suitable for QC (~25 µm voxels, fast)
vol2atlas export state.json -o out/rigid                     # rigid only
vol2atlas export state.json -o out/tps    --tps              # rigid + TPS-on-residuals (needs ≥4 landmarks)

# optional deformable refinement on top of an `export` output (intensity-driven, no landmarks needed)
vol2atlas ants     out/rigid -o out/ants                     # ANTs SyNOnly + masks
vol2atlas elastix  out/rigid -o out/elastix                  # elastix B-spline FFD
vol2atlas brainreg out/rigid -o out/brainreg                 # brainreg (assumes full brain/hemisphere)

# apply the saved rigid to the FULL OME-Zarr at every pyramid level (chunked, dask-image)
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr
```

Output of `export` / `ants` / `elastix` / `brainreg` under `out/<dir>/`:

```
sample_in_ccf.nii.gz     atlas-space NIfTI of the warped sample
atlas_cropped.nii.gz     the cropped CCF reference, same grid
sample_in_ccf.zarr/      multiscale OME-Zarr of the sample (5 pyramid levels)
atlas_cropped.zarr/      multiscale OME-Zarr of the atlas (5 pyramid levels)
```

`alignFull` writes a SINGLE multiscale OME-Zarr whose levels mirror the
input pyramid (typically much larger — TB-scale at level 0).

## QC

```bash
# atlas vs. one method
python scripts/qc_export.py out/rigid

# atlas vs. several methods at once (toggle layers to compare)
python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/elastix
```

## Serving to Neuroglancer

```bash
cd out/rigid && python -m http.server 8000 --bind 127.0.0.1
# or, after alignFull:
cd /data/big && python -m http.server 8000 --bind 127.0.0.1
```

Then in Neuroglancer, add layers:

```
zarr://http://127.0.0.1:8000/atlas_cropped.zarr
zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

Remote: `ssh -L 8000:127.0.0.1:8000 user@server` first.

## What each command does

### `vol2atlas init <zarr>`
Records sample path, pyramid level, voxel size, channel, target atlas
into `state.json`. No napari window.

### `vol2atlas prealign state.json` — 3D rough prealign + CCF crop
3D MIP napari window. Six sliders (translation / rotation) + flip
checkboxes. The displayed sample IS a live scipy resample onto the CCF
voxel grid — what you see IS what gets saved. A second tab provides
six min/max sliders to crop the CCF to a bounding box around your
sample (so subsequent steps work on the relevant subvolume only).

### `vol2atlas refine state.json` — fine refine in ortho views
Single napari window with axial / coronal / sagittal switch buttons.
Tighter slider ranges (±3 mm, ±30°) for precise tuning.

### `vol2atlas landmarks state.json` — pick pairs + optional Procrustes
- `+ SAMPLE (next click)` → next click adds a landmark on the sample.
- `+ CCF (next click)` → next click adds the corresponding atlas landmark.
- Same row index = corresponding pair.
- `Fit` runs Procrustes/Kabsch on ≥3 pairs and **replaces** the saved
  rigid transform. **Skip Fit if your hand alignment from prealign/refine
  is already good** — a Procrustes fit on a few noisy landmarks can be
  worse than the hand pose.
- Landmarks are saved to `state.json` regardless of whether you Fit.

### `vol2atlas export state.json` — downsampled output for QC
Uses the same resample as the `landmarks` viewer to write the sample on the cropped CCF grid,
plus the cropped atlas, both via one shared writer with one identical
diagonal affine. Same code path for both ⇒ correct overlay.

With `--tps`: fits a thin-plate spline on the *residuals* between the
rigid prediction and your landmark clicks. Far from landmarks the
correction → 0 (rigid is preserved); at landmarks the sample lands
exactly on its CCF target.

`--tps-smoothing X`: relax exact-interpolation. Useful when landmarks
are noisy. Units of µm²; 0 = exact, 2500 ≈ tolerates ±50 µm click error.

### `vol2atlas ants <export_dir>` — ANTs SyNOnly refinement
Intensity-driven deformable refinement on top of an `export` output.
Uses `type_of_transform='SyNOnly'` (the rigid+affine stages of plain
`SyN` would overwrite your prealignment) plus auto-built brain masks
(unmasked metrics on partial samples cause the field to run wild into
zero regions). Result lands in the same NIfTI + multiscale-zarr layout
as `export`.

Tunable knobs: `--iter-high/--iter-mid/--iter-low`, `--flow-sigma`,
`--total-sigma`, `--transform`. Defaults are conservative for partial
samples; bump iterations / lower sigmas to let SyN deform more.

Requires: `pip install -e ".[ants]"`.

### `vol2atlas elastix <export_dir>` — elastix B-spline FFD
B-spline free-form deformation via `itk-elastix`. Defaults try to match
Humbel et al. 2024 (µCT mouse brain → CCF, 0.08 mm landmark error on a
full brain): grid 16 voxels, bending-energy weight 1000, 4-level
pyramid, Mattes MI metric, Otsu masks. **Note:** the wrapper currently
falls back to stock elastix defaults plus grid/iter/resolution overrides
because the multi-metric + masked-sampler combination is fragile on
partial samples; the `--bending` flag is effectively informational and
smoothness comes from the B-spline grid spacing alone.

Requires: `pip install -e ".[elastix]"`.

### `vol2atlas brainreg <export_dir>` — BrainGlobe brainreg refinement
Wraps brainreg (NiftyReg engine). **Assumes a full brain or a complete
hemisphere** — on a partial sample it will distort to fill the missing
region. Kept for reference; not recommended for partial-hemisphere data.

Requires: `pip install -e ".[brainreg]"`.

### `vol2atlas alignFull state.json` — full-resolution chunked rigid warp
Applies the rigid transform from `state.json` to the FULL OME-Zarr at
every (or chosen) pyramid level, chunkwise via `dask_image.ndinterp`.
RAM stays bounded by chunk size regardless of total volume size, so
TB-scale level 0 is fine if you have the disk for it.

Default `--levels` = all available levels of the input zarr. Output is
written one level at a time with a per-chunk tqdm progress bar; final
OME-Zarr `multiscales` metadata is stitched at the end.

```bash
# typical TB-scale run, all levels
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr

# just the lowest-resolution level for a quick smoke test
vol2atlas alignFull state.json --levels 2 -o out/quick.zarr

# throttle dask workers if RAM is tight
DASK_NUM_WORKERS=4 vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr
```

Currently rigid-only. ANTs / elastix deformable composition into
`alignFull` is on the TODO inside `src/vol2atlas/steps/align_full.py`.

## TPS limitations

- Needs ≥4 landmark pairs.
- With few or clustered landmarks the spline has no support outside the
  cluster and extrapolates wildly. **Spread your landmarks across the
  whole volume** (front / back / top / bottom / left / right of the crop).
- If `--tps` makes the overlay worse than rigid-only, you don't have
  enough landmark coverage — add more, or stay rigid.

## Comparing methods

`export`, `ants`, `elastix`, and `brainreg` all write to whatever
`-o <dir>` you pass; nothing is overwritten and `state.json` is only
appended-to (history line). A typical compare-all run:

```bash
vol2atlas export   state.json -o out/rigid
vol2atlas export   state.json -o out/tps --tps
vol2atlas ants     out/rigid  -o out/ants
vol2atlas elastix  out/rigid  -o out/elastix
vol2atlas brainreg out/rigid  -o out/brainreg

python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/elastix out/brainreg
```

## Repo layout

```
src/vol2atlas/
  cli.py                    Typer CLI
  io.py                     OME-Zarr multiscale reader
  atlas.py                  Allen CCF loader (brainglobe-atlasapi)
  state.py                  State dataclass + load/save (state.json)
  transform.py              RigidTransform (rot + trans + flips, in µm)
  steps/
    step1_prealign.py       3D rough prealign + CCF crop
    step2_refine.py         fine refine in ortho views
    landmarks_landmarks.py      landmark pairs + Procrustes
    step_export.py          downsampled NIfTI + multiscale OME-Zarr (rigid, + optional TPS-on-residuals)
    refine_ants.py          ANTs SyNOnly + masks on top of an export
    refine_elastix.py       elastix B-spline FFD on top of an export
    refine_brainreg.py      brainreg on top of an export
    align_full.py           chunked dask-image rigid apply to all pyramid levels
scripts/
  qc_export.py              napari overlay of atlas vs. one warped sample
  qc_compare_methods.py     napari overlay of atlas vs. several methods (toggleable layers)
  make_methods_slides.py    regenerate methods_survey.pptx (literature overview)
methods_survey.pptx         literature review: registration methods landscape
```

## Coordinate conventions

- All math in (z, y, x) µm.
- Landmarks in `state.json`:
  - `sample_um[i]` = intrinsic sample µm (so it survives transform updates).
  - `ccf_um[i]`    = full-CCF µm (not cropped).
- RigidTransform: `output_µm = R · F · (input_µm − center_µm) + center_µm + (tz, ty, tx)`,
  R = intrinsic ZYX Euler, F = `diag(±1, ±1, ±1)` from the three flip flags.

## Caveats

- Partial-hemisphere samples are fully supported by the rigid + TPS
  path (`prealign` crop + landmark pinning). `brainreg` and the affine
  stage of plain `SyN` assume a full brain or hemisphere and will
  distort to fill the missing region — `ants` defaults to `SyNOnly` +
  masks to avoid this; `elastix` is currently fragile on partial
  samples (sampling edge cases — see code comments).
- landmarks's `Fit` button **replaces** the saved rigid transform with the
  Procrustes solution from your landmarks. A fit on few or noisy
  landmarks can be worse than the prealign/refine hand pose; if your hand
  alignment already looks right, just Save and exit without clicking Fit.
- Deformable warps distort feature shape/size. For region-based
  analysis (cell counting, region IDs, cross-sample averaging) this is
  fine — that's the point. For measuring true anatomical dimensions
  (volumes, lengths, pathology), keep the rigid result and work in
  sample-native coordinates.

## References

- BrainGlobe atlasapi: https://brainglobe.info/documentation/brainglobe-atlasapi/
- OME-Zarr spec: https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3: https://atlas.brain-map.org/
- Humbel et al. 2024, "Cellular-resolution X-ray microtomography of an
  entire mouse brain" — https://arxiv.org/abs/2405.13971
- See [methods_survey.pptx](methods_survey.pptx) for a literature
  overview of registration methods.
