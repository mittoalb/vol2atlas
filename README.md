# vol2atlas

Align large OME-Zarr volumes (µCT, light-sheet, EM, …) to the Allen Mouse
CCF — or to any other OME-Zarr reference — through one CLI.

## Install

```bash
conda create -n vol2atlas -c conda-forge python=3.11 napari pyqt
conda activate vol2atlas
pip install -e .
```

Optional refinement engines:

```bash
pip install -e ".[ants]"       # antspyx
pip install -e ".[elastix]"    # itk-elastix
pip install -e ".[brainreg]"   # brainreg (full brain / hemisphere only)
pip install -e ".[full]"       # everything
```

## Use

Every step reads and writes one `state.json`. Quit and resume any time.

```bash
vol2atlas init  /data/sample.zarr  --level 2  --voxel-um 2.74 \
                --atlas allen_mouse_25um  --orientation pir
# --atlas defaults to allen_mouse_25um. Run `vol2atlas list-atlases`
# for all options (Allen 10/25/50/100 µm, Kim, Osten, Princeton,
# Gubra, BlueBrain CCFv3-augmented, etc.). Pick a coarse resolution
# for the interactive work — see "Switching atlas resolution" below
# to upgrade to 10 µm only for the final refinement / export.
#
# --orientation is a 3-letter BrainGlobe code (R/L, A/P, S/I). Each
# letter = direction the corresponding numpy axis INCREASES TOWARD.
# Allen CCF default is "asr". Pre-rotates the sample to match the atlas
# so prealign opens with the sample roughly oriented (no manual
# ±90° / 180° spinning). Sample must be right-handed; init refuses a
# left-handed orientation (would require a flip — not supported).
#
# You can also set it on prealign (overrides existing transform only,
# does NOT wipe other state fields like crops / landmarks):
vol2atlas prealign state.json --orientation pir
# Re-run prealign with different codes to iterate; ccf_crop_bbox and
# everything else are preserved.

vol2atlas prealign  state.json     # 3D rough alignment + CCF crop
vol2atlas refine    state.json     # fine refine in ortho views

# Higher-resolution preview for any interactive step:
vol2atlas refine    state.json --level 1 --preview-size 1000

# RIGID refinement — pick one (both write state.transform):
vol2atlas landmarks state.json     # interactive landmark pairs + Procrustes
vol2atlas mi        state.json     # automated rigid via MI (ANTs)

# AFFINE refinement — pick one (both write state.affine):
vol2atlas landmarks state.json     # in the GUI: click "Fit AFFINE from landmarks"
vol2atlas mi        state.json --affine   # global 12-DOF MI on intensity / shape

# In the landmarks GUI you can also run MI / JOINT MI+landmarks
# (single-shot or iterative) directly via buttons. Those run IN-MEMORY:
# state.json is only written when you click an explicit Save button.

# downsampled QC output
vol2atlas export    state.json -o out/rigid
vol2atlas export    state.json -o out/tps --tps     # rigid + TPS on residuals

# optional DEFORMABLE refinement on top of an export
vol2atlas ants      out/rigid -o out/ants
vol2atlas elastix   out/rigid -o out/elastix
vol2atlas brainreg  out/rigid -o out/brainreg

# apply rigid (+ affine if present) to the FULL OME-Zarr at every pyramid level
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr
```

Each refinement command writes the same layout under `out/<dir>/`:

```
sample_in_ccf.nii.gz     warped sample in atlas space
atlas_cropped.nii.gz     CCF reference, same grid
sample_mask.nii.gz       binary: where sample data exists inside the crop
sample_in_ccf.zarr/      multiscale OME-Zarr (5 levels)
atlas_cropped.zarr/      multiscale OME-Zarr (5 levels)
sample_mask.zarr/        multiscale OME-Zarr (5 levels)
```

Full help for every subcommand (args, options, defaults):

```bash
vol2atlas help-all
```

## EM (Stage B) — ROADMAP, not implemented

> **Not in `main`.** The `--reference` flag on `init`, `alignFull --tps`,
> and the voxel-size-aware UI are all unmerged. No EM data has been run
> through this pipeline. The block below is the target shape only.

```bash
# (proposed) Stage A — µCT to CCF
vol2atlas init      sample_uct.zarr -s state_uct.json --voxel-um 2.74
vol2atlas prealign  state_uct.json
vol2atlas refine    state_uct.json
vol2atlas landmarks state_uct.json
vol2atlas alignFull state_uct.json -o out/uct_in_ccf.zarr

# (proposed) Stage B — EM to that aligned µCT
vol2atlas init      sample_em.zarr -s state_em.json \
                    --reference out/uct_in_ccf.zarr --voxel-um 0.004
vol2atlas prealign  state_em.json
vol2atlas refine    state_em.json
vol2atlas landmarks state_em.json
vol2atlas alignFull state_em.json --tps -o out/em_in_ccf.zarr
```

See [vol2atlas_method.pptx](vol2atlas_method.pptx) and
[vol2atlas_method.docx](vol2atlas_method.docx) for the design and the
open risks (UI ergonomics at nm, OME-Zarr level selection on TB inputs,
TPS conditioning at nm).

## QC

```bash
python scripts/qc_export.py          out/rigid
python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/elastix
```

## Serve to Neuroglancer

```bash
cd out/rigid && python -m http.server 8000 --bind 127.0.0.1
```

Layers:

```
zarr://http://127.0.0.1:8000/atlas_cropped.zarr
zarr://http://127.0.0.1:8000/sample_in_ccf.zarr
```

Remote: `ssh -L 8000:127.0.0.1:8000 user@server` first.

## Switching atlas resolution

Landmarks (`sample_um`, `ccf_um`), `state.transform`, and `state.affine`
are stored in physical µm and are **resolution-independent**. Only
`state.ccf_crop_bbox` is in CCF voxel indices and must be rescaled.

```bash
# List available atlases (filtered to mouse)
vol2atlas list-atlases
vol2atlas list-atlases --downloaded         # only ones cached locally
vol2atlas list-atlases --species ""         # all species

# Switch atlas + auto-rescale crop bbox; landmarks unchanged
vol2atlas change-atlas state.json --to allen_mouse_10um
```

Allen CCFv3 comes at 10, 25 (default), 50, and 100 µm. Other mouse
atlases available: Kim Lab, Osten, Princeton (20 µm), Gubra (LSFM &
MRI), BlueBrain CCFv3-augmented, DeMBA developmental. First load of a
new resolution downloads + caches the atlas in `~/.brainglobe/`
(one-time, hundreds of MB for fine resolutions).

**Recommended strategy** for a final 10 µm alignment: do all the
interactive / iterative work (prealign, refine, landmarks, Fit AFFINE)
at the default 25 µm — napari stays responsive, ANTs runs are ~10×
faster, and finer atlas detail doesn't help when you're picking gross
correspondences. Then `change-atlas --to allen_mouse_10um`, optionally
re-run `mi --affine --shape` for fine MI refinement, and `export` /
`alignFull` to produce the 10 µm output. Cross-Allen-resolution
landmarks survive exactly — same µm coordinates index the same
anatomy at any voxel grid.

**Cross-atlas switches** (Allen → Princeton, Allen → Gubra, etc.) are
different reference brains with their own µm origins and label
schemes; landmarks will NOT survive. Restart from `init` with the
new `--atlas` if you change reference brain.

## Caveats

- Partial-hemisphere samples: use the rigid + TPS path. `brainreg` and
  plain `SyN` assume a full brain/hemisphere and will distort.
- `mi --affine` is a global 12-DOF refinement. If singular values
  print as 0.97–1.03 you've found a small real scale fix; if they
  diverge to e.g. 0.5 or 2.0 the optimizer over-fit — Revert in the
  preview and your `state.affine` stays unchanged.
- `alignFull` applies rigid + affine if both are set; deformable
  composition (TPS / B-spline) is not yet integrated.
- The CCF crop bbox from `prealign` is a HARD output frame for
  `export` and `alignFull` — sample voxels outside the crop are
  clipped. Expand the crop in `prealign` if you need more headroom.
  Interactive previews show the full sample so you can spot clipping.
- Prealign/refine UI now only exposes **translation + rotation** (no
  axis-flip checkboxes). Rotation sliders run −180°…+180°, which
  covers any orientation that flips would have covered.

## Landmarks workflow

Sample landmarks are stored in **raw sample µm** and reference the
**full** transform that renders the sample volume (`state.affine ∘
state.transform`). The click handler, display refresh, and resample
all use the same combined transform — landmarks remain anchored to
sample anatomy across fit operations.

**Iterating without re-clicking everything**:

1. Pick anatomy you can identify in BOTH volumes (vessels, ventricle
   corners, distinctive structure boundaries).
2. Click **Fit AFFINE** (or **Fit RIGID** if you don't want
   scale/shear).
3. Read the per-pair RMS report in the terminal — pairs flagged
   `← outlier?` (residual > 3× mean RMS) are bad clicks.
4. Select the bad row in the sample or CCF list → **Delete selected**.
5. Add more good landmarks if useful → **Fit** again.
6. Repeat 3–5 until residuals are uniformly small.

The transform improves each cycle PROVIDED you pick landmarks based on
**anatomy you can independently identify in both images**. Do NOT
click sample landmarks at positions designed to "compensate" for a
previous fit's residual — that encodes the prior fit's errors into the
next fit and drives the optimization away from truth.

**Revert** only when:

- A fit blew up (affine collapsed to near-singular, sample appears
  stretched/sheared visibly wrong).
- You want to discard the most recent transform change wholesale and
  keep your landmarks.

**Clear all + re-pick** only when:

- You inherit a landmark set captured under an incompatible code
  version (pre-frame-fix releases) or under a broken reference frame.
- It is faster than hunting and deleting individual outliers.

**GUI buttons that run MI / JOINT MI+landmarks**: these are
in-memory only. `state.json` is not touched by them. After the run
completes the preview updates; click **Save state.json (keep window
open)** or **Save state.json && exit** to persist.

The single-shot **JOINT MI + landmarks** button uses the landmark LSQ
fit as the ANTs initial moving transform, then refines with Mattes MI.
The **iterative JOINT** button alternates a regularized LSQ landmark
fit (pulled toward the previous MI result by `smooth_weight`) with a
short ANTs MI step, looping until the transform stops moving by `tol`.
Both write only to `state.affine` (with rigid reset to identity);
neither modifies disk until you Save.

## References

- BrainGlobe atlasapi — https://brainglobe.info/documentation/brainglobe-atlasapi/
- OME-Zarr spec — https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3 — https://atlas.brain-map.org/
- See [methods_survey.pptx](methods_survey.pptx) for a registration-methods literature overview.
- See [vol2atlas_strategy.pptx](vol2atlas_strategy.pptx) for the multi-modal scaling strategy.
