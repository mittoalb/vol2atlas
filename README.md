# vol2atlas

Align large OME-Zarr volumes (µCT, light-sheet, EM, …) to the Allen Mouse
CCF — or to any other OME-Zarr reference — through one CLI.

## Contents

- [Install](#install) — base + optional engines
- [Use](#use) — canonical end-to-end pipeline
- [Atlas selection & switching resolution](#switching-atlas-resolution) —
  `list-atlases`, `change-atlas`
- [Import / export landmarks](#import--export-landmarks) — round-trip
  external CSV / JSON
- [Live TPS preview in landmarks](#live-tps-preview-in-landmarks) —
  see deformable warp while picking
- [Local refinements (masked transforms)](#local-refinements-masked-transforms)
  — fix one region without disturbing the rest
- [Portable transform export](#portable-transform-export) — emit ITK
  `.mat` / OME-NGFF `.json`; reapply with `apply-transform`
- [Landmarks workflow tips](#landmarks-workflow) — iterating without
  blowing up the fit
- [Caveats](#caveats)
- [EM (Stage B) roadmap](#em-stage-b-roadmap-not-implemented)

Full per-command help: `vol2atlas help-all`.

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
# See "Live TPS preview" and "Local refinements" sections below for the
# in-GUI deformable preview and the masked-transform tools.

# downsampled QC output
vol2atlas export    state.json -o out/rigid
vol2atlas export    state.json -o out/tps     --tps              # rigid + TPS on residuals
vol2atlas export    state.json -o out/portable --write-transform # +transform.mat/json/landmarks.csv

# optional DEFORMABLE refinement on top of an export
vol2atlas ants      out/rigid -o out/ants
vol2atlas elastix   out/rigid -o out/elastix
vol2atlas brainreg  out/rigid -o out/brainreg

# apply rigid (+ affine + local refinements) to the FULL OME-Zarr at every pyramid level
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr

# Reapply a saved transform to a different channel / volume (no state.json needed)
vol2atlas apply-transform out/portable/transform.mat /data/other_channel.zarr \
                          --out /data/other_channel_in_ccf.zarr

# Share / version-control / regenerate landmarks externally
vol2atlas landmarks-export state.json -o landmarks.csv
vol2atlas landmarks-import state.json landmarks.csv

# Upgrade to a finer atlas WITHOUT losing landmarks
vol2atlas change-atlas     state.json --to allen_mouse_10um
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

## EM (Stage B) roadmap, not implemented

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

## Import / export landmarks

Landmarks can be round-tripped through external files in physical µm
(so the same file works at any atlas resolution). Two formats:

- **BigWarp CSV** (`.csv`): one row per pair, columns
  `Pt-i,True,sx,sy,sz,cx,cy,cz` — readable by BigWarp's File ›
  Import landmarks. Note the column order is `(x,y,z)` while
  vol2atlas tuples are `(z,y,x)`; the swap is handled on read/write.
- **vol2atlas JSON** (`.json`): `{"sample_um": [[z,y,x], ...],
  "ccf_um": [[z,y,x], ...]}` — straight dump of `state.landmarks`.

CLI:

```bash
vol2atlas landmarks-export state.json -o landmarks.csv      # or .json
vol2atlas landmarks-import state.json landmarks.csv          # mode=replace
vol2atlas landmarks-import state.json more.csv --mode append # merge
```

In the landmarks GUI: **Import landmarks…** and **Export landmarks…**
buttons (under "Clear ALL landmarks") open a file dialog. Importing
in the GUI always replaces the current set.

Use cases: share a labeled set with a collaborator, version-control
your picks, pre-pick in BigWarp / a custom tool, regenerate after a
`change-atlas` to a different reference brain (note: landmarks made
in one atlas's µm frame are NOT valid in a different brain's frame —
they survive Allen resolution changes but NOT cross-atlas switches).

## Live TPS preview in landmarks

In the landmarks GUI there's now a **"Live TPS preview"** checkbox with
a smoothing spinbox. When checked AND there are ≥4 landmark pairs, the
sample image layer renders with a thin-plate-spline correction layered
on top of the current affine baseline — same math `export --tps` uses,
but applied live on the crop preview. As you add landmarks / delete
outliers / refit, the TPS warp updates so you can see how a deformable
final would look before committing to one at export time.

The TPS is fit on **landmark residuals** (`sample_µm − M_base⁻¹ · ccf_µm`)
and applied as a per-voxel shift on top of the affine source. Far from
landmarks the shift → 0 and the affine warp is preserved; at landmark
positions the warped sample lands exactly on the CCF target.

Toggling the checkbox OFF restores the standard affine warp instantly.
Refit cost is ~ms for typical landmark counts (<100); rendering cost
scales with the crop voxel budget (`--preview-size`), capped at
~7M voxels by default.

The preview does not persist — it's a visualization only. To bake the
TPS into the output, run `vol2atlas export state.json -o out/tps
--tps` after Save.

## Local refinements (masked transforms)

Sometimes the global rigid+affine lands the whole brain correctly but
one region (e.g., one cortical lobe, a single subcortical structure)
is still slightly off. Re-fitting the global transform would help that
region but would degrade the rest. **Local refinements** apply an
extra affine *only inside a sphere* around chosen landmarks, with a
smooth sigmoid blend at the boundary so the rest of the volume is
unchanged.

```bash
# 1. Pick landmarks normally (vol2atlas landmarks ...). Pay attention
#    to which indices are in the region you want to refine — the
#    landmarks step prints them as you click:
#      [step3] + SAMPLE #5: world=... -> sample_µm=...

# 2. Add a local refinement using landmark indices 5,6,7,8,9 (≥4 pairs):
vol2atlas add-local-refinement state.json --name left_lobe \
        --landmarks 5,6,7,8,9 \
        --falloff-um 300 --radius-pad-um 200

# 3. Inspect what's set
vol2atlas list-local-refinements state.json

# 4. export / alignFull automatically compose any local refinements
#    on top of the global transform:
vol2atlas export state.json -o out/refined --write-transform
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr

# 5. Remove if you don't like it (no other state touched)
vol2atlas remove-local-refinement state.json --name left_lobe
```

**Math**: each local refinement defines a sphere in SAMPLE µm (center
+ radius derived from the chosen landmarks' centroid + max distance +
`--radius-pad-um`), and an extra affine fit ONLY from those
landmarks. At warp time, for each output voxel Q in atlas µm:

```
w(Q)              = sigmoid((radius - distance_to_center) / (falloff/4))
global_source(Q)  = M_global⁻¹ · Q
local_source(Q)   = M_local⁻¹  · Q
blended_source(Q) = (1 − w(Q)) · global_source + w(Q) · local_source
```

`w` ≈ 1 inside the sphere, ≈ 0 outside the sphere + falloff zone,
smooth in between. Multiple refinements stack (each one is evaluated
against the previously-blended source, so non-overlapping spheres
behave independently; overlapping spheres compose in sequence).

Local refinements travel with the portable transform: `export
--write-transform` writes them into the `transform.json` (under
`local_refinements`), and `vol2atlas apply-transform <transform.json>`
honors them when warping a different volume.

## Portable transform export

`state.json` carries the registration internally, but other tools
(ANTs, Slicer, BigWarp, custom Python scripts) don't read it. To emit
the transform as standalone files:

```bash
vol2atlas export state.json -o out/rigid --write-transform
# Adds to out/rigid/:
#   transform.mat        ITK GenericAffine (readable by antsApplyTransforms)
#   transform.json       OME-NGFF coordinateTransformations
#   landmarks.csv        sample/CCF pairs, BigWarp-format

# Reapply the saved transform to a different channel or acquisition:
vol2atlas apply-transform out/rigid/transform.mat /data/other_channel.zarr \
                          --out /data/other_channel_in_ccf.zarr \
                          --out-voxel-um 25,25,25
```

Both `.mat` and `.json` formats are accepted by `apply-transform`. The
`.json` includes atlas voxel size so `--out-voxel-um` is optional with
it; the `.mat` is voxel-info-free (ITK convention) so you must specify.

## Caveats

- Partial-hemisphere samples: use the rigid + TPS path. `brainreg` and
  plain `SyN` assume a full brain/hemisphere and will distort.
- `mi --affine` is a global 12-DOF refinement. If singular values
  print as 0.97–1.03 you've found a small real scale fix; if they
  diverge to e.g. 0.5 or 2.0 the optimizer over-fit — Revert in the
  preview and your `state.affine` stays unchanged.
- `alignFull` applies rigid + affine + local refinements when present;
  full-volume TPS / B-spline composition is not yet integrated (use the
  `out/tps` export or a downstream `ants` / `elastix` run for that).
- The CCF crop bbox from `prealign` is a HARD output frame for
  `export` and `alignFull` — sample voxels outside the crop are
  clipped. Expand the crop in `prealign` if you need more headroom.
  Interactive previews show the full sample so you can spot clipping.
- Prealign/refine UI exposes only **translation + rotation** (no
  axis-flip checkboxes). Rotation sliders run −180°…+180°, which
  covers any orientation that flips would have covered. If your
  sample is left-handed relative to the atlas (det < 0), `init
  --orientation` will refuse — pre-flip one axis of the sample data
  on disk first.
- Sample landmarks are stored in **raw sample µm** and reference the
  full transform (`state.affine ∘ state.transform`); they survive
  Allen-resolution switches (`change-atlas`) exactly. They do NOT
  survive a different reference brain (Princeton, Gubra, etc.) —
  different µm origin.

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
