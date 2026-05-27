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
vol2atlas init  /data/sample.zarr  --level 2  --voxel-um 2.74
vol2atlas prealign  state.json     # 3D rough alignment + CCF crop
vol2atlas refine    state.json     # fine refine in ortho views
vol2atlas landmarks state.json     # pick landmark pairs (optional)

# downsampled QC output
vol2atlas export    state.json -o out/rigid
vol2atlas export    state.json -o out/tps --tps     # rigid + TPS on residuals

# optional deformable refinement on top of an export
vol2atlas ants      out/rigid -o out/ants
vol2atlas elastix   out/rigid -o out/elastix
vol2atlas brainreg  out/rigid -o out/brainreg

# apply rigid transform to the FULL OME-Zarr at every pyramid level
vol2atlas alignFull state.json -o /data/big/sample_in_ccf.zarr
```

Each refinement command writes the same layout under `out/<dir>/`:

```
sample_in_ccf.nii.gz     warped sample in atlas space
atlas_cropped.nii.gz     CCF reference, same grid
sample_in_ccf.zarr/      multiscale OME-Zarr (5 levels)
atlas_cropped.zarr/      multiscale OME-Zarr (5 levels)
```

Full help for every subcommand (args, options, defaults):

```bash
vol2atlas help-all
```

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

## Caveats

- Partial-hemisphere samples: use the rigid + TPS path. `brainreg` and
  plain `SyN` assume a full brain/hemisphere and will distort.
- `landmarks` Fit button **replaces** the saved rigid pose. Skip Fit
  if your hand alignment from prealign/refine already looks right.
- `alignFull` is currently rigid-only.

## References

- BrainGlobe atlasapi — https://brainglobe.info/documentation/brainglobe-atlasapi/
- OME-Zarr spec — https://ngff.openmicroscopy.org/0.4/
- Allen CCFv3 — https://atlas.brain-map.org/
- See [methods_survey.pptx](methods_survey.pptx) for a registration-methods literature overview.
- See [vol2atlas_strategy.pptx](vol2atlas_strategy.pptx) for the multi-modal scaling strategy.
