"""Compare several method outputs in one napari viewer.

Pass any number of directories that each contain a
`sample_in_ccf.nii.gz` (as produced by `vol2atlas export`, `vol2atlas ants`,
`vol2atlas brainreg`). The first dir's `atlas_cropped.nii.gz` is used as
the gray reference; each sample is added as its own toggleable layer.

Usage:
    python scripts/qc_compare_methods.py out/rigid out/tps out/ants out/brainreg
"""
import sys
from pathlib import Path

import nibabel as nib
import napari
import numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: qc_compare_methods.py <dir1> [<dir2> ...]")

dirs   = [Path(a) for a in sys.argv[1:]]
colors = ["magenta", "cyan", "yellow", "green", "red", "blue"]

# Atlas comes from the first dir (all dirs share the same cropped CCF).
atlas_path = dirs[0] / "atlas_cropped.nii.gz"
atlas      = np.asarray(nib.load(str(atlas_path)).dataobj)
voxel_um   = tuple(nib.load(str(atlas_path)).header.get_zooms()[:3])
voxel_um   = tuple(float(v) * 1000.0 for v in voxel_um)  # mm → µm
print(f"atlas       {atlas.shape}  voxel {voxel_um} µm  (from {dirs[0].name})")

v = napari.Viewer(ndisplay=2, title="vol2atlas — method comparison")
v.add_image(atlas, name="ATLAS", scale=voxel_um,
            colormap="gray", blending="additive", opacity=0.5)

# One image layer per method, only the first turned on by default —
# toggle the eye icons in the layer list to flip between them.
for i, d in enumerate(dirs):
    f = d / "sample_in_ccf.nii.gz"
    if not f.exists():
        print(f"  skip {d}: no sample_in_ccf.nii.gz"); continue
    arr = np.asarray(nib.load(str(f)).dataobj)
    if arr.shape != atlas.shape:
        print(f"  WARNING {d.name}: shape {arr.shape} != atlas {atlas.shape}")
    name  = d.name.replace("_", " ")
    color = colors[i % len(colors)]
    print(f"  {name:<20} {arr.shape}  layer color {color}")
    v.add_image(arr, name=name, scale=voxel_um,
                colormap=color, blending="additive",
                opacity=0.6, visible=(i == 0))

print("\nToggle layers in the napari panel to compare methods.")
napari.run()
