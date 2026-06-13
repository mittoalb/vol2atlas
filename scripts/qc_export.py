"""Quick QC: load atlas + warped sample (+ sample mask) from an export
directory and overlay. Sample background is rendered transparent so it
doesn't wash over the CCF.

Usage:
    python scripts/qc_export.py [out_dir]
"""
import sys
from pathlib import Path
import nibabel as nib
import napari
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/export_rigid")
a = np.asarray(nib.load(str(out / "atlas_cropped.nii.gz")).dataobj)
s = np.asarray(nib.load(str(out / "sample_in_ccf.nii.gz")).dataobj).astype(np.float32)
mask_path = out / "sample_mask.nii.gz"
if mask_path.exists():
    m = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
    s[~m] = np.nan
    print(f"sample mask: {m.sum()/m.size*100:.1f}% of frame covered")
else:
    print("(no sample_mask.nii.gz — re-export to get the mask)")
print(f"atlas  {a.shape}  dtype={a.dtype}")
print(f"sample {s.shape}  dtype={s.dtype}")
assert a.shape == s.shape, "SHAPE MISMATCH — that's the bug"

v = napari.Viewer(ndisplay=2, title="qc_export")
v.add_image(a, name="atlas",  scale=(25, 25, 25),
            colormap="gray", blending="additive", opacity=0.5)
v.add_image(s, name="sample", scale=(25, 25, 25),
            colormap="gray", blending="additive", opacity=0.7)
napari.run()
