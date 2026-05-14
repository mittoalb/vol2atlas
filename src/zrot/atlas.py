"""Allen CCF reference loader (via brainglobe-atlasapi)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CCFReference:
    reference: np.ndarray            # (z, y, x) intensity volume
    annotation: np.ndarray | None    # (z, y, x) region IDs, optional
    voxel_um: tuple[float, float, float]   # spacing in micrometers
    orientation: str                 # e.g. "asr" — anatomical axes order
    name: str


def load_ccf(
    atlas_name: str = "allen_mouse_25um",
    with_annotation: bool = False,
) -> CCFReference:
    """Download (cached) and load an Allen-style atlas from BrainGlobe.

    Common choices:
      - "allen_mouse_100um"  (small, fast — good for first alignment)
      - "allen_mouse_50um"
      - "allen_mouse_25um"   (default — matches ~5th level of typical light-sheet pyramids)
      - "allen_mouse_10um"   (large, only if you need fine alignment)
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(atlas_name)
    ref = np.asarray(atlas.reference)
    ann = np.asarray(atlas.annotation) if with_annotation else None
    res = tuple(float(r) for r in atlas.resolution)  # (z, y, x) µm
    return CCFReference(
        reference=ref,
        annotation=ann,
        voxel_um=res,  # type: ignore[arg-type]
        orientation=str(atlas.orientation),
        name=atlas_name,
    )
