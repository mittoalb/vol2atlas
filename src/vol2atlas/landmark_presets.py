"""Curated CCF-side landmark presets that ship with vol2atlas.

Each preset is a JSON file under `src/vol2atlas/data/ccf_landmark_presets/`
with one of these schemas:

  vol2atlas-native (no conversion):
    {
      "source_convention": "vol2atlas_zyx",
      "ccf_um_zyx": [[z0, y0, x0], ...]
    }

  Allen / Neuroglancer-source (identity to vol2atlas):
    {
      "source_convention": "allen_xyz",
      "ccf_um_xyz": [[x0, y0, z0], ...]
    }

Empirically verified for the Allen Mouse Brain CCFv3 as distributed via
brainglobe-atlasapi (`allen_mouse_*um`): although BrainGlobe reports the
orientation string as "asr", the loaded numpy array preserves Allen's
native voxel ordering — no axis flips are applied on load. So a point
captured in a Neuroglancer view of Allen's precomputed source
(`gs://allen_neuroglancer_ccf/average_template_10_8bit`) at (x, y, z)
maps to vol2atlas (numpy z, y, x) via simple identity:

    numpy_z = allen_x        # axis 0 (BrainGlobe's "Anterior+" label)
    numpy_y = allen_y        # axis 1
    numpy_x = allen_z        # axis 2

The asr orientation label is descriptive of display convention, not
data layout — confirmed by visual placement of test landmarks.
"""
from __future__ import annotations

import json
from pathlib import Path


_PRESETS_DIR = Path(__file__).parent / "data" / "ccf_landmark_presets"


def list_presets() -> list:
    """Return the names of all available presets (filename stems)."""
    if not _PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def _resolve_path(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    p = _PRESETS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no preset named {name!r} (looked in {_PRESETS_DIR} and as "
            f"a literal path). Available built-ins: {list_presets()}"
        )
    return p


def preset_metadata(name: str) -> dict:
    """Return the full preset dict (for showing description / count)."""
    return json.loads(_resolve_path(name).read_text())


def load_preset(name: str, ccf=None) -> list:
    """Load a CCF preset and return a list of (z, y, x) µm tuples in
    vol2atlas's BrainGlobe-loaded convention.

    `name` may be a built-in preset name OR a full path to a JSON file
    with the same schema.

    `ccf` is unused for the current conventions but accepted for forward
    compatibility (a future preset format may need atlas extents for
    axis flips).

    Raises ValueError on unknown source_convention; FileNotFoundError
    if the preset name doesn't resolve.
    """
    d = json.loads(_resolve_path(name).read_text())
    conv = d.get("source_convention", "vol2atlas_zyx")

    if conv == "vol2atlas_zyx":
        pts = d.get("ccf_um_zyx")
        if not pts:
            raise ValueError(
                f"preset {name!r} has source_convention='vol2atlas_zyx' "
                f"but no 'ccf_um_zyx' field")
        return [tuple(float(v) for v in pt) for pt in pts]

    if conv == "allen_xyz":
        pts = d.get("ccf_um_xyz")
        if not pts:
            raise ValueError(
                f"preset {name!r} has source_convention='allen_xyz' but "
                f"no 'ccf_um_xyz' field")
        # Identity mapping: (allen_x, allen_y, allen_z) → (z, y, x) =
        # (allen_x, allen_y, allen_z). See module docstring for the
        # empirical justification.
        return [(float(x), float(y), float(z)) for x, y, z in pts]

    raise ValueError(
        f"preset {name!r}: unknown source_convention {conv!r}. "
        f"Supported: 'vol2atlas_zyx', 'allen_xyz'.")
