"""Per-project state file. Each pipeline step reads it and writes it back.

Schema (all fields optional except sample_zarr + atlas_name + sample_level):

    {
      "sample_zarr": "/abs/path/to/sample.zarr",
      "sample_level": 2,
      "sample_voxel_um": [2.74, 2.74, 2.74] | null,
      "sample_channel": 0,
      "atlas_name": "allen_mouse_25um",
      "ccf_crop_bbox": {"z":[z0,z1], "y":[y0,y1], "x":[x0,x1]} | null,
      "transform": { ...RigidTransform.to_dict() ... } | null,
      "affine":    [[a, b, c, d], [e, f, g, h], [i, j, k, l], [0, 0, 0, 1]] | null,
                   # 4x4 in CCF µm space, applied AFTER `transform` at
                   # export/alignFull time. Output from `vol2atlas mi --affine`.
      "landmarks": {
          "sample_um": [[z,y,x], ...],
          "ccf_um":    [[z,y,x], ...]
      },
      "history": [
        {"step": "init",     "when": "2026-05-16T16:01:00", "note": "..."},
        {"step": "step1",    "when": "2026-05-16T16:42:00", "note": "..."},
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class State:
    sample_zarr: str
    sample_level: int
    atlas_name: str
    sample_voxel_um: Optional[list] = None
    sample_channel: int = 0
    # 3-letter BrainGlobe orientation code for the sample (e.g. "asr",
    # "psr"). Optional; if set, init pre-populates state.transform with
    # the rotation that maps this orientation to the atlas's orientation
    # so prealign opens with the sample already roughly aligned.
    sample_orientation: Optional[str] = None
    ccf_crop_bbox: Optional[dict] = None
    transform: Optional[dict] = None
    affine: Optional[list] = None        # 4x4 in CCF µm, composed after `transform`
    landmarks: dict = field(default_factory=lambda: {"sample_um": [], "ccf_um": []})
    history: list = field(default_factory=list)

    def add_history(self, step: str, note: str = ""):
        self.history.append({
            "step": step,
            "when": datetime.now().isoformat(timespec="seconds"),
            "note": note,
        })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        # Tolerate older state files missing newer fields.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def save(state: State, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    tmp.replace(path)


def load(path: Path) -> State:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no state file at {path}")
    return State.from_dict(json.loads(path.read_text()))
