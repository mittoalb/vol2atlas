#!/usr/bin/env python3
"""Open a BrainGlobe atlas in napari with the parcellation overlaid.

Hover over the parcellation layer to see id, acronym, and name in the
bottom status bar.

Usage:
    python view_atlas.py
    python view_atlas.py --atlas allen_mouse_50um --ndisplay 3
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="allen_mouse_25um",
                    help="BrainGlobe atlas name (default: allen_mouse_25um).")
    ap.add_argument("--ndisplay", type=int, default=2,
                    help="2 (slice view) or 3 (volume MIP). Default: 2.")
    ap.add_argument("--opacity", type=float, default=0.5,
                    help="Parcellation overlay opacity (default: 0.5).")
    args = ap.parse_args()

    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    print(f"loading {args.atlas}...")
    atlas = BrainGlobeAtlas(args.atlas)
    print(f"  reference shape: {atlas.reference.shape}  voxel: {atlas.resolution} µm")
    print(f"  {len(atlas.structures)} regions")

    v = napari.Viewer(ndisplay=args.ndisplay)
    v.add_image(atlas.reference, name="CCF template",
                colormap="gray", scale=atlas.resolution)
    labels = v.add_labels(atlas.annotation, name="regions",
                          scale=atlas.resolution, opacity=args.opacity)

    # ------------------------------------------------------------------
    # Hover: show id/acronym/name in status bar.
    # ------------------------------------------------------------------
    @labels.mouse_move_callbacks.append
    def _show_region(layer, event):
        val = layer.get_value(event.position, world=True)
        if val is None or val == 0:
            v.status = ""
            return
        s = atlas.structures.get(int(val))
        v.status = (f'{int(val)}  {s["acronym"]:>10s}  —  {s["name"]}'
                    if s else f"{int(val)} (unknown)")

    # ------------------------------------------------------------------
    # Side panel: searchable list of all regions.
    #   - Type in the search box to filter (matches name OR acronym).
    #   - Click a region in the list to jump napari to its centroid AND
    #     highlight only that region (others dimmed).
    #   - Click "Show all regions" to restore the full parcellation.
    # ------------------------------------------------------------------
    import numpy as np
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QListWidget,
                                QListWidgetItem, QPushButton, QLabel)

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(4, 4, 4, 4)

    title = QLabel(f"<b>{args.atlas}</b><br>{len(atlas.structures)} regions")
    layout.addWidget(title)

    search = QLineEdit()
    search.setPlaceholderText("filter (name or acronym)…")
    layout.addWidget(search)

    list_widget = QListWidget()
    layout.addWidget(list_widget)

    show_all_btn = QPushButton("Show all regions")
    layout.addWidget(show_all_btn)

    # Populate sorted by acronym
    structs = sorted(atlas.structures.values(), key=lambda s: s["acronym"])

    def _populate(query: str = ""):
        list_widget.clear()
        q = query.lower().strip()
        for s in structs:
            if q and q not in s["acronym"].lower() and q not in s["name"].lower():
                continue
            it = QListWidgetItem(f'{s["acronym"]:>10s}   {s["name"]}')
            it.setData(0x0100, s["id"])  # Qt.UserRole = 256
            list_widget.addItem(it)
    _populate()
    search.textChanged.connect(_populate)

    # Cache annotation as ndarray for centroid + masking
    ann = np.asarray(atlas.annotation)
    original_data = ann   # keep reference for restore

    def _on_click(item):
        rid = item.data(0x0100)
        s = atlas.structures.get(int(rid))
        if not s:
            return
        # Find all voxels with this ID OR any descendant ID
        try:
            descendants = atlas.get_structure_descendants(s["acronym"])
            ids = {rid} | {atlas.structures[d]["id"] for d in descendants}
        except Exception:
            ids = {rid}
        mask = np.isin(ann, list(ids))
        if not mask.any():
            v.status = f'no voxels for {s["acronym"]}'
            return
        # Highlight: set everything else to 0
        labels.data = np.where(mask, ann, 0)
        # Jump to centroid (in voxel index → world)
        idx = np.array(np.where(mask)).mean(axis=1)
        world = idx * np.asarray(atlas.resolution)
        v.dims.set_point(0, world[0])   # z-slice
        v.camera.center = tuple(world)
        v.status = (f'{rid}  {s["acronym"]} — {s["name"]}  '
                    f'({mask.sum():,} voxels)')

    list_widget.itemClicked.connect(_on_click)

    def _restore_all():
        labels.data = original_data
        v.status = "all regions visible"
    show_all_btn.clicked.connect(_restore_all)

    v.window.add_dock_widget(panel, name="regions", area="right")

    print("napari open. tips:")
    print("  - hover any region → status bar shows id / acronym / name")
    print("  - 'regions' panel (right): type to filter, click to isolate + jump")
    print("  - 'Show all regions' to restore the full parcellation")
    napari.run()


if __name__ == "__main__":
    main()
