#!/usr/bin/env bash
# ============================================================
# VOL2ATLAS TEST SUITE — exercises every subcommand
# ============================================================
# Run sections individually OR copy-paste the whole script.
# Headless tests just need a 0 exit code + sensible output.
# Interactive tests open a napari window — clearly marked "GUI".
# Set ZARR + OUT below to match your environment before running.
# ============================================================

set -u
ZARR=${ZARR:-/local_data/ESRF/715697L_4.349um.zarr}
OUT=${OUT:-/tmp/vol2atlas_test_out}
mkdir -p "$OUT" && cd "$OUT"

run() { echo ">>> $*"; "$@"; rc=$?; [ $rc -ne 0 ] && echo "!!! FAILED ($rc)" >&2; return $rc; }

# ============================================================
# 0. DISCOVERY — no state.json needed
# ============================================================
run vol2atlas --help                                    # top-level help
run vol2atlas help-all                                  # full --help per subcommand
run vol2atlas info "$ZARR"                              # OME-Zarr pyramid summary
run vol2atlas list-atlases                              # mouse atlases (filtered)
run vol2atlas list-atlases --downloaded                 # cached only
run vol2atlas list-atlases --species ""                 # all species
run vol2atlas landmarks-list-presets                    # ships at least default_v1 (49 pts)

# ============================================================
# 1. INIT — creates state.json
# ============================================================
rm -f state.json; rm -rf state.json.d/
run vol2atlas init "$ZARR" --level 2 --atlas allen_mouse_25um --orientation pir
# Expect: prints orientation rotation tuple (rz/ry/rx).
# Verify: state.json exists; sample_orientation, sample_level, atlas_name populated.

# ============================================================
# 2. ATLAS SWITCHING — verify physical-µm landmarks survive
# ============================================================
run vol2atlas change-atlas state.json --to allen_mouse_50um
# Expect: prints ccf_crop_bbox rescaled (if one set) + "atlas → ..."
run vol2atlas change-atlas state.json --to allen_mouse_25um   # back to default

# ============================================================
# 3. PREALIGN — GUI, sets state.transform + ccf_crop_bbox
# ============================================================
# GUI: drag sliders OR re-run with --orientation. Draw a crop bbox.
# When done: click "Save state.json && exit"
run vol2atlas prealign state.json
# Verify: state.json now has transform + ccf_crop_bbox.

# Alternative: override orientation without wiping state
# run vol2atlas prealign state.json --orientation psl

# ============================================================
# 4. REFINE — GUI, fine-tunes state.transform
# ============================================================
run vol2atlas refine state.json --level 2 --preview-size 600
# GUI: ortho ±3 mm / ±30° sliders. Save && exit.

# Higher-res preview
# run vol2atlas refine state.json --level 1 --preview-size 1000

# ============================================================
# 5. LANDMARKS — GUI; tests every panel
# ============================================================
# 5a. Load curated CCF preset (replace, fresh start)
run vol2atlas landmarks-load-preset state.json --preset default_v1 --mode replace
# Expect: "replaced CCF landmarks with 49 from preset 'default_v1'"

# 5b. Open GUI
run vol2atlas landmarks state.json --level 1 --preview-size 1000
# In GUI, test:
#   - Click each CCF list row → viewer jumps to that landmark
#   - Pick a sample landmark for ≥4 pairs in order
#   - "Fit RIGID from landmarks" → preview updates
#   - "Fit AFFINE from landmarks (≥4 pairs)" → preview updates
#   - Per-pair RMS in terminal; row with high RMS → "Delete selected" → re-fit
#   - "Live TPS preview" checkbox → preview re-renders with TPS shift
#       + smoothing spinbox → re-renders
#   - Multi-select 4+ rows, type "test_lr", click "Add local refinement from
#       selected" → preview shows masked correction
#   - Select "test_lr" → "Remove selected local refinement"
#   - "Revert last fit" → undoes most recent Fit
#   - "Clear ALL landmarks" → both lists empty, preview reverts
#   - Re-load preset via "CCF preset" dropdown
#   - "Save state.json (keep window open)" → state.json mtime updates
#   - "Save state.json && exit"

# 5c. Landmark I/O round-trip
run vol2atlas landmarks-export state.json -o lms.csv      # BigWarp format
run vol2atlas landmarks-export state.json -o lms.json     # vol2atlas JSON
run vol2atlas landmarks-import state.json lms.csv --mode replace
run vol2atlas landmarks-import state.json lms.json --mode append
# Verify: state.json's landmarks count grew

# ============================================================
# 6. MI / JOINT — automated registration
# ============================================================
# 6a. Inspect brain mask only (no register)
run vol2atlas mi state.json --check-mask
# GUI shows masks; close. Nothing written.

# 6b. Rigid intensity MI
run vol2atlas mi state.json --skip-view

# 6c. Affine intensity MI
run vol2atlas mi state.json --affine --skip-view

# 6d. Affine on SDT (shape) — partial-hemisphere friendly
run vol2atlas mi state.json --affine --shape --skip-view
# Terminal prints singular values (sane: 0.9–1.1) and translation

# 6e. JOINT (run via GUI button in landmarks):
# Re-open landmarks → "Run JOINT MI + landmarks (single opt)"
# Then "Run JOINT iterative (block-coord descent)"
# Both run in-memory; click Save to persist.

# ============================================================
# 7. LOCAL REFINEMENTS (CLI)
# ============================================================
# Use the LAST 4 landmark indices
N=$(python -c "import json; print(len(json.load(open('state.json'))['landmarks']['sample_um']))")
echo "have $N landmark pairs"
IDX="$((N-4)),$((N-3)),$((N-2)),$((N-1))"
run vol2atlas add-local-refinement state.json --name test_region \
    --landmarks $IDX --falloff-um 300 --radius-pad-um 200
run vol2atlas list-local-refinements state.json
# Expect: [0] test_region: center=(…) radius=… fall=… landmarks=[…]

# ============================================================
# 8. EXPORT
# ============================================================
run vol2atlas export state.json -o "$OUT/rigid"                       # warped sample + cropped atlas
run vol2atlas export state.json -o "$OUT/tps" --tps                   # + TPS on landmark residuals
run vol2atlas export state.json -o "$OUT/portable" --write-transform  # + transform.mat / .json / landmarks.csv
ls -la "$OUT/portable/"                                                # verify .mat .json .csv all present

# ============================================================
# 9. APPLY-TRANSFORM — re-warp via the portable transform alone
# ============================================================
run vol2atlas apply-transform "$OUT/portable/transform.mat" "$ZARR" \
    --out "$OUT/applied_from_mat.zarr" --out-voxel-um 25,25,25
run vol2atlas apply-transform "$OUT/portable/transform.json" "$ZARR" \
    --out "$OUT/applied_from_json.zarr"
# Compare $OUT/applied_from_mat.zarr against $OUT/rigid/sample_in_ccf.zarr —
# should be visually identical (same affine math).

# ============================================================
# 10. REMOVE LOCAL REFINEMENT — verify clean removal
# ============================================================
run vol2atlas remove-local-refinement state.json --name test_region
run vol2atlas list-local-refinements state.json                       # "(no local refinements)"

# ============================================================
# 11. ALIGNFULL — chunked, scales to TB
# ============================================================
# Bottom of pyramid for fast test (level 4 is small)
run vol2atlas alignFull state.json -o "$OUT/full_l4.zarr" --levels 4
# Multi-level — production output
# run vol2atlas alignFull state.json -o "$OUT/full_all.zarr"

# ============================================================
# 12. DEFORMABLE BACKENDS — optional, need extras installed
# ============================================================
# pip install -e '.[ants]'
run vol2atlas ants "$OUT/rigid" -o "$OUT/ants"
# pip install -e '.[elastix]'
run vol2atlas elastix "$OUT/rigid" -o "$OUT/elastix"
# pip install -e '.[brainreg]'
# Whole-brain / whole-hemisphere only — skips for partial:
# run vol2atlas brainreg "$OUT/rigid" -o "$OUT/brainreg"

# ============================================================
# 13. QC scripts
# ============================================================
SCRIPT_DIR="$(dirname "$0")"
run python "$SCRIPT_DIR/qc_export.py" "$OUT/rigid"
run python "$SCRIPT_DIR/qc_compare_methods.py" \
    "$OUT/rigid" "$OUT/tps" "$OUT/ants" "$OUT/elastix"

# ============================================================
# What to look for at each stage (red flags):
#   * init / atlas switching: state.json always reloads cleanly
#       python -c 'import json; json.load(open("state.json"))'
#       should never fail.
#   * prealign / refine / landmarks: sliders responsive; volume visibly
#       aligns; closing without Save does NOT modify state.json mtime.
#   * mi: terminal prints singular values 0.9–1.1 for sane fits;
#       >2 or <0.5 = optimizer blew up → click Revert in preview.
#   * TPS preview: <4 pairs = silently no-op; >=4 = preview deforms;
#       higher smoothing = less aggressive correction.
#   * Local refinement: preview updates instantly on Add / Remove;
#       outside the sphere should look identical to before.
#   * Export / alignFull: warped output's anatomy lines up with the
#       atlas in napari; sample_mask.nii.gz nonzero only inside sample.
#   * apply-transform: applied_from_mat.zarr should match
#       rigid/sample_in_ccf.zarr voxel-for-voxel (interp tolerance ok).
#
# To capture an error: command 2>&1 | tee /tmp/error.log
# Then paste the tail of /tmp/error.log for diagnosis.
#
# Intentionally NOT tested (out of scope / removed):
#   * segment      (removed)
#   * antspynet    (removed)
#   * EM Stage B   (roadmap only)
# ============================================================
