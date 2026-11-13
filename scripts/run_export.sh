#!/usr/bin/env bash
# CLEAR360 — Run Blender headless export for M1 simulation frames.
# Usage: bash scripts/run_export.sh [--frames N] [--output DIR]
#
# Requires Blender 4.x to be on PATH or set BLENDER_PATH env var.
# Example with custom Blender path:
#   BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender bash scripts/run_export.sh

set -euo pipefail

# ── Blender binary ────────────────────────────────────────────────────────────
BLENDER="${BLENDER_PATH:-blender}"

# ── Defaults ─────────────────────────────────────────────────────────────────
FRAMES=20
OUTPUT="."

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash scripts/run_export.sh [--frames N] [--output DIR]" >&2
            exit 1
            ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================"
echo "  CLEAR360 M1 Blender Export"
echo "========================================================"
echo "  Blender  : $BLENDER"
echo "  Frames   : $FRAMES"
echo "  Output   : $OUTPUT"
echo "  Script   : $PROJECT_ROOT/sim/export_frames.py"
echo "========================================================"

# Verify blender is accessible
if ! command -v "$BLENDER" &>/dev/null 2>&1; then
    # Try the literal path in case it's not a bare command
    if [[ ! -x "$BLENDER" ]]; then
        echo "ERROR: Blender not found at '$BLENDER'." >&2
        echo "       Install Blender 4.x and ensure it is on PATH, or set BLENDER_PATH." >&2
        exit 1
    fi
fi

# ── Run Blender ───────────────────────────────────────────────────────────────
"$BLENDER" --background \
    --python "$PROJECT_ROOT/sim/export_frames.py" \
    -- \
    --output "$OUTPUT" \
    --frames "$FRAMES"

BLENDER_EXIT=$?

echo ""
echo "========================================================"
echo "  Export complete. Data directory sizes:"
echo "========================================================"

# Print directory sizes of data/ subdirs (non-fatal if data/ doesn't exist yet)
if [[ -d "$OUTPUT/data" ]]; then
    du -sh "$OUTPUT/data/sim_frames"   2>/dev/null || echo "  sim_frames/   : (empty)"
    du -sh "$OUTPUT/data/gt_depth"     2>/dev/null || echo "  gt_depth/     : (empty)"
    du -sh "$OUTPUT/data/gt_position"  2>/dev/null || echo "  gt_position/  : (empty)"
    du -sh "$OUTPUT/data/gt_semantic"  2>/dev/null || echo "  gt_semantic/  : (empty)"
else
    echo "  (no data/ directory found at $OUTPUT/data)"
fi

echo "========================================================"

exit $BLENDER_EXIT
