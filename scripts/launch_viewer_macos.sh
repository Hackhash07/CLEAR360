#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CLEAR360_CONDA_ENV:-clear360-py314}"

# Load Conda even when this script is started from a fresh Terminal window.
if ! command -v conda >/dev/null 2>&1; then
    for BASE in "${CONDA_BASE:-}" /opt/anaconda3 "$HOME/anaconda3" "$HOME/miniconda3" /opt/homebrew/Caskroom/miniforge/base; do
        if [[ -n "$BASE" && -f "$BASE/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "$BASE/etc/profile.d/conda.sh"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda was not found. Set CONDA_BASE or add conda to PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$ROOT"

COLMAP_BIN="$(command -v colmap || true)"
if [[ -z "$COLMAP_BIN" ]]; then
    echo "ERROR: COLMAP is not available in Conda environment '$ENV_NAME'." >&2
    exit 1
fi

export CLEAR360_COLMAP_BIN="$COLMAP_BIN"
# When CLEAR360_OPENMVS_DIR is not explicitly set, CLEAR360 now automatically
# selects ~/Applications/CLEAR360-OpenMVS-fixed-texture if all five tools exist.
echo "CLEAR360 source: $ROOT"
echo "Python: $(command -v python)"
echo "COLMAP: $CLEAR360_COLMAP_BIN"
echo "OpenMVS: ${CLEAR360_OPENMVS_DIR:-automatic fixed-overlay discovery}"

exec python -m viewer.app "$@"
