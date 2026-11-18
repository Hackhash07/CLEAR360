#!/usr/bin/env bash
set -euo pipefail

FIX_COMMIT="eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d"
FIX_URL="https://github.com/cdcseacave/openMVS/commit/${FIX_COMMIT}"
DEST_ROOT="${OPENMVS_TEXTURE_FIX_DIR:-$HOME/Applications/OpenMVS-texture-fix-eeedab7}"
SOURCE_DIR="$DEST_ROOT/source"
BUILD_DIR="$DEST_ROOT/build"
BUILD_JOBS="${OPENMVS_BUILD_JOBS:-2}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this helper is intended for macOS." >&2
  exit 2
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: expected native Apple-Silicon arm64 shell; got $(uname -m)." >&2
  exit 2
fi
for tool in git cmake clang++ python3; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "ERROR: required tool not found: $tool" >&2
    exit 2
  }
done

if [[ -n "${VCPKG_ROOT:-}" ]]; then
  VCPKG_TOOLCHAIN="${VCPKG_ROOT%/}/scripts/buildsystems/vcpkg.cmake"
  if [[ ! -f "$VCPKG_TOOLCHAIN" ]]; then
    echo "ERROR: VCPKG_ROOT is set but its CMake toolchain is missing:" >&2
    echo "  $VCPKG_TOOLCHAIN" >&2
    exit 2
  fi
  echo "Using vcpkg toolchain: $VCPKG_TOOLCHAIN"
else
  echo "NOTICE: VCPKG_ROOT is not set."
  echo "CMake will rely on dependencies already installed in discoverable system paths."
  echo "If Boost, Eigen3, OpenCV, CGAL or VCG is missing, configuration will stop."
fi
if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: OPENMVS_BUILD_JOBS must be a positive integer." >&2
  exit 2
fi
if [[ -e "$DEST_ROOT" ]]; then
  echo "ERROR: refusing to overwrite existing build directory: $DEST_ROOT" >&2
  echo "Choose a new OPENMVS_TEXTURE_FIX_DIR or inspect/remove that folder yourself." >&2
  exit 2
fi

mkdir -p "$(dirname "$DEST_ROOT")"
echo "Building OpenMVS TextureMesh from upstream fix ${FIX_COMMIT}"
echo "Separate build root: $DEST_ROOT"
echo "Parallel jobs: $BUILD_JOBS"
echo "Existing OpenMVS installations will not be touched."

git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --detach "$FIX_COMMIT"
git -C "$SOURCE_DIR" submodule update --init --recursive
ACTUAL_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$FIX_COMMIT" ]]; then
  echo "ERROR: source checkout mismatch: $ACTUAL_COMMIT" >&2
  exit 3
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DOpenMVS_BUILD_TOOLS=ON \
  -DOpenMVS_BUILD_VIEWER=OFF \
  -DOpenMVS_USE_PYTHON=OFF \
  -DOpenMVS_ENABLE_TESTS=OFF

cmake --build "$BUILD_DIR" --target TextureMesh --parallel "$BUILD_JOBS"

TEXTURE_MESH="$(find "$BUILD_DIR" -type f -name TextureMesh -perm -111 -print -quit)"
if [[ -z "$TEXTURE_MESH" ]]; then
  echo "ERROR: build completed but executable TextureMesh was not found under $BUILD_DIR" >&2
  exit 4
fi

# Different builds may return either 0 or 1 for --help; preserve its output.
"$TEXTURE_MESH" --help > "$(dirname "$TEXTURE_MESH")/TextureMesh.help.txt" 2>&1 || true

python3 - "$TEXTURE_MESH" "$SOURCE_DIR" "$FIX_COMMIT" "$FIX_URL" "$BUILD_JOBS" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

binary = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
commit = sys.argv[3]
fix_url = sys.argv[4]
jobs = int(sys.argv[5])
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
metadata = {
    "source_repository": "https://github.com/cdcseacave/openMVS",
    "source_commit": commit,
    "source_commit_url": fix_url,
    "source_head_verified": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "binary_path": str(binary),
    "binary_sha256": digest,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "build_type": "Release",
    "build_jobs": jobs,
    "vcpkg_root": os.environ.get("VCPKG_ROOT") or None,
    "built_at_utc": datetime.now(timezone.utc).isoformat(),
}
metadata_path = binary.with_name("TextureMesh.build.json")
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Build provenance written to: {metadata_path}")
print(f"TextureMesh binary: {binary}")
print(f"TextureMesh SHA-256: {digest}")
PY

echo
echo "SUCCESS: pinned fixed TextureMesh binary built separately."
echo "Executable: $TEXTURE_MESH"
echo "Build tree retained at: $BUILD_DIR"
echo "No existing OpenMVS installation or reconstruction session was changed."
