# OpenMVS 2.4.0 texture-atlas corruption: isolated fix and A/B procedure

## Why this experiment exists

The user's accepted six-photo baseline is CLEAR360 session `session_049`, produced by the restored source snapshot `1d7d7bcc088e753b9db8a24d9fc3ff05229ec9b2`. Its OpenMVS external-stage sum was 328.8 seconds (5m28.8s). **Do not change the geometry-producing pipeline or quality profile while testing this colour fix.**

OpenMVS issue [#1251](https://github.com/cdcseacave/openMVS/issues/1251) reports black texture-patch edges and random-colour patch borders on v2.4.0. The identified sampler type mismatch was corrected upstream in commit [`eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`](https://github.com/cdcseacave/openMVS/commit/eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d). That commit updates `libs/Common/Types.inl` and seam-leveling code in `libs/MVS/SceneTexture.cpp`; the corrected `Types.inl` call remains present in upstream `develop`.

## Build a separate fixed TextureMesh binary on Apple Silicon

Do not overwrite `~/Applications/OpenMVS-2.4.0`. OpenMVS's build guide lists CMake, Git and a C++17-capable compiler; it documents vcpkg as its dependency-management route. See [OpenMVS Building](https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Building.md).

### Dependency check before cloning/building

The pinned OpenMVS CMake configuration automatically loads vcpkg only when
`VCPKG_ROOT` is set. Your latest terminal output reported
`VCPKG_ROOT=not-set`. That is not automatically fatal: CMake can use libraries
already installed in standard system prefixes, but it will not automatically
fetch the manifest dependencies without the vcpkg toolchain.

The TextureMesh target still requires Boost, Eigen 3.4+, OpenCV, CGAL and VCG.
Before running the build helper, check whether an existing vcpkg installation
is available and whether common Homebrew dependencies are installed:

```bash
command -v vcpkg || true
for root in "$HOME/vcpkg" /opt/homebrew/opt/vcpkg /opt/homebrew/share/vcpkg \
            /usr/local/vcpkg /usr/local/share/vcpkg; do
  if [[ -f "$root/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "VCPKG_ROOT candidate: $root"
  fi
done
if command -v brew >/dev/null 2>&1; then
  brew list --versions boost eigen opencv cgal libomp 2>/dev/null || true
fi
```

If a valid vcpkg root is found, set `VCPKG_ROOT` to that directory in the
current terminal only. Do not install or upgrade dependencies yet. If no vcpkg
root exists, send the dependency-check output before proceeding; we can choose
between the already-installed system libraries and an isolated dependency
setup without touching the existing OpenMVS installation.

The build helper now validates a supplied `VCPKG_ROOT` and records it in
`TextureMesh.build.json`. If it is unset, it prints a warning before CMake
configuration rather than pretending dependency auto-installation is enabled.

First inspect available tools and free disk space:

```bash
cmake --version
git --version
clang++ --version
xcode-select -p
df -h "$HOME"
echo "VCPKG_ROOT=${VCPKG_ROOT:-not-set}"
```

Then run the dedicated isolated build helper from the CLEAR360 repository root:

```bash
OPENMVS_BUILD_JOBS=2 bash scripts/build_openmvs_texture_fix_macos.sh
```

The helper:
- refuses to overwrite an existing build directory;
- clones OpenMVS and checks out the exact fix commit;
- builds only the `TextureMesh` target in Release mode for `arm64`;
- disables the optional Viewer, Python bindings and tests to keep the build focused;
- uses the existing `VCPKG_ROOT` automatically when configured by OpenMVS CMake;
- verifies the executable contains an `arm64` slice and can run `--help` (catching missing linked libraries before it is trusted);
- keeps the source/build tree and creates an adjacent `TextureMesh.build.json` with the exact source commit, binary architecture, vcpkg root (if used), and binary SHA-256.

The build is limited to two parallel jobs by default for the MacBook Air. Set `OPENMVS_BUILD_JOBS` to another positive number only after checking available RAM. If CMake reports missing dependencies, stop and inspect the existing OpenMVS/vcpkg dependency setup rather than changing the working installation. No build folder or dependency cache is automatically deleted.

The helper prints the exact binary path and writes `TextureMesh.build.json` next to it. Discover the path and verify the metadata before the dry-run:

```bash
TEXTURE_MESH="$(find "$HOME/Applications/OpenMVS-texture-fix-eeedab7/build" -type f -name TextureMesh -perm -111 -print -quit)"
"$TEXTURE_MESH" --help | head -40
ls -l "$(dirname "$TEXTURE_MESH")/TextureMesh.build.json"
```

The re-texturing utility rejects a binary without this adjacent metadata or whose SHA-256 no longer matches it. This prevents accidentally re-running the old 2026-01-19 v2.4.0 executable.

## Preflight the saved session

The original OpenMVS backend writes its ReconstructMesh scene and PLY here:

- `session_049/openmvs/scene_mesh.mvs`
- `session_049/openmvs/scene_mesh.ply`
- Undistorted views: `session_049/openmvs/dense/images/`

The utility also fingerprints every file in the existing `session_049/mesh/` package. If the saved scene, PLY, image folder or baseline mesh package is missing, stop. Do not rerun COLMAP, DensifyPointCloud, fusion or ReconstructMesh as an automatic fallback.

## Dry-run first (no files are created)

From the CLEAR360 repository root:

```bash
TEXTURE_MESH="$(find "$HOME/Applications/OpenMVS-texture-fix-eeedab7/build" -type f -name TextureMesh -perm -111 -print -quit)"

python scripts/retexture_openmvs_session.py \
  --session data/phone_reconstruction/session_049 \
  --texture-mesh "$TEXTURE_MESH"
```

The utility refuses missing inputs, an unknown/non-executable binary, a binary without matching pinned-build provenance, an existing output folder or any output path that is not a fresh direct child of the session. The dry-run prints the exact command; it does not create files.

## First A/B run: change only the TextureMesh executable

Only after the dry-run paths and binary are checked, execute:

```bash
python scripts/retexture_openmvs_session.py \
  --session data/phone_reconstruction/session_049 \
  --texture-mesh "$TEXTURE_MESH" \
  --output-dir texture_ab_fixed \
  --max-texture-size 2048 \
  --virtual-face-images 1 \
  --local-seam-leveling 1 \
  --global-seam-leveling 0 \
  --run
```

This reproduces the session-049 TextureMesh flags. The **only intended A/B variable is the fixed TextureMesh executable**. The tool reads the original run's `texture_threads` value from `openmvs_pipeline.log` (falling back to two threads only if the old log has no configuration record), calls TextureMesh against `openmvs/scene_mesh.mvs` from the original `openmvs/` working directory and writes a new OBJ/MTL/atlas package to `session_049/texture_ab_fixed/`. It does not run geometry stages or replace the baseline `mesh/` package.

A JSON manifest records the command, executable hash and pinned source metadata, input scene/PLY/image/baseline-mesh SHA-256 hashes before and after, image list, output maps, exit code and elapsed time. A successful manifest requires all source hashes to remain unchanged and every diffuse map referenced by the new MTL to exist. The manifest explicitly records `geometry_stages_run: []`.

## Visual comparison and optional second experiment

Open the generated atlas JPG(s) alongside the original `session_049/mesh/scene_textured_material_*_map_Kd.jpg` files. Compare the same views of the fountain and brick wall. The first run is successful only if black patch-edge corruption and saturated random borders disappear while the photographic detail is retained.

If the first run removes corruption but seam blending is still poor, make a second output folder such as `texture_ab_fixed_global/` and change **only** `--global-seam-leveling 1`. Keep local seam leveling at 1 and all remaining values identical. Do not change `--ignore-mask-label=-1`; issue #1251 reports that `-2` caused a crash in the affected build.

Do not replace `session_049/mesh/` until the user has approved the atlas and model visually. The saved original remains the rollback version.
