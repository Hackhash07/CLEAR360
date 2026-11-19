# CLEAR360 — Photo-to-3D Reconstruction & 360° Perception

> **Status — M7.2 & M7.3 COMPLETE**:
> - **M7.2**: Photorealistic 3D Automotive Surround Reconstruction Viewer with true UV composite texture atlas, visibility-aware overlap triangle culling, detailed 2022 Hyundai Creta SUV model, and distinct Driver vs. Engineering views.
> - **M7.3**: Smartphone Photograph to 3D World Reconstruction via PyCOLMAP / OpenCV Structure-from-Motion, dense multi-view stereo, surface meshing, texture generation, and optional metric scale calibration.

---

## 1. Project Overview

**CLEAR360** is a photo-to-3D reconstruction and 360° perception platform with two complementary capabilities:
1. **General Photo-to-3D Reconstruction**: Accepts a collection of photographs of the same static object or environment, estimates camera poses, reconstructs dense geometry, generates a textured surface, and exports reusable 3D assets.
2. **Automotive Surround Perception**: Provides the automotive 360° viewer and vehicle-centred perception tools as a specialized application mode.

Metric scale is available only when a trustworthy reference measurement or calibrated camera setup is supplied. Photogrammetry does not guarantee a watertight solid: open surfaces, missing regions and disconnected components are reported for inspection rather than silently filled or deleted.

---

## 2. Desktop Application Architecture (3 Modes)

Launch the interactive PySide6 + VisPy 3D desktop application:
```bash
python -m viewer.app
```

The application provides three operating modes via the top mode selector bar:

- **MODE 1: Automotive Surround**
  - **Driver View (`D`)**: Clean, photorealistic cockpit perspective displaying the textured 3D ground, walls, and surroundings without engineering clutter.
  - **Surface View**: Continuous surface mesh with photographic UV texture mapping.
- **MODE 2: Phone Photo 3D World**
  - Ingests photographs (`.jpg`, `.jpeg`, `.png`, `.heic`, and supported image formats) of one static scene.
  - Audits blur, exposure, feature density, exact duplicates, camera/lens metadata and image dimensions before reconstruction.
  - Estimates camera poses and sparse 3D structure with incremental Structure-from-Motion.
  - Reconstructs dense geometry, meshes and photographic textures using the selected backend.
  - Supports optional metric reference-distance calibration (for example, a known object dimension).
  - Exports textured OBJ/MTL, GLB and PLY assets with machine-readable quality reports.
  - Uses versioned sessions under `data/phone_reconstruction/session_XXX/`.
- **MODE 3: Engineering / Debug (`E`)**
  - Full diagnostic telemetry: 3D occupancy grid voxels (free/occupied/unknown), obstacle bounding boxes, clearance distance vectors, camera frustums, and real-time FPS counter (>100 FPS on Apple Silicon).

---

## 3. Running M7.2 & M7.3 Pipelines

### 3.1 Photorealistic Surround Texture Generation (M7.2)
```bash
python scripts/generate_textured_meshes.py
```
Outputs textured OBJ+MTL+PNG and binary UV cache PLY models to `data/textured_meshes/`.

### 3.2 Smartphone Photogrammetry Experiments (M7.3)
```bash
python scripts/run_phone_photogrammetry_experiment.py
```
Executes:
- **Experiment 1 (Phase F — 4 Photos)**: 4 sequential orbital photos with ~70% overlap around vehicle front, calibrated with 2.61m wheelbase reference distance (`session_001`).
- **Experiment 2 (Phase F — Failure Detection)**: 4 outward-facing surround cameras (0% mutual overlap) triggering automated failure diagnosis and user capture guidance (`session_002`).
- **Experiment 3 (Phase G — 12 Photos 360°)**: 12 sequential orbital photos around full 360° perimeter producing 490,000 dense points and 175,567 mesh triangles (`session_003`).

---

## Automatic Photo-to-3D Reconstruction

In **MODE 2: Phone Photo 3D World**, the user uploads photographs and starts reconstruction. No manual backend, quality preset, region drawing, or depth values are required.

CLEAR360 automatically:
- Selects **COLMAP + OpenMVS** when COLMAP and all required OpenMVS executables are available.
- Falls back to the built-in **CLEAR360 Python / Open3D** reconstruction pipeline when the external tools are unavailable.
- Opens with **Fast Preview** as a geometry-only diagnostic mode. Select **Standard**, **High**, or **Maximum** when you need a complete textured mesh and GLB; Fast Preview intentionally skips meshing and texturing.
- Archives uploaded originals under `source_uploads/` and creates SfM-readable working copies under `original_images/`; HEIC, TIFF and BMP inputs are normalized to JPEG without discarding the original upload.
- Accepts walkaround videos (`.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`, `.webm`, `.3gp`) as an alternative input. It samples at most 40 candidate timestamps per import, skips severely blurred and near-identical adjacent frames, and adds the extracted JPEGs to the same photo-set audit and reconstruction pipeline. The original video is archived under `source_uploads/videos/`, and each extracted frame's source video and timestamp are recorded in the session manifests.
- Saves `input_manifest.json` to map each original filename to its working filename and conversion status.
- Saves a reconstruction session, textured OBJ/MTL/PNG for the built-in viewer, and a textured GLB for Blender/external tools when using OpenMVS.
- Estimates the scene's vertical axis from registered COLMAP camera poses and aligns OBJ, PLY, dense cloud and GLB exports to +Z when the camera-up directions are consistent.
- The built-in OBJ loader resolves independent vertex/UV/normal indices and combines multiple diffuse material atlases, so UV seams do not silently fall back to a grey mesh.

For automatic COLMAP + OpenMVS detection, CLEAR360 resolves the tools in this order:

1. An explicit `CLEAR360_OPENMVS_DIR` environment variable.
2. A complete `~/Applications/CLEAR360-OpenMVS-fixed-texture` overlay (when all five executable tools are present).
3. The conventional `~/Applications/OpenMVS-2.4.0` installation, then `PATH`.

The fixed overlay lets the existing geometry executables remain unchanged while replacing only `TextureMesh`. It is selected automatically on launch when complete, so the texture fix does not depend on a one-terminal-only export.

On macOS, launch the application through the project wrapper so the intended Conda environment and COLMAP binary are selected every time:

```bash
bash scripts/launch_viewer_macos.sh
```

The default Conda environment name is `clear360-py314`; override it with `CLEAR360_CONDA_ENV` if your environment uses another name. You can still set `CLEAR360_OPENMVS_DIR` explicitly to override automatic tool discovery.

The backend selector checks for COLMAP and all required OpenMVS executables before starting. Its decision is shown in the reconstruction status label. The OpenMVS processing log is saved at `data/phone_reconstruction/<session>/openmvs_pipeline.log`.

The OpenMVS result remains in uncalibrated photogrammetry units. The automatic workflow does not request a manually supplied scale measurement.

For a session created before camera-up alignment was added, update the code and run the one-time post-process helper below. It aligns the newest saved OpenMVS session without rerunning dense reconstruction; original OBJ/PLY/GLB files are backed up with a `.before_orientation` suffix:

```bash
python scripts/align_openmvs_session.py
```


**Capture requirements still matter, including for video:** the camera must move physically around the same static scene. Panning or rotating from one fixed position does not create parallax. Video sampling is a convenience for collecting frames; it does not manufacture missing viewpoints or guarantee sufficient overlap. Walk steadily, keep the subject visible, maintain strong overlap between adjacent views, and include different heights where possible. COLMAP's guidance recommends high overlap, different viewpoints, consistent illumination, and good surface texture. See the [COLMAP reconstruction tutorial](https://colmap.github.io/tutorial.html).

## Reconstruction quality and limitations

A successful export means the files were written; it does not by itself prove that the scene is complete or geometrically accurate. Each completed session now includes:

- `photo_set_report.json`: readable-image, duplicate, image-quality, camera/lens and dimension diagnostics.
- `reconstruction_quality.json`: mesh topology, disconnected components, small fragments, OBJ material/texture references and GLB loadability.
- `report.json`: registration, camera-overlap, geometry, texturing and reconstruction-grade summaries.

See [docs/RECONSTRUCTION_QUALITY.md](docs/RECONSTRUCTION_QUALITY.md) for metric definitions and interpretation. Open surfaces are valid outputs for many real scenes. Scale remains in arbitrary SfM coordinate units unless metric calibration is provided.

## 4. Documentation Reports

- **M7.2 Photorealistic Viewer Report**: [docs/m7_2_photorealistic_report.md](docs/m7_2_photorealistic_report.md)
- **M7.3 Smartphone Photogrammetry Report**: [docs/m7_3_photogrammetry_report.md](docs/m7_3_photogrammetry_report.md)
- **Single-Camera Validation Report**: [docs/m7_2_single_camera_validation_report.md](docs/m7_2_single_camera_validation_report.md)
- **Diagnostic Audit**: [docs/m7_2_initial_audit.md](docs/m7_2_initial_audit.md)

---

## 3. Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | 3.11+ | Required |
| Blender | 4.x | Must be on `PATH` or set `BLENDER_PATH` |
| GPU | Optional | Apple MPS or NVIDIA CUDA accelerates depth inference |
| OpenEXR libraries | system | Required by the `OpenEXR` Python package |

**macOS OpenEXR system dependency:**
```bash
brew install openexr imath
```

**Linux:**
```bash
sudo apt-get install libopenexr-dev
```

---

## 4. Installation

```bash
cd CAR360
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 5. Running the Blender Export (M1)

### Direct Blender invocation

```bash
blender --background --python sim/export_frames.py -- --output . --frames 20
```

### Using the helper script

```bash
bash scripts/run_export.sh
```

With custom options:

```bash
bash scripts/run_export.sh --frames 20 --output /path/to/output
```

If Blender is not on `PATH`, set the environment variable:

```bash
BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender bash scripts/run_export.sh
```

### Verifying the export

After the export completes, verify all files are present and valid:

```bash
python scripts/verify_export.py --data-root .
```

---

## 6. Running Tests

```bash
pytest pipeline/tests/ -v
```

Run the full test suite (all modules):

```bash
pytest -v
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html -v
```

---

## 7. Data Directory Layout

```
data/
├── .gitignore                        # ignores sim_frames/, gt_depth/, gt_position/, gt_semantic/
├── sim_frames/                       # RGB renders (gitignored)
│   ├── front/
│   │   ├── frame_0000.png            # 1280×720 RGB 8-bit PNG
│   │   ├── frame_0001.png
│   │   └── ... (20 frames total)
│   ├── rear/
│   ├── left/
│   └── right/
├── gt_depth/                         # Ground-truth depth EXRs (gitignored)
│   ├── front/
│   │   ├── frame_0000.exr            # 1280×720 float32, channel "Z" = camera Z-depth metres
│   │   └── ...
│   ├── rear/
│   ├── left/
│   └── right/
├── gt_position/                      # World-space position EXRs (gitignored)
│   ├── front/
│   │   ├── frame_0000.exr            # 1280×720 float32, channels X Y Z = world XYZ metres
│   │   └── ...
│   ├── rear/
│   ├── left/
│   └── right/
└── gt_semantic/                      # Semantic class PNGs (gitignored)
    ├── front/
    │   ├── frame_0000.png            # 1280×720 uint8, pixel value = pass_index class
    │   └── ...
    ├── rear/
    ├── left/
    └── right/
```

`cameras.yaml` is written at the project root (or `--output` dir) by `export_frames.py`.

---

## 8. Coordinate Conventions

CLEAR360 uses a right-hand coordinate system throughout. A brief summary:

- **World / Vehicle frame**: X = right, Y = forward (direction of travel), Z = up. Origin at vehicle centre.
- **Camera frame**: OpenCV convention — x = right in image, y = down, z = into scene (+Z forward).
- **Blender cameras** look along local −Z; a correction matrix `diag(1,−1,−1,1)` converts to OpenCV convention.
- **Depth**: Blender Z-pass stores camera-space Z-depth (not Euclidean distance). See [`docs/coordinate_conventions.md`](docs/coordinate_conventions.md) for full derivations.

See **[docs/coordinate_conventions.md](docs/coordinate_conventions.md)** for the complete mathematical reference.

---

## 9. Semantic Class Table

| `pass_index` | Class Name | Description |
|:---:|---|---|
| 0 | unknown / background | Sky, unclassified surfaces |
| 1 | road | Ground plane, driveable surface |
| 2 | vehicle\_ego | The CLEAR360 host vehicle (all parts) |
| 3 | parked\_vehicle | Other vehicles in the scene |
| 4 | wall | Structural walls, barriers |
| 5 | pillar | Structural pillars, columns |
| 6 | obstacle\_dynamic | Moving / dynamic obstacles |
| 7 | kerb\_speed\_breaker | Kerbs, speed breakers, raised features |

---

## 10. Milestone Progress

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M1** | Blender synthetic data generation | 🟡 **IN PROGRESS** |
| M2 | AI depth estimation (Depth Anything V2) | ⬜ Planned |
| M3 | Point-cloud reconstruction from depth | ⬜ Planned |
| M4 | Multi-camera point-cloud fusion | ⬜ Planned |
| M5 | 3D scene graph construction | ⬜ Planned |
| M6 | Obstacle detection and tracking | ⬜ Planned |
| M7 | Real-time PySide6 3D viewer | ⬜ Planned |
| M8 | Full pipeline integration | ⬜ Planned |
| M9 | Performance optimisation & final demo | ⬜ Planned |

---

## 11. Limitations and Assumptions

See **[docs/limitations.md](docs/limitations.md)** for the full list of known M1 limitations, including:

- Static scene within each frame (no motion blur)
- Perfect metric depth from Blender (AI depth will have scale ambiguity)
- Pinhole camera model — no lens distortion in simulation
- EXR Position pass requires Cycles renderer (not EEVEE)
- 20 frames is a minimal test set; production dataset must include varied lighting and obstacle positions
