# CLEAR360 — Real-Time 3D Automotive Surroundings & Smartphone Photogrammetry

> **Status — M7.2 & M7.3 COMPLETE**:
> - **M7.2**: Photorealistic 3D Automotive Surround Reconstruction Viewer with true UV composite texture atlas, visibility-aware overlap triangle culling, detailed 2022 Hyundai Creta SUV model, and distinct Driver vs. Engineering views.
> - **M7.3**: Smartphone Photograph to 3D World Reconstruction via pycolmap / OpenCV Structure-from-Motion, supplementary Depth Anything V2 multi-view stereo, Screened Poisson Surface Meshing, and metric scale calibration.

---

## 1. Project Overview

**CLEAR360** is an automotive computer vision and 3D perception system combining:
1. **Four-Camera Automotive Surround**: Reconstructs a full 360° photorealistic 3D perspective environment around a Hyundai Creta SUV in real time using four wide-angle surround cameras (front, rear, left, right).
2. **Smartphone Photogrammetry 3D World**: An independent photogrammetry pipeline that accepts arbitrary collections of sequential smartphone photos (4 to 40+), extracts EXIF hardware metadata, performs incremental bundle adjustment, backprojects dense metric depth, and recovers watertight, scaled 3D environments.

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
  - Ingests handheld smartphone photographs (`.jpg`, `.jpeg`, `.png`, `.heic`).
  - Automated EXIF extraction (35mm equivalent focal length, camera make/model).
  - Sharpness and exposure quality scoring (Laplacian variance, pixel clipping).
  - Incremental Structure-from-Motion camera registration.
  - Known metric reference distance calibration (e.g. wheelbase or known object dimension).
  - Versioned session management (`data/phone_reconstruction/session_XXX/`).
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
- Uses the balanced `standard` quality preset to keep memory use and runtime reasonable on laptop hardware.
- Saves a reconstruction session, textured OBJ/MTL/PNG for the built-in viewer, and a textured GLB for Blender/external tools when using OpenMVS.

For automatic COLMAP + OpenMVS detection, install the tools locally and set the OpenMVS executable directory:

```bash
export CLEAR360_OPENMVS_DIR="$HOME/Applications/OpenMVS-2.4.0"
export CLEAR360_COLMAP_BIN="$(command -v colmap)"
```

The backend selector checks for COLMAP and all required OpenMVS executables before starting. Its decision is shown in the reconstruction status label. The OpenMVS processing log is saved at `data/phone_reconstruction/<session>/openmvs_pipeline.log`.

The OpenMVS result remains in uncalibrated photogrammetry units. The automatic workflow does not request a manually supplied scale measurement.

**Capture requirements still matter:** photographs must show the same static scene from physically different viewpoints, with strong visual overlap. Each important surface should be visible in multiple images. Six unrelated views or photographs taken from one position by only rotating the camera do not provide enough parallax for a complete 3D model. COLMAP's guidance recommends high overlap, different viewpoints, consistent illumination, and good surface texture. See the [COLMAP reconstruction tutorial](https://colmap.github.io/tutorial.html).

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
