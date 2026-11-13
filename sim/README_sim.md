# CLEAR360 — Simulation & Synthetic Data Generation Layer

The `sim/` package implements the procedural 3D simulation environment and multi-camera surround-view rendering pipeline for **CLEAR360** (Real-Time 3D Automotive Surroundings Reconstruction) using Blender headless mode and Cycles.

---

## 1. Prerequisites

- **Blender**: Version 4.0 or newer (tested on Blender 4.x).
  - Headless execution uses Blender's bundled Python interpreter.
  - Bundled libraries used: `bpy`, `mathutils`, `numpy`, `yaml`.
- **Hardware**:
  - **GPU**: Metal (Apple Silicon macOS), OptiX / CUDA (NVIDIA Linux / Windows), HIP (AMD).
  - **CPU**: Supported on any modern multi-core x86_64 or ARM64 processor (Cycles will automatically fallback to CPU compute if no GPU device is detected).
- **RAM**: Minimum 8 GB recommended for 32-sample Cycles rendering at 1280×720.

---

## 2. Command to Run

Execute the export script in background (headless) mode:

```bash
blender --background --python sim/export_frames.py -- --output /path/to/CAR360 --frames 20
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--output` | `Path` | Current working directory | Root project directory under which `data/` is written |
| `--frames` | `int` | `20` | Number of sequential frames to render (Blender frames 1–20) |
| `--samples` | `int` | `32` | Cycles samples per pixel |
| `--dry-run` | `flag` | `False` | Construct the scene and write `cameras.yaml` without rendering pixels |

---

## 3. Output File Tree

All output paths are generated relative to the `--output` argument:

```
data/
├── cameras.yaml                                # Complete rig calibration (W2C & intrinsics)
├── sim_frames/                                 # RGB rendered frames (8-bit PNG, 1280×720)
│   ├── front/frame_0000.png ... frame_0019.png
│   ├── rear/frame_0000.png  ... frame_0019.png
│   ├── left/frame_0000.png  ... frame_0019.png
│   └── right/frame_0000.png ... frame_0019.png
├── gt_depth/                                   # Camera-space Z-depth in metres (32-bit float OpenEXR, channel "Z")
│   ├── front/frame_0000.exr ... frame_0019.exr
│   ├── rear/frame_0000.exr  ... frame_0019.exr
│   ├── left/frame_0000.exr  ... frame_0019.exr
│   └── right/frame_0000.exr ... frame_0019.exr
├── gt_position/                                # World XYZ coordinates (32-bit float OpenEXR, channels X, Y, Z)
│   ├── front/frame_0000.exr ... frame_0019.exr
│   ├── rear/frame_0000.exr  ... frame_0019.exr
│   ├── left/frame_0000.exr  ... frame_0019.exr
│   └── right/frame_0000.exr ... frame_0019.exr
└── gt_semantic/                                # Semantic instance segmentation masks (8-bit PNG, pixel = pass_index)
    ├── front/frame_0000.png ... frame_0019.png
    ├── rear/frame_0000.png  ... frame_0019.png
    ├── left/frame_0000.png  ... frame_0019.png
    └── right/frame_0000.png ... frame_0019.png
```

> **Note on Indexing**: Blender timeline runs on frames `1` through `20`. The exporter automatically remaps these outputs to zero-based indexing (`frame_0000` through `frame_0019`) to match standard Python ML dataset conventions.

---

## 4. Coordinate Conventions

All reconstruction and unprojection code downstream relies strictly on these conventions:

### World / Vehicle Frame (Blender default axes)
- **Origin**: Centre of ego vehicle ground contact plane (`Z=0`).
- **+X**: **RIGHT** side of vehicle
- **+Y**: **FORWARD** direction of vehicle travel
- **+Z**: **UP** (perpendicular to road plane)
- **Ego Extents**:
  - Length: 4.30 m ($Y \in [-2.15, +2.15]$ m; front bumper at $Y \approx +2.15$ m)
  - Width: 1.79 m ($X \in [-0.895, +0.895]$ m)
  - Height: 1.635 m ($Z \in [0.0, 1.635]$ m)

### Camera Frame (OpenCV convention)
- **+x**: RIGHT across the image sensor
- **+y**: DOWN across the image sensor
- **+z**: FORWARD into the scene (optical axis)

### Blender Camera to OpenCV Extrinsics Conversion
Blender camera objects point along their local $-Z$ axis with $+Y$ up. To convert Blender's camera-to-world matrix $M_{\text{blender}}$ to OpenCV convention:

$$T_{\text{c2w}}^{\text{cv}} = M_{\text{blender}} \cdot \text{diag}(1, -1, -1, 1)$$

The stored matrix in `cameras.yaml` is the world-to-camera matrix $T_{\text{cam\_world}}$:

$$T_{\text{cam\_world}} = \left( T_{\text{c2w}}^{\text{cv}} \right)^{-1}$$

Given a world point in homogeneous coordinates $P_{\text{world}} = [X_w, Y_w, Z_w, 1]^T$:

$$p_{\text{cam}} = T_{\text{cam\_world}} \cdot P_{\text{world}}$$

### Depth Convention
- **Camera Z-depth**: Perpendicular distance along the camera's optical axis ($+z$), measured in **metres**, stored as a 32-bit float.
- This is **NOT** Euclidean ray length. To convert between camera-Z depth $d_z$ and Euclidean distance $d_e$:

$$d_e(u, v) = d_z(u, v) \cdot \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}$$

---

## 5. Camera Rig Specifications

Four fixed cameras provide 360° surround coverage:

| Camera | Position $(X, Y, Z)$ [m] | Look Direction | Downward Tilt | HFOV | Resolution |
|---|---|---|---|---|---|
| **front** | $(0.00, +2.15, 0.90)$ | $+Y$ (Forward) | $-8^\circ$ | $\approx 120^\circ$ | $1280 \times 720$ |
| **rear** | $(0.00, -2.15, 0.90)$ | $-Y$ (Backward) | $-8^\circ$ | $\approx 120^\circ$ | $1280 \times 720$ |
| **left** | $(-1.09, +0.30, 1.25)$ | $-X$ (Left) | $-6^\circ$ | $\approx 120^\circ$ | $1280 \times 720$ |
| **right** | $(+1.09, +0.30, 1.25)$ | $+X$ (Right) | $-6^\circ$ | $\approx 120^\circ$ | $1280 \times 720$ |

All cameras share identical pinhole intrinsics:
- Sensor width: $36.0\text{ mm}$
- Focal length: $10.39\text{ mm}$
- Focal length (pixels): $f_x = f_y = \frac{10.39}{36.0} \times 1280 \approx 369.42\text{ px}$
- Principal point: $c_x = 640.0\text{ px},\ c_y = 360.0\text{ px}$

---

## 6. Semantic Class Map (`pass_index`)

The simulation assigns Blender object pass indices (`pass_index`) to every object in the scene. The compositor renders the `IndexOB` pass to uint8 PNG files:

| `pass_index` | Class Name | Scene Entities |
|:---:|---|---|
| **0** | `background / sky` | Environment background, void |
| **1** | `road / ground` | Road surface plane ($20 \times 20\text{ m}$) |
| **2** | `ego vehicle` | Creta chassis, roof, 4 wheels, 2 side mirrors |
| **3** | `other vehicle` | Parked cars 1 & 2 ($4.2 \times 1.75 \times 1.5\text{ m}$) |
| **4** | `wall / structure` | North, south, east, west perimeter concrete walls |
| **5** | `pillar / post` | Cylindrical column ($r=0.20\text{ m}, h=2.5\text{ m}$) |
| **6** | `dynamic obstacle` | Animated obstacle mover cube (keyframes 1, 10, 20) |
| **7** | `road furniture` | Concrete kerbs left & right, yellow speed breaker |

---

## 7. How to Verify Outputs

After running the export script, verify data completeness, shapes, and intrinsics using the built-in verifier:

```bash
python scripts/verify_export.py --data-root /path/to/CAR360
```

The script checks:
1. All 80 RGB images exist and are non-empty $(4 \text{ cameras} \times 20 \text{ frames})$.
2. All 80 depth EXRs exist and contain float32 values with channel `"Z"`.
3. All 80 position EXRs exist with channels `"X"`, `"Y"`, `"Z"`.
4. All 80 semantic PNG masks exist.
5. Loads sample frame (`front`, frame 0) to ensure dimensions are exactly $(720, 1280)$ and depth values are positive and non-zero.
6. Validates `cameras.yaml` against expected camera names and intrinsics ($f_x \approx 369.4$, $c_x = 640$, $c_y = 360$).

---

## 8. Known Assumptions and Limitations

1. **Static Scene Within Frame**: No sub-frame shutter time / motion blur is rendered.
2. **Ideal Pinhole Model**: Lens distortion (radial/tangential) is zero in the simulation. Physical automotive lenses typically feature barrel distortion requiring undistortion models (Brown-Conrady / Kannala-Brandt).
3. **Lighting**: Outdoor daytime sun lamp (elevation 55°, azimuth 160°) with fixed intensity. Does not currently simulate dynamic weather, dusk/night conditions, or indoor underground artificial fluorescent lights.
4. **Simplified Vehicle Geometry**: The vehicle is procedurally assembled from primitives (bevelled cuboids and cylinders). A high-poly CAD/OBJ asset can be substituted in `assets/vehicle/` for future milestones.
5. **EXR Position Pass**: The world-space position pass is exclusively supported by Cycles; EEVEE does not produce coordinate passes.
