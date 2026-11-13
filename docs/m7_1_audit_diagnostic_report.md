# CLEAR360 — M7.1 Part 1 Audit Diagnostic Report: Point Cloud Rendering & Color Analysis

**Date:** November 13, 2026  
**Objective:** Investigate why the M7 3D viewer displays sparse white/grey point clouds in "Photorealistic RGB" mode and determine root causes across the data pipeline, filtering stage, color normalization, and rendering stack.

---

## 1. Executive Summary of Audit Findings

```
+------------------------------------------------------------------------------------------------------------------+
| AUDIT QUESTION                                       | FINDING    | FACTUAL EVIDENCE                             |
+------------------------------------------------------+------------+----------------------------------------------+
| 1. Are original RGB colours stored in PLY vertices?  | YES        | (136751, 3) float64 colors in PLY; 9,298     |
|                                                      |            | unique colors present.                       |
| 2. Does the loader preserve and display colours?     | YES        | ViewerDataLoader loads pcd.colors correctly  |
|                                                      |            | and passes RGBA to visuals.Markers.          |
| 3. Are colours incorrectly normalized (0–1 vs 0–255)?| NO         | Colors are normalized via / 255.0 to [0, 1]  |
|                                                      |            | matching VisPy float RGBA requirements.      |
| 4. Is viewer loading analytical or photoreal data?   | PHOTOREAL  | Loads frame_0000_ground_truth_clean.ply      |
|                                                      |            | derived from photoreal_frames/ (Cycles).     |
| 5. Did M6 filtering discard original RGB colours?    | NO         | SceneFilter preserves pcd.colors through     |
|                                                      |            | voxel downsampling and RANSAC plane.         |
| 6. Does Photorealistic RGB mode use camera imagery?  | YES        | Point colors directly sampled from camera    |
|                                                      |            | RGB frames during unprojection.              |
+------------------------------------------------------+------------+----------------------------------------------+
```

---

## 2. Root Cause Analysis: Why Points Appear White / Grey

Despite original RGB values being preserved in vertex arrays, the scene visually manifests as a pale grey/white cloud due to three fundamental technical causes:

### Root Cause A: Point Cloud Representation vs Continuous Surface Geometry
- In `viewer/viewport.py`, point clouds are rendered as discrete 2.5-pixel unshaded markers (`vispy.scene.visuals.Markers`).
- Because points are disconnected spatial samples (subsampled to $5\text{ cm}$ spacing in M6), over $80\%$ of the screen area between points is empty dark background (`#121318`).
- Without continuous polygonal rasterization, human vision integrates the high-frequency stipple pattern into an optical blend, destroying the perception of solid surfaces.

### Root Cause B: Zero Shading, Normals, and Lighting Response
- Point sprites in OpenGL do not possess surface normal orientation ($\vec{N}$). Consequently, VisPy cannot apply Lambertian diffuse lighting ($I = \vec{N} \cdot \vec{L}$) or Phong specular highlights.
- Without directional light shading or shadow gradients, all geometry looks completely flat and washed out.

### Root Cause C: High Scene Albedo & Lighting Calibration in Frame 0000
- In `data/photoreal_frames/*/frame_0000.png`, the Cycles parking lot environment was rendered under high-elevation sun and strong skylight:
  - Road Surface (Class 1): Mean RGB = `[165, 176, 192]` (normalized float: `[0.647, 0.690, 0.753]`)
  - Concrete Walls (Class 4): Mean RGB = `[199, 205, 212]` (normalized float: `[0.780, 0.804, 0.831]`)
  - Clear Sky (Class 0): Mean RGB = `[233, 239, 247]` (normalized float: `[0.914, 0.937, 0.969]`)
- Overall mean RGB across the entire point cloud is `[0.775, 0.788, 0.815]`.
- Under flat unshaded point rendering, `[0.775, 0.788, 0.815]` appears as a nearly uniform light grey/white cloud.

---

## 3. Required Architectural Upgrades for M7.1

1. **RGB-D Surface Mesh Reconstruction:**
   - Transform 2D depth grids into organized 3D surface meshes with triangles between neighbouring pixels.
   - Reject triangles crossing depth discontinuities ($\Delta z > 0.35\text{ m}$ or edge length $> 0.25\text{ m}$) to avoid edge stretching.
   - Assign per-vertex or per-face RGB colors directly from camera imagery.
   - Calculate surface normals to enable smooth Phong/Lambertian shading in the 3D viewport.
2. **Realistic 3D Vehicle Model:**
   - Replace the simplistic blue box model with a detailed, dimensionally accurate 2022 Hyundai Creta SUV model with headlights, taillights, glass windows, grille, side mirrors, and wheels.
3. **Immersive Automotive Surround Viewpoint:**
   - Change the default camera preset from a high-altitude overhead perspective to a realistic low-angle chase view ($16^\circ$ elevation, $7.2\text{ m}$ distance).
4. **Three Independent Visualization Modes:**
   - **PHOTOREALISTIC 3D SURFACE** (default)
   - **POINT CLOUD DEBUG**
   - **SEMANTIC / OCCUPANCY**

