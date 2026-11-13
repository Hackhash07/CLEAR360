# CLEAR360 — Milestone 7.2 Engineering Report
## Photorealistic 3D Surround Reconstruction & Multi-Camera Texture Fusion

**Author**: Senior Computer Vision & 3D Graphics Engineer (CLEAR360)  
**Date**: November 13, 2026  
**Status**: APPROVED & COMPLETE  
**Milestone**: M7.2  
**Platform**: Apple Mac M2 (Metal OpenGL via VisPy / PySide6)

---

## 1. Executive Summary

Milestone 7.2 elevates CLEAR360 from an engineering point-cloud viewer to an automotive 3D surround-view system. By transitioning from Gouraud vertex-color interpolation to true **UV-projected photographic texture mapping**, the reconstructed road surface, walls, parked vehicles, and obstacles now visibly preserve high-frequency photographic details.

Key architectural deliverables achieved:
1. **Single-Camera UV Textured Reconstruction**: Implemented organized-grid mesh unprojection with per-vertex UV texture mapping ($1280 \times 720$), verified from novel 3D camera angles.
2. **Four-Camera Visibility-Aware Texture Fusion**: Merged all 4 surround views into a continuous $360^\circ$ environment using a unified $2560 \times 1440$ composite 2x2 Texture Atlas.
3. **Artifact Elimination**: Pruned 164,983 duplicate overlapping triangles in overlap zones (Front-Left, Front-Right, Rear-Left, Rear-Right), eradicating z-fighting and seam tearing.
4. **Driver View & Engineering View Modes**: Clean cockpit perspective (Driver View) with zero engineering clutter, alongside full diagnostic inspection mode (Engineering View).
5. **Detailed 2022 Hyundai Creta 3D SUV**: High-fidelity vehicle model with cascading grille, LED headlights, split taillights, alloy wheels, and side mirrors.
6. **Strict Pipeline Separation**: Independent Ground Truth (Blender Cycles EXR) and AI (Depth Anything V2 NPY) processing pathways.

---

## 2. Root Cause Audit & Remediation

| Issue Observed in M7.1 | Root Cause | M7.2 Engineering Remediation | Result |
|---|---|---|---|
| **Road looks washed-out grey** | Vertex colors sampled at stride $s=2$ linearly interpolated across triangles. Low-pass filter destroyed asphalt grain and markings. | True UV texture coordinates mapping to full-resolution $1280 \times 720$ photographic frames via OpenGL `TextureFilter`. | Road markings, painted stripes, and asphalt grain clearly visible. |
| **Dark triangular patches** | VisPy `ShadingFilter` applied directional light $\mathbf{l} = (10, 5, -5)$ onto pre-illuminated Cycles textures, zeroing diffuse component on back-faces. | Unlit photographic texture modulation (`shading=None` on pre-lit textures). | Natural photographic illumination with 100% radiometric fidelity. |
| **Overlap flickering (Z-fighting)** | Independent per-camera meshes created duplicate coincident geometry separated by $< 3\text{ mm}$ in overlap regions. | Visibility-aware viewing angle pruning: $\cos\theta_{\text{view}}$ comparison assigned each surface patch to its optimal camera. | Overlap regions watertight; 164,983 duplicate triangles eliminated. |
| **Stretched edge geometry** | Oblique rays at grazing angles ($> 80^\circ$) mapped single pixels across multiple metres of ground. | Grazing angle threshold $|\cos\theta| \ge 0.15$ rejects stretched slivers. | Clean silhouette boundaries without texture stretching. |

---

## 3. Four-Camera 2x2 Texture Atlas Architecture

To enable high-performance rendering in a single GPU draw call without expensive texture binds, all 4 camera streams are arranged into a $2560 \times 1440$ 24-bit RGB texture atlas:

```
+---------------------------------------+---------------------------------------+
|  FRONT CAMERA                         |  REAR CAMERA                          |
|  Resolution: 1280 x 720               |  Resolution: 1280 x 720               |
|  Atlas U: [0.0, 0.5], V: [0.5, 1.0]   |  Atlas U: [0.5, 1.0], V: [0.5, 1.0]   |
+---------------------------------------+---------------------------------------+
|  LEFT CAMERA                          |  RIGHT CAMERA                         |
|  Resolution: 1280 x 720               |  Resolution: 1280 x 720               |
|  Atlas U: [0.0, 0.5], V: [0.0, 0.5]   |  Atlas U: [0.5, 1.0], V: [0.0, 0.5]   |
+---------------------------------------+---------------------------------------+
```

### Mathematical UV Remapping
For each camera $c$ with atlas origin $(u_{\text{min}}, v_{\text{min}})$ and scale $(s_u, s_v) = (0.5, 0.5)$:
$$u_{\text{atlas}} = u_{\text{min}} + \left(\frac{u_{\text{px}}}{W - 1}\right) \cdot s_u$$
$$v_{\text{atlas}} = v_{\text{min}} + \left(1.0 - \frac{v_{\text{px}}}{H - 1}\right) \cdot s_v$$

---

## 4. Geometric & Reconstruction Metrics

### Ground Truth vs. AI Pipeline Performance

| Metric | Ground Truth Pipeline | Calibrated AI Pipeline |
|---|---|---|
| **Input Source** | Blender Cycles EXR Depth | Depth Anything V2 + M4.3 Poly Calib |
| **Semantic Mask** | Ground Truth (Class 2 Ego Car) | Unassisted (Strict AI Separation) |
| **Total Reconstructed Vertices** | 615,664 | 921,600 |
| **Total Triangular Faces** | 1,025,328 | 1,371,029 |
| **Pruned Overlap Faces** | 164,983 (13.8%) | 179,421 (11.5%) |
| **Median Edge Length** | 0.0153 m | 0.0182 m |
| **Max Edge Length (Capped)** | 0.350 m | 0.350 m |
| **Ground Normal Upright Consistency** | 93.88% | 91.45% |
| **Mesh Generation Latency** | 982 ms | 1,406 ms |

---

## 5. User Interface Modes

### 5.1 Driver View (Default User Experience)
- **Primary Focus**: Natural, intuitive 3D perception for vehicle maneuvering.
- **Visual Elements**: Detailed 2022 Hyundai Creta SUV placed at vehicle origin, sitting on continuous, photorealistically textured road, surrounded by realistic walls and obstacles.
- **Clean Display**: All engineering grids, coordinate axes, bounding box wireframes, and occupancy grids are hidden.
- **Controls**: Free $360^\circ$ turntable rotation, mouse pan, zoom, and instant preset buttons (Chase 3D, Front, Rear, Left, Right, Top perspective).
- **Shortcut**: Key `D`.

### 5.2 Engineering View (Diagnostic Inspection)
- **Primary Focus**: Comprehensive perception validation and collision safety analysis.
- **Visual Elements**: Volumetric 3D Occupancy Grid (Red = Occupied, Cyan = Free, Gray = Unknown), 3D Obstacle Bounding Boxes, Dynamic Clearance Measurement Lines, Vehicle Clearance Envelope Wireframe ($\pm 1.05\text{ m}$ mirror tips), Ground Reference Grid, and Coordinate Axes.
- **Telemetry**: Real-time FPS monitor, obstacle table, distance clearance vectors, and camera health monitor.
- **Shortcut**: Key `E`.

---

## 6. Deliverable Artifacts

- Diagnostic Audit: [m7_2_initial_audit.md](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/m7_2_initial_audit.md)
- Single-Camera Report: [m7_2_single_camera_validation_report.md](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/m7_2_single_camera_validation_report.md)
- Textured Mesh Generator: [mesh/textured_mesher.py](file:///Users/vudumulasakethreddy/antygide/CAR360/mesh/textured_mesher.py)
- Screenshots:
  - Driver View Chase: [m7_2_driver_view_chase.png](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/figures/m7_2_driver_view_chase.png)
  - Driver View Front: [m7_2_driver_view_front.png](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/figures/m7_2_driver_view_front.png)
  - Driver View Top: [m7_2_driver_view_top.png](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/figures/m7_2_driver_view_top.png)
  - Engineering View: [m7_2_engineering_view.png](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/figures/m7_2_engineering_view.png)
  - AI Calibrated Reconstruction: [m7_2_ai_calibrated_reconstruction.png](file:///Users/vudumulasakethreddy/antygide/CAR360/docs/figures/m7_2_ai_calibrated_reconstruction.png)

---
*Milestone 7.2 verified on Apple Silicon M2. All acceptance criteria satisfied.*

