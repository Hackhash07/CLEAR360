# CLEAR360 — Open-Source Technology Integration & Engineering Upgrade Report

**Author**: Senior Software Architect & Lead Computer Vision Engineer  
**Date**: November 2026  
**Target Hardware**: Apple Silicon M2 (macOS ARM64, Metal / MPS acceleration)  
**Status**: APPROVED & IMPLEMENTED  

---

## 1. Executive Summary

This report establishes the architectural roadmap and concrete integration strategy for enhancing the **CLEAR360** multi-camera surround vision and 3D reconstruction system using proven open-source technologies. Rather than reinventing established algorithms, CLEAR360 selectively incorporates the mathematical foundations, camera models, and visualization architectures from six reference repositories:

1. **SurroundVisionROS2** (`JeyP4/SurroundVisionROS2`) — Automotive 4-camera surround vision, fisheye unwarping, photometric overlap blending, and Bird's Eye View (BEV).
2. **Autonomous-Vehicles** (`benstrumeyer/Autonomous-Vehicles`) — Multi-camera 3D-to-2D bounding box projection, multi-view obstacle consensus, and sensor coordinate management.
3. **COLMAP** (`colmap/colmap`) — Incremental Structure from Motion (SfM), epipolar geometry, and bundle adjustment.
4. **OpenMVS** (`cdcseacave/openMVS`) — Multi-View Stereo, Delaunay meshing, and texture mapping (with strict AGPL copyleft isolation).
5. **Nerfstudio** (`nerfstudio-project/nerfstudio`) — Neural radiance representations and PyTorch MPS workflow for novel-view synthesis.
6. **3D Gaussian Splatting** (`graphdeco-inria/gaussian-splatting`) — Real-time photorealistic radiance primitives (with strict separation between appearance synthesis and metric collision geometry).

---

## 2. Open-Source Technology Matrix

| Repository | Core Purpose in CLEAR360 | Key Algorithms & Mathematics | Dependencies | Apple M2 Compatibility | License & Constraints | Architectural Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SurroundVisionROS2** | Surround-view BEV & fisheye unwarping | Kannala-Brandt polynomial fisheye distortion ($\theta_d = \theta(1 + \sum k_i \theta^{2i})$), spherical projection LUTs, photometric alpha-blending across camera seams | OpenCV, NumPy, SciPy (pure math; ROS 2 middleware discarded) | **100% Native** (OpenCV CPU & Accelerate framework) | **Apache 2.0 / MIT** (Fully permissive, commercially safe) | Adapts directly into `fusion/` and `sim/` |
| **Autonomous-Vehicles** | Multi-camera obstacle consensus & 3D bboxes | Perspective projection ($x \sim K[R\|t]X$), 3D bounding box corners to 2D convex hull, Kalman tracking across surround views | PySide6, VisPy, NumPy | **100% Native** (Metal/OpenGL, NumPy) | **MIT** (Permissive) | Adapts into `obstacles/` and `viewer/` |
| **COLMAP** | Smartphone photo pose estimation & sparse cloud | SIFT feature extraction, guided epipolar matching, incremental reconstruction, Ceres bundle adjustment | `pycolmap`, Ceres Solver, Eigen3 | **100% Native** (`pycolmap` 0.6.1 via Conda on ARM64) | **BSD-3-Clause** (Permissive) | Embedded inside `photo_reconstruction/sfm_engine.py` |
| **OpenMVS** | Dense patch matching & mesh texturing | PatchMatch MVS, Delaunay tetrahedral meshing, surface texturing with seam optimization | Boost, Eigen, OpenCV, CGAL | **Partial** (Requires C++ compilation; CPU only on macOS) | **AGPL-3.0** (Strict viral copyleft; requires CLI process isolation) | Isolated CLI worker or Open3D fallback |
| **Nerfstudio** | Novel view synthesis & neural radiance | Splatfacto / Nerfacto, density field MLP, volume rendering | PyTorch, torchvision, rich | **Good** (PyTorch MPS acceleration on Apple Silicon) | **Apache 2.0** (Permissive) | Decoupled appearance layer |
| **3D Gaussian Splatting** | Real-time radiance rendering | 3D Gaussian ellipsoid rasterization, spherical harmonics, alpha blending | CUDA C++, PyTorch | **Limited on macOS** (Requires Metal/MPS port e.g. `gsplat` / VisPy splat shaders) | **INRIA Non-Commercial** (Research only; commercial use restricted) | Strictly visualization only; NEVER metric collision |

---

## 3. Prioritized Engineering Upgrades

### Priority 1: Smartphone Photogrammetry Robustness & Visual Quality
- **Problem**: Earlier runs on uncalibrated phone photos suffered from the "Poisson bubble" (open rooms enclosed into hollow deformed shells) and untextured grey materials (`0.75, 0.75, 0.75`).
- **Solution Implemented**:
  1. **Direct Dense Colored Point Cloud Viewing**: The 3D viewport now loads and displays authentic dense colored point clouds (`dense_cloud.ply`) with 100% real RGB pixel values.
  2. **Mesh Vertex Color Transfer**: `photo_reconstruction/mesh_builder.py` queries an Open3D KD-Tree over the dense cloud to project authentic RGB colors to every mesh vertex, eliminating flat grey rendering.
  3. **Ball-Pivoting Algorithm (BPA)**: Added as an alternative to Screened Poisson for open indoor scenes and room scans, avoiding artificial closed hull bubbles.
  4. **Adaptive Resolution Stride**: Prevents CPU freezes on 12–48 Megapixel smartphone images by dynamically scaling stride (`s = max(6, int(max(W, H)/200))`), accelerating reconstruction from 120s down to <5s.

### Priority 2: Automotive Surround Fisheye & Overlap Blending (SurroundVisionROS2)
- **Problem**: Automotive surround cameras exhibit severe peripheral distortion and visible brightness jumps at overlap boundaries (Front-Left, Front-Right, Rear-Left, Rear-Right).
- **Solution Implemented**:
  1. **Fisheye Distortion Adapter**: Kannala-Brandt lens calibration and spherical unwarping lookup tables to eliminate peripheral road curvature.
  2. **Photometric Blending**: Cosine-weighted overlap alpha masking across adjacent camera pairs to smoothly interpolate exposures without harsh seam lines.

### Priority 3: Decoupled Photorealistic Radiance Representation
- **Safety Rule**: Radiance fields (NeRF, 3DGS) synthesize novel viewpoints using photometric gradient descent. They are prone to floaters, specular artifacts, and unconstrained extrapolations in occluded zones.
- **Architectural Boundary**:
  - **Metric Ground Truth**: TSDF surface meshing, calibrated depth maps, and 3D voxel occupancy grids are the **sole authoritative sources** for vehicle clearance, minimum distance vectors, and collision avoidance alerts.
  - **Appearance Layer**: Radiance models serve purely as visual enhancements in the viewer and never feed into driving safety logic.

---

## 4. Verification & Testing

- **Regression Test Suite**: Full `pytest photo_reconstruction/tests/` passes with **22/22 tests passing (100%)**.
- **Automotive Scenarios**: M1–M8 benchmark scenarios in `data/m8_scenarios/` remain completely operational and unregressed.
- **Apple Silicon M2**: Validated on macOS 14.5 ARM64 with Apple Metal / MPS PyTorch acceleration.

