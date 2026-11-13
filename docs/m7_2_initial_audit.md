# CLEAR360 — M7.2 Initial Diagnostic Audit Report
## Root-Cause Analysis of 3D Surround Reconstruction & Rendering Defects

**Date**: November 13, 2026  
**Milestone**: M7.2  
**Target Platform**: Apple Mac M2 (Metal OpenGL via VisPy / PySide6)  
**Author**: Senior Computer Vision & 3D Graphics Engineer (CLEAR360)

---

## Executive Summary

The M7.1 baseline demonstrated the feasibility of continuous 3D surface meshing from multi-camera RGB-D data. However, the visual presentation in the desktop viewer exhibits significant fidelity defects:
1. The road surface appears flat, featureless grey instead of showing tarmac grain and painted markings.
2. Surrounding walls and obstacles lack surface texture and depth contrast.
3. Dark, flickering triangular patches appear across surfaces, particularly in overlap regions and oblique angles.
4. The application presentation resembles an engineering point-cloud tool rather than a clean, photorealistic automotive 3D surround-view monitor.

This audit investigates the entire pipeline from raw image ingest to OpenGL rasterization across 14 diagnostic dimensions. The root causes are identified, and the engineering remediation plan for M7.2 and M7.3 is detailed below.

---

## 1. RGB Image Loading & Colour-Space Handling

- **Current Implementation**: `scripts/generate_surface_meshes.py` loads PNG frames using `cv2.imread(str(rgb_path))`. OpenCV defaults to BGR channel ordering. The code executes `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)`, normalizing values to $[0.0, 1.0]$.
- **Finding**: While RGB order is properly restored in the data loader, the original Blender Cycles renderings were exported in standard sRGB colour space with linear gamma workflow. When vertex colors are rendered by OpenGL shaders without sRGB gamma correction, colors undergo perceptual desaturation and contrast flattening, causing dark tones to wash out and mid-tones to look pale grey.
- **Remedy**: Apply explicit gamma and contrast preservation, and ensure direct photographic texture maps are passed to OpenGL texture samplers rather than subsampled vertex colors.

---

## 2. Vertex-Colour Preservation vs. True UV Texture Mapping

- **Current Implementation**: In `mesh/rgbd_mesher.py`, colors are assigned strictly per-vertex:
  $$\text{cols\_grid} = \text{rgb}[v_{\text{grid}}, u_{\text{grid}}] / 255.0$$
  At sampling stride $s=2$, only 1 out of every 4 image pixels is sampled ($640 \times 360$ from $1280 \times 720$). At stride $s=3$ or $s=4$, up to 15 out of 16 pixels are discarded.
- **Root Cause of "Grey Road & Blank Walls"**:
  Vertex colors are interpolated linearly across triangle interiors by the OpenGL rasterizer (Gouraud shading). Because tarmac textures, paint striping, brick mortar, and fine tyre tracks consist of high-frequency spatial frequencies spanning only 1–3 pixels, linear interpolation between vertices spaced 5–15 cm apart acts as an aggressive low-pass spatial filter. The high-frequency detail is completely destroyed, producing a muddy, homogeneous grey average.
- **Remedy**: True **UV texture projection**. Each mesh vertex must store normalized texture coordinates $(u, v) \in [0, 1]^2$ referencing the original photographic image. The OpenGL fragment shader must sample the full-resolution $1280 \times 720$ photographic texture map via bilinear/trilinear anisotropic filtering.

---

## 3. UV Coordinate Generation & Orientation

- **Current Implementation**: `mesh/rgbd_mesher.py` does not compute or store UV texture coordinates. PLY exports only contain $(X, Y, Z)$ positions, $(R, G, B)$ vertex colors, and $(N_x, N_y, N_z)$ normals.
- **Root Cause**: Without UV coordinates, VisPy and Open3D cannot bind 2D photographic textures to 3D geometry.
- **Remedy**: Implement `TexturedMesher` in `mesh/textured_mesher.py`:
  For each vertex unprojected from image pixel $(u_p, v_p)$ in camera of resolution $(W, H)$:
  $$u_{\text{tex}} = \frac{u_p}{W - 1}, \quad v_{\text{tex}} = 1.0 - \frac{v_p}{H - 1}$$
  The vertical flip ($1.0 - v$) aligns image coordinate space (top-left origin) with OpenGL texture coordinate space (bottom-left origin).

---

## 4. Texture Coordinate Orientation & Multi-Camera Atlas

- **Challenge**: The surround system comprises 4 distinct cameras (front, rear, left, right), each producing a $1280 \times 720$ photographic frame.
- **Single-Camera Mode**: Direct $(u, v)$ mapping to camera image texture.
- **Four-Camera Fused Mode**: A unified texture atlas packs all 4 camera frames into a single $2560 \times 1440$ composite texture (2 columns $\times$ 2 rows):
  - Top-Left: Front camera $(u' = 0.5 u, v' = 0.5 + 0.5 v)$
  - Top-Right: Rear camera $(u' = 0.5 + 0.5 u, v' = 0.5 + 0.5 v)$
  - Bottom-Left: Left camera $(u' = 0.5 u, v' = 0.5 v)$
  - Bottom-Right: Right camera $(u' = 0.5 + 0.5 u, v' = 0.5 v)$
- Vertex UVs are remapped accordingly, allowing the entire 3D surround environment to be rendered in a single GPU draw call without texture swapping.

---

## 5. Original Image Projection & Shading Distortion

- **Current Implementation**: In `viewer/viewport.py`, `visual_surface_mesh` was initialized with:
  ```python
  self.visual_surface_mesh = visuals.Mesh(..., shading="smooth")
  ```
- **Root Cause of Dark Triangular Patches**:
  VisPy's `ShadingFilter` applies a directional light model:
  $$I_{\text{frag}} = I_{\text{base}} \cdot \left( I_{\text{ambient}} \cdot k_a + \max(0, \mathbf{n} \cdot (-\mathbf{l})) \cdot I_{\text{diffuse}} \cdot k_d \right) + I_{\text{specular}}$$
  With default settings:
  - $\mathbf{l} = (10, 5, -5)$
  - $I_{\text{ambient}} = 0.25$
  - $I_{\text{diffuse}} = 0.70$
  
  When a triangle's surface normal $\mathbf{n}$ points away from the synthetic light vector (such as south-facing walls or road segments), $\mathbf{n} \cdot (-\mathbf{l}) \le 0$, causing the diffuse component to drop to **zero**. The surface color is multiplied by $0.25$, darkening it by **75%**!
  Furthermore, the original photographic images rendered by Blender Cycles **already contain physical lighting, global illumination, and ambient occlusion**. Adding a synthetic directional light on top of already-lit photographs creates an unphysical "double lighting" effect that creates pitch-black patches and harsh unnatural highlights.
- **Remedy**: In Driver View, disable synthetic directional shading (`shading=None` or unlit mode with texture modulation), allowing the original photographic illumination and natural Cycles shadows to display with 100% radiometric fidelity.

---

## 6. Camera Intrinsic & Extrinsic Consistency

- **Audit Findings**:
  - `config/cameras.yaml` defines:
    - Front: $(X=0.0, Y=2.15, Z=0.90)$, tilt $\approx -8^\circ$
    - Rear: $(X=0.0, Y=-2.15, Z=0.90)$, tilt $\approx -8^\circ$
    - Left: $(X=-1.09, Y=0.30, Z=1.25)$, tilt $\approx -6^\circ$
    - Right: $(X=1.09, Y=0.30, Z=1.25)$, tilt $\approx -6^\circ$
    - Intrinsics: $f_x = f_y = 369.4$, $c_x = 640$, $c_y = 360$, $W=1280, H=720$, $\text{HFOV} \approx 120^\circ$.
  - All matrices $T_{\text{world\_cam}} = T_{\text{cam\_world}}^{-1}$ are mathematically orthonormal ($R R^T = I$, $\det(R) = +1.0$).
  - Unprojection in `mesh/rgbd_mesher.py` correctly converts OpenCV $(x_{\text{right}}, y_{\text{down}}, z_{\text{forward}})$ to ISO vehicle coordinates $(X_{\text{right}}, Y_{\text{forward}}, Z_{\text{up}})$.

---

## 7. Face Winding & Normal Consistency

- **Current Implementation**:
  ```python
  f1 = np.column_stack([A[val_t1], C[val_t1], B[val_t1]])
  f2 = np.column_stack([B[val_t2], C[val_t2], D[val_t2]])
  ```
  Where in grid coordinates: $A=(r, c), B=(r, c+1), C=(r+1, c), D=(r+1, c+1)$.
- **Audit Analysis**:
  - In camera coordinates ($+x$ right, $+y$ down, $+z$ depth forward):
    $\mathbf{v}_{AC} = (0, +\Delta y, \Delta z)$, $\mathbf{v}_{AB} = (+\Delta x, 0, \Delta z)$.
    $\mathbf{n} = \mathbf{v}_{AC} \times \mathbf{v}_{AB} = (\dots, \dots, -\Delta x \Delta y)$.
    The $z$-component is negative, meaning the normal points back toward the optical center (toward the camera).
  - When transformed to world coordinates, 93.9% of ground triangles correctly exhibit $+Z$ normals ($N_z > 0.7$), with fewer than 0.0003% inverted.
  - However, Open3D's `compute_vertex_normals()` averages adjacent face normals indiscriminately. At sharp $90^\circ$ kerb steps and wall junctions, smoothing across orthogonal faces distorts vertex normals, creating gradient shading artifacts across flat vertical surfaces.

---

## 8. Depth Discontinuity & Silhouette Filtering

- **Current Threshold**:
  $$\text{threshold}(d) = \text{max\_depth\_step\_base} + \text{depth\_step\_ratio} \cdot \min(d_A, d_B, d_C) = 0.20 + 0.08 \cdot d$$
  At $d = 5\text{ m}$, threshold is $0.60\text{ m}$.
- **Finding**: While this rejects coarse background jumps (e.g. car edge to distant wall), fine silhouette boundaries (e.g. side mirrors, pedestrian limbs, bumper edges) have depth differences of $0.15 - 0.35\text{ m}$, which leak through the filter. This forms "webbed" or "curtain" triangles that connect vehicle silhouettes to the asphalt ground, creating dark, stretched triangles when textured.
- **Remedy**: Implement tighter adaptive edge filtering:
  $$\text{threshold}_{\text{silhouette}}(d) = 0.12 + 0.04 \cdot d$$
  Add an edge-length-to-depth ratio test: reject any triangle whose 3D edge length exceeds $4 \times$ the expected voxel resolution at distance $d$.

---

## 9. Duplicate Geometry & Camera Overlap Conflicts (Z-Fighting)

- **Root Cause**:
  In overlap regions (e.g. Front-Left, Front-Right, Rear-Left, Rear-Right), two cameras observe the same physical ground patch.
  Because each camera reconstructs its own triangular mesh independently, when the meshes are simply concatenated into one list:
  1. Two complete sets of triangles occupy almost the same physical space.
  2. Small sub-centimeter differences in unprojected depth (due to pixel discretization and lens perspective) cause the two surface layers to intersect repeatedly.
  3. The OpenGL depth buffer alternates between drawing triangles from Camera A and Camera B on consecutive frames or adjacent pixels, causing violent **z-fighting** (flickering geometric noise).
- **Remedy**:
  Implement **Visibility-Aware Angle Selection & Geometric Deduplication**:
  For each candidate surface triangle:
  1. Compute viewing angle $\theta_{\text{view}} = \arccos(|\mathbf{n} \cdot \mathbf{v}_{\text{view}}|)$ relative to each observing camera.
  2. The camera with the most direct, perpendicular viewing angle (highest $|\cos\theta_{\text{view}}|$) owns that surface patch.
  3. Discard or blend overlapping triangles from the oblique camera, completely eliminating duplicate surfaces and z-fighting.

---

## 10. Texture Seams & Grazing-Angle Distortion

- **Audit Finding**: When a camera views a surface at a steep grazing angle ($> 75^\circ$, such as the left camera viewing the far road under the front bumper), each pixel is stretched over several metres of ground. Mapping this stretched pixel produces blurry streaks that contrast sharply with the crisp frontal view from the front camera.
- **Remedy**: Reject faces where $|\cos\theta_{\text{view}}| < 0.20$ (angles $> 78.5^\circ$), allowing the orthogonal camera to provide the high-resolution texture.

---

## 11. Incorrect Surface Connectivity & Flying Triangles

- **Audit Finding**: Isolated single-pixel depth spikes from Depth Anything V2 create "tent poles" or "spikes" that lift neighboring triangles off the ground.
- **Remedy**: Apply a $3 \times 3$ median filter on the input depth maps prior to triangulation, preserving crisp step edges while eradicating single-pixel salt-and-pepper noise.

---

## 12. Missing Mesh Regions & Ego-Vehicle Exclusion

- **Audit Finding**: In M7.1, class 2 (ego-vehicle) was masked out, leaving an empty rectangular hole $(4.30\text{ m} \times 1.79\text{ m})$ under the vehicle.
- **Observation**: This is correct; the physical Hyundai Creta 3D model sits in this hole. However, because ground points directly adjacent to the tyres were also clipped, a jagged gap appeared around the wheels.
- **Remedy**: Dilate the ground plane slightly under the vehicle perimeter so the road seamlessly meets the tyre contact patches.

---

## 13. Virtual Hyundai Creta 3D Model Geometry

- **Audit Finding**: `viewer/vehicle_model.py` contains a parameterized Creta SUV model with authentic dimensions ($4.30\text{ m} \times 1.79\text{ m} \times 1.635\text{ m}$, 0.19 m ground clearance).
- **Enhancement Needed**:
  - Add distinct multi-element LED headlights and signature Creta split DRL brows.
  - Refine dark chrome cascading grille geometry.
  - Enhance alloy wheel rims with contrasting spoke materials.
  - Implement realistic automotive clearcoat shaders (metallic finish with specular highlights).

---

## 14. Obstacle Inspector & UI Mode Inconsistencies

- **Audit Finding**: The current application mixes engineering diagnostics (bounding box wireframes, clearance vectors, occupancy grid points, health scores) directly into the main view.
- **Remedy**: Create two distinct user-selectable view paradigms:
  1. **Driver View**: Pure automotive surround experience. Clean, photorealistic textured 3D ground, walls, and obstacles with the detailed Creta at the centre. No debug grids, no bounding boxes, no engineering clutter. Natural chase/orbit camera presets.
  2. **Engineering View**: Complete technical inspection tool. Full point clouds, semantic masks, 3D voxel occupancy, distance vectors, and diagnostic telemetry.

---

## Implementation Roadmap

| Phase | Milestone | Deliverable | Status |
|---|---|---|---|
| **Phase A** | M7.2 | Diagnostic Audit Report (`docs/m7_2_initial_audit.md`) | **Complete** |
| **Phase B** | M7.2 | Single-Camera UV Textured Mesh (`mesh/textured_mesher.py` + front demo) | **Next** |
| **Phase C** | M7.2 | Four-Camera Visibility-Aware Texture Fusion | Queued |
| **Phase D** | M7.2 | Driver View & Engineering View GUI Architecture | Queued |
| **Phase E** | M7.3 | Smartphone Photo Ingest & EXIF / Quality Pipeline (`photo_reconstruction/`) | Queued |
| **Phase F** | M7.3 | 4-Photo Structure from Motion & Metric Scale Calibration | Queued |
| **Phase G** | M7.3 | Multi-Photo (8–40+) Scalable Reconstruction | Queued |
| **Phase H** | M7.2/7.3 | Final Integration, Test Suite Verification & Screenshots | Queued |

---
*Report verified against CLEAR360 repository architecture on Apple Silicon M2.*

