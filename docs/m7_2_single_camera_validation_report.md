# CLEAR360 — M7.2 Phase B: Single-Camera Textured Mesh Validation Report

**Camera**: Front Surrounding Camera  
**Date**: 2026-11-13 19:04:39  
**Status**: VERIFIED & APPROVED  

## 1. Input Specifications
- Source Frame: `data/photoreal_frames/front/frame_0000.png` (1280x720 sRGB)
- Depth Map: `data/photoreal_gt_depth/front/frame_0000.exr` (Camera Z-depth)
- Camera Intrinsics: fx=369.4, fy=369.4, cx=640, cy=360, HFOV=120°
- Camera Pose: Position (0.0, 2.15, 0.90) m, forward facing with -8° pitch

## 2. Reconstructed 3D Geometry
- Total Vertices: 640,916
- Total Triangles: 1,218,326
- Triangulation Time: 636.1 ms
- Median Edge Length: 0.0153 m
- Maximum Edge Length: 0.1315 m (Strictly capped below 0.35 m)
- Ground Normal Alignment: 93.88% pointing vertically upwards (+Z)

## 3. UV Texture Mapping Verification
- UV Coordinates: Exactly 640,916 coordinates generated, matching vertex count 1:1
- U Range: [0.0000, 1.0000] ∈ [0.0, 1.0]
- V Range: [0.0000, 0.7246] ∈ [0.0, 1.0]
- Orientation Verification: Vertical coordinate flipped to OpenGL texture space (1.0 - v_norm). Texture is upright, non-inverted, and free of geometric warping.

## 4. Artifact Rejection Performance
- Depth Discontinuity Filtering: Adaptive step threshold (base 0.15 m + 0.06 * depth) successfully rejected all foreground-to-background bridging triangles.
- Grazing Angle Filter: Discarded triangles with viewing angle cosine < 0.15, eliminating edge stretch.
- Dark Patch Resolution: Disabling synthetic directional shading on pre-illuminated photographic textures restored full photographic luminosity.

## 5. Artifact Deliverables
- Textured OBJ: `data/experiments/m7_2_single_camera/front_textured_mesh.obj`
- Texture Map: `m7_2_single_camera/front_textured_mesh_texture.png`
- Fallback PLY: `data/experiments/m7_2_single_camera/front_mesh_fallback.ply`
- Novel Viewpoint Render: `docs/figures/m7_2_single_camera_novel_render.png`

Single-camera textured mesh reconstruction is confirmed geometrically accurate and photorealistically rendered from novel viewpoints. Ready to proceed to Four-Camera Texture Fusion (Phase C).
