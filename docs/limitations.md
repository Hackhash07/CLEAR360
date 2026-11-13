# CLEAR360 — Milestone M1 Limitations and Technical Assumptions

This document catalogs known constraints, design trade-offs, and synthetic dataset assumptions present in Milestone 1 (M1). These limitations provide important context when evaluating downstream depth estimation (M2), point-cloud fusion (M3–M4), and obstacle detection (M5–M6).

---

## 1. Static Scene Within Each Rendered Frame
- **Constraint**: Synthetic camera frames are rendered with zero motion blur and instantaneous shutter time.
- **Impact**: Real vehicle cameras suffer from motion blur during movement and rolling-shutter artifacts over bumpy surfaces. In M1, dynamic objects move between frames, but each individual frame is instantaneously crisp.

## 2. Ideal Pinhole Projection (No Lens Distortion)
- **Constraint**: Synthetic simulation uses Blender's ideal perspective pinhole camera model ($k_1 = k_2 = k_3 = p_1 = p_2 = 0$).
- **Impact**: Real automotive surround-view cameras employ ultra-wide / fisheye lenses exhibiting heavy optical distortion. Downstream modules must undistort real-world imagery before feeding into the CLEAR360 pinhole pipeline.

## 3. Absence of Optical Vignetting, Chromatic Aberration, and Sensor Noise
- **Constraint**: Synthetic renders do not simulate lens vignetting (light falloff at image periphery), Bayer demosaicing artifacts, chromatic aberration, or photon shot noise.
- **Impact**: AI models trained or evaluated solely on synthetic images may experience domain gap when transferred to real-world automotive camera sensors.

## 4. Ground-Truth Depth Scale: Perfect Metric vs. AI Relative Depth
- **Constraint**: Blender ground-truth depth (`gt_depth/`) is absolute metric distance in metres. Monocular depth models (e.g., Depth Anything V2) estimate relative or affine-invariant depth up to an unknown scale and shift.
- **Impact**: Direct comparison of AI depth to Blender GT requires scale-and-shift alignment (or metric fine-tuning) during validation.

## 5. Domain Shift for Depth Anything V2
- **Constraint**: Off-the-shelf Depth Anything V2 (ViT-Small) was predominantly trained on outdoor public datasets (KITTI, NYUv2, MegaDepth, diverse web images).
- **Impact**: Indoor covered parking garages with repetitive textures, artificial ceiling lamps, and low-contrast concrete walls may exhibit higher relative error than outdoor road scenes.

## 6. Constant Environmental Lighting Across Rendered Frames
- **Constraint**: The lighting setup in M1 uses static ambient and directional lighting. There are no sudden changes in exposure, headlight glare, or harsh exterior sunlight transitions.
- **Impact**: Automatic exposure and dynamic range challenges common in real garage entrances are not tested in M1.

## 7. Limited Sequence Length (20 Frames)
- **Constraint**: M1 exports a 20-frame demonstration sequence designed for unit testing and pipeline prototyping.
- **Impact**: 20 frames is sufficient for verifying geometry, projection math, and fusion algorithms, but represents a small benchmark. Future milestones will scale sequence duration and scene variety.

## 8. Renderer Requirement for Position Pass: Cycles vs. EEVEE
- **Constraint**: Blender's `Vector` / `Position` render pass requires the **Cycles** engine. The EEVEE real-time engine does not natively support the position pass node setup in headless mode without workarounds.
- **Impact**: If rendering in fast EEVEE mode, `gt_position` maps may be omitted or generated via depth unprojection; the validation scripts treat missing position passes as non-fatal warnings.

## 9. Object `pass_index` Semantic Granularity
- **Constraint**: Semantic segmentation is driven by Blender object-level `pass_index` values ($0$ through $7$).
- **Impact**: Segmentation is instance/object-level rather than part-level. For example, wheels and vehicle windows share the `vehicle_ego` index ($2$) rather than separate sub-classes.

## 10. Simplified Geometric Representation of Host Vehicle
- **Constraint**: The vehicle model in M1 is constructed from primitive boxes, cylinders, and bevel modifiers approximating the dimensions of a Hyundai Creta 2022.
- **Impact**: Provides accurate spatial boundary boxes and occlusion silhouettes for camera placement validation, but lacks high-fidelity body curves and panel gaps until detailed CAD/OBJ assets are introduced in later milestones.
