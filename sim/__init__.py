"""
sim — Blender simulation layer for CLEAR360
============================================

This package generates synthetic ground-truth data (RGB frames, depth maps,
world-space position maps, and semantic segmentation masks) for the CLEAR360
real-time 3D automotive surroundings reconstruction pipeline.

Sub-modules
-----------
constants      Physical constants, camera specs, semantic class map.
vehicle_model  Procedural Hyundai-Creta-approximate SUV in Blender.
environment    Indian parking lot environment + lighting + obstacle animation.
camera_rig     Four-camera surround rig + OpenCV extrinsics export.
export_frames  Top-level Blender script: build scene, render all frames,
               write cameras.yaml.

Coordinate conventions (DO NOT change without updating cameras.yaml):
  World  : X=right, Y=forward (travel direction), Z=up  (Blender default)
  Camera : OpenCV  — x=right-in-image, y=down-in-image, z=into-scene

Entry point:
  blender --background --python sim/export_frames.py -- \\
      --output /path/to/CAR360 --frames 20
"""
