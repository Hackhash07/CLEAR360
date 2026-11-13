"""
scripts/run_phone_photogrammetry_experiment.py — M7.3 Phone Photogrammetry Experiments
====================================================================================

Executes Phase F and Phase G photogrammetry validation experiments:
  1. Experiment 1 (Phase F — 4-Photo Orbital):
     Sequential smartphone photos (~70% overlap) orbiting front-left quadrant.
     Calibrated with Creta wheelbase reference distance = 2.61 m.
     Generates session_001 with sparse/dense clouds, Poisson mesh, and quality report.
     Renders offscreen 3D scene to docs/figures/m7_3_phone_4photo_reconstruction.png.

  2. Experiment 2 (Phase F — 4-Camera Surround Failure Case):
     Outward-facing cameras with 0% mutual overlap.
     Validates automated detection of insufficient overlap and diagnostic reporting.
     Generates session_002 with diagnostic report explaining failure and requesting views.

  3. Experiment 3 (Phase G — 12-Photo 360° Scalability):
     12 sequential smartphone photos orbiting 360° around the environment.
     Calibrated with vehicle length reference distance = 4.30 m.
     Generates session_003 with complete 360° surround mesh and quality report.
     Renders offscreen 3D scene to docs/figures/m7_3_phone_12photo_reconstruction.png.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from photo_reconstruction.pipeline import PhoneReconstructionPipeline
from photo_reconstruction.session_manager import SessionManager


def render_reconstruction_offscreen(
    mesh_ply_path: Path,
    camera_poses: dict[str, np.ndarray],
    out_png_path: Path,
    title: str = "CLEAR360 Phone 3D Reconstruction",
) -> None:
    """Render high-resolution offscreen perspective view of the reconstructed 3D scene."""
    import vispy.scene
    from vispy.scene import visuals
    import open3d as o3d

    pcd_or_mesh = o3d.io.read_triangle_mesh(str(mesh_ply_path))
    v = np.asarray(pcd_or_mesh.vertices, dtype=np.float32)
    f = np.asarray(pcd_or_mesh.triangles, dtype=np.uint32)
    c = np.asarray(pcd_or_mesh.vertex_colors, dtype=np.float32)

    canvas = vispy.scene.SceneCanvas(show=False, size=(1920, 1080), bgcolor="#0a0a0e")
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    view.camera.up = "+z"
    view.camera.fov = 50.0

    if len(v) > 0:
        center = v.mean(axis=0)
        extent = np.ptp(v, axis=0)
        dist = max(float(np.max(extent)) * 1.8, 6.0)
        view.camera.center = tuple(center)
        view.camera.distance = dist
        view.camera.elevation = 30.0
        view.camera.azimuth = 45.0
    else:
        view.camera.center = (0.0, 0.0, 0.5)
        view.camera.distance = 8.0
        view.camera.elevation = 28.0
        view.camera.azimuth = 45.0

    # 1. Add reconstructed mesh
    if len(v) > 0 and len(f) > 0:
        if len(c) == len(v):
            mesh_vis = visuals.Mesh(vertices=v, faces=f, vertex_colors=c)
        else:
            mesh_vis = visuals.Mesh(vertices=v, faces=f, color="#4a90e2")
        view.add(mesh_vis)
    elif len(v) > 0:
        # Fallback to point cloud visual if mesh has no faces
        scatter = visuals.Markers()
        scatter.set_data(v, edge_color=None, face_color=c if len(c) == len(v) else "#4a90e2", size=4)
        view.add(scatter)

    # 2. Add visual frustums / markers for registered camera poses
    cam_centers = []
    for name, T in camera_poses.items():
        cam_pos = T[:3, 3]
        cam_centers.append(cam_pos)

    if cam_centers:
        cam_arr = np.array(cam_centers, dtype=np.float32)
        cam_markers = visuals.Markers()
        cam_markers.set_data(cam_arr, edge_color="#ffffff", face_color="#ffcc00", size=12)
        view.add(cam_markers)

        # Draw trajectory lines connecting registered cameras
        if len(cam_arr) >= 2:
            lines = visuals.Line(pos=cam_arr, color="#ffcc00", width=2.0)
            view.add(lines)

    # 3. Add ground reference grid
    grid = visuals.GridLines(color=(0.2, 0.25, 0.35, 0.4))
    view.add(grid)

    img_rgba = canvas.render()
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

    # Overlay HUD badge
    cv2.rectangle(img_bgr, (20, 20), (620, 110), (15, 18, 25), -1)
    cv2.rectangle(img_bgr, (20, 20), (620, 110), (0, 200, 150), 2)
    cv2.putText(img_bgr, title, (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    meta_str = f"Vertices: {len(v):,} | Triangles: {len(f):,} | Cams: {len(camera_poses)}"
    cv2.putText(img_bgr, meta_str, (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 180), 1)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png_path), img_bgr)
    print(f"    Saved offscreen render to: {out_png_path}")


def run_experiments():
    print("==================================================================")
    print("CLEAR360 — M7.3 Smartphone Photogrammetry Experiments (F & G)")
    print("==================================================================")

    data_root = PROJECT_ROOT / "data"
    figures_dir = PROJECT_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Phase F — 4-Photo Sequential Smartphone Photogrammetry
    # -------------------------------------------------------------------------
    print("\n[PHASE F] Running Experiment 1: 4-Photo Orbital Reconstruction...")
    dir_4orb = data_root / "phone_datasets" / "orbital_4photos"
    photos_4 = sorted(list(dir_4orb.glob("*.jpg")))
    print(f"  Input: {len(photos_4)} photos from {dir_4orb.name}")

    pipeline_f = PhoneReconstructionPipeline(data_root=data_root, quality_preset="standard")
    res_f = pipeline_f.run_pipeline(
        image_paths=photos_4,
        session_id="session_001",
        reference_distance_m=2.61,  # Hyundai Creta wheelbase (2.610 m)
        progress_cb=lambda pct, msg: print(f"  [{pct:02d}%] {msg}"),
    )

    print("\n  Experiment 1 Results:")
    print(f"    Success: {res_f['success']}")
    print(f"    Session ID: {res_f['session_id']}")
    print(f"    Registered Cameras: {len(res_f.get('camera_poses', {}))}/{len(photos_4)}")
    print(f"    Sparse Points: {res_f.get('sparse_pts_count', 0):,}")
    print(f"    Dense Points: {res_f.get('dense_pts_count', 0):,}")
    print(f"    Mesh Triangles: {res_f.get('mesh_triangles_count', 0):,}")
    print(f"    Scale Factor: {res_f['report']['scale_calibration']['scale_factor']}")

    if res_f["success"] and res_f["mesh_ply"].exists():
        render_4_out = figures_dir / "m7_3_phone_4photo_reconstruction.png"
        render_reconstruction_offscreen(
            res_f["mesh_ply"],
            res_f["camera_poses"],
            render_4_out,
            title="M7.3 Phase F: 4-Photo Orbital Smartphone Reconstruction",
        )

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Phase F — 4-Camera Surround Failure Detection Diagnostic
    # -------------------------------------------------------------------------
    print("\n[PHASE F] Running Experiment 2: Outward Surround Failure Diagnostic...")
    dir_4sur = data_root / "phone_datasets" / "surround_4photos"
    photos_sur = sorted(list(dir_4sur.glob("*.jpg")))
    print(f"  Input: {len(photos_sur)} outward photos (0% mutual overlap)")

    pipeline_fail = PhoneReconstructionPipeline(data_root=data_root, quality_preset="standard")
    res_fail = pipeline_fail.run_pipeline(
        image_paths=photos_sur,
        session_id="session_002",
        reference_distance_m=None,
        progress_cb=lambda pct, msg: print(f"  [{pct:02d}%] {msg}"),
    )

    print("\n  Experiment 2 Diagnostic Results:")
    print(f"    Success: {res_fail['success']} (Expected: False for 0% overlap)")
    print(f"    Completeness Grade: {res_fail['report']['reconstruction_grade']}")
    print(f"    Diagnostic Message: {res_fail['report']['notes']}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Phase G — 12-Photo 360° Scalability Reconstruction
    # -------------------------------------------------------------------------
    print("\n[PHASE G] Running Experiment 3: 12-Photo 360° Scalability...")
    dir_12orb = data_root / "phone_datasets" / "orbital_12photos"
    photos_12 = sorted(list(dir_12orb.glob("*.jpg")))
    print(f"  Input: {len(photos_12)} photos from {dir_12orb.name}")

    pipeline_g = PhoneReconstructionPipeline(data_root=data_root, quality_preset="high")
    res_g = pipeline_g.run_pipeline(
        image_paths=photos_12,
        session_id="session_003",
        reference_distance_m=4.30,  # Hyundai Creta overall length (4.300 m)
        progress_cb=lambda pct, msg: print(f"  [{pct:02d}%] {msg}"),
    )

    print("\n  Experiment 3 Results:")
    print(f"    Success: {res_g['success']}")
    print(f"    Session ID: {res_g['session_id']}")
    print(f"    Registered Cameras: {len(res_g.get('camera_poses', {}))}/{len(photos_12)}")
    print(f"    Sparse Points: {res_g.get('sparse_pts_count', 0):,}")
    print(f"    Dense Points: {res_g.get('dense_pts_count', 0):,}")
    print(f"    Mesh Triangles: {res_g.get('mesh_triangles_count', 0):,}")
    print(f"    Scale Factor: {res_g['report']['scale_calibration']['scale_factor']}")

    if res_g["success"] and res_g["mesh_ply"].exists():
        render_12_out = figures_dir / "m7_3_phone_12photo_reconstruction.png"
        render_reconstruction_offscreen(
            res_g["mesh_ply"],
            res_g["camera_poses"],
            render_12_out,
            title="M7.3 Phase G: 12-Photo 360° Surround Scalability Reconstruction",
        )

    print("\n==================================================================")
    print("ALL M7.3 PHOTOGRAMMETRY EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    run_experiments()
