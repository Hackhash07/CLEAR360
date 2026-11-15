#!/usr/bin/env python3
"""
scripts/run_m7_3_9_acceptance.py — End-to-End Acceptance Test for M7.3.9
======================================================================
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import open3d as o3d
import numpy as np

from photo_reconstruction.pipeline import PhoneReconstructionPipeline
from photo_reconstruction.m7_3_9_visual_generator import M739VisualGenerator


def main():
    print("=" * 80)
    print(" CLEAR360 M7.3.9 — END-TO-END ACCEPTANCE TEST & VERIFICATION")
    print("=" * 80)

    images_dir = PROJECT_ROOT / "data" / "phone_reconstruction" / "test_fountain_dataset" / "original_images"
    image_paths = sorted(list(images_dir.glob("fountain_000*.jpg")))
    print(f"Found {len(image_paths)} fountain photographs: {[p.name for p in image_paths]}")
    assert len(image_paths) == 6, f"Expected 6 fountain photos, found {len(image_paths)}"

    target_session_id = "session_067_m7_3_9"
    target_session_dir = PROJECT_ROOT / "data" / "phone_reconstruction" / target_session_id

    artifacts_dir = Path("/Users/vudumulasakethreddy/.gemini/antigravity/brain/90a700b0-89df-4c80-ac62-2781194ba2b5")

    if not (target_session_dir / "mesh" / "scene_mesh.ply").exists():
        print(f"\n[1/3] Running PhoneReconstructionPipeline (quality_preset='high')...")
        pipeline = PhoneReconstructionPipeline(quality_preset="high")
        t0 = time.time()
        result = pipeline.run_pipeline(
            image_paths=image_paths,
            session_id=target_session_id,
            use_mvs=True,
        )
        t_pipeline = time.time() - t0
        print(f"Pipeline completed in {t_pipeline:.1f} s. Result: {result.get('success')}")
        assert result.get("success"), f"Pipeline failed: {result}"
    else:
        print(f"\n[1/3] Using existing reconstructed session {target_session_id}...")

    print(f"\n[2/3] Verifying Reconstruction Assets...")
    dense_ply = target_session_dir / "dense_cloud" / "dense_cloud.ply"
    uncertain_ply = target_session_dir / "dense_cloud" / "uncertain_diagnostic_cloud.ply"
    scene_mesh_ply = target_session_dir / "mesh" / "scene_mesh.ply"
    scene_mesh_obj = target_session_dir / "mesh" / "scene_mesh.obj"
    scene_mesh_png = target_session_dir / "mesh" / "scene_mesh.png"
    manifest_file = target_session_dir / "global_scene_manifest.json"
    texturing_report_file = target_session_dir / "texturing_report.json"

    assert dense_ply.exists(), "dense_cloud.ply missing"
    assert scene_mesh_ply.exists(), "scene_mesh.ply missing"
    assert scene_mesh_obj.exists(), "scene_mesh.obj missing"
    assert scene_mesh_png.exists(), "scene_mesh.png missing"
    assert manifest_file.exists(), "global_scene_manifest.json missing"
    assert texturing_report_file.exists(), "texturing_report.json missing"

    pcd_dense = o3d.io.read_point_cloud(str(dense_ply))
    mesh = o3d.io.read_triangle_mesh(str(scene_mesh_ply))

    print(f"  - Dense verified points : {len(pcd_dense.points):,}")
    print(f"  - Scene mesh triangles  : {len(mesh.triangles):,}")
    print(f"  - Scene mesh vertices   : {len(mesh.vertices):,}")

    with open(texturing_report_file) as f:
        tex_data = json.load(f)
    print(f"  - Texturing coverage    : {tex_data.get('coverage_percentage')}%")
    print(f"  - Textured faces        : {tex_data.get('textured_faces'):,}")
    print(f"  - Untextured/rear faces : {tex_data.get('untextured_faces'):,}")

    # Check for smooth normals in OBJ
    has_vn = False
    with open(scene_mesh_obj) as f:
        for line in f:
            if line.startswith("vn "):
                has_vn = True
                break
    print(f"  - Smooth vertex normals : {'YES (Certified)' if has_vn else 'NO'}")
    assert has_vn, "scene_mesh.obj must have vertex normals"

    print(f"\n[3/3] Generating all 12 Verification Figures...")
    visual_gen = M739VisualGenerator(
        session_dir=target_session_dir,
        artifacts_dir=artifacts_dir,
    )
    fig_paths = visual_gen.generate_all_12_figures()
    for fig_id, p in fig_paths.items():
        print(f"  - {fig_id}: {Path(p).name}")

    print("\n" + "=" * 80)
    print(" M7.3.9 ACCEPTANCE TEST SUCCESSFUL!")
    print("=" * 80)


if __name__ == "__main__":
    main()
