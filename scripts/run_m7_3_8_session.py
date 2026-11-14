"""
scripts/run_m7_3_8_session.py — Execution script for target session session_066_m7_3_8
=====================================================================================

Executes the end-to-end M7.3.8 pipeline:
  1. Data-driven canonical scene alignment (zero hardcoded bounding boxes)
  2. Multi-model COLMAP audit & connected-component graph analysis
  3. Ground-anchored coordinate system (Z=0 datum, X=0 bilateral symmetry)
  4. Continuous regular grid background wall (zero void, zero crescent bowing)
  5. Continuous ground pavement plane
  6. Taubin-smoothed fountain relief (eliminates spiky cornflake triangles)
  7. Visibility-aware texture atlas (unseen rear faces mapped to neutral gray #8c8c8c)
  8. Authentically baked vertex colors in scene_mesh.ply
"""

import sys
import shutil
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_reconstruction.pipeline import PhoneReconstructionPipeline

def main():
    session_id = "session_066_m7_3_8"
    src_session = ROOT / "data" / "phone_reconstruction" / "session_065_m7_3_7"
    target_session = ROOT / "data" / "phone_reconstruction" / session_id

    # Create target session directory and pre-seed depth cache
    target_depth_cache = target_session / "depth_cache"
    target_depth_cache.mkdir(parents=True, exist_ok=True)

    src_depth_cache = src_session / "depth_cache"
    if src_depth_cache.exists():
        for npy_file in src_depth_cache.glob("*.npy"):
            target_npy = target_depth_cache / npy_file.name
            if not target_npy.exists():
                shutil.copy2(npy_file, target_npy)
                print(f"Pre-seeded cached depth: {npy_file.name}")

    image_paths = [
        src_session / "original_images" / f"fountain_000{i}.jpg"
        for i in range(3, 9)
    ]

    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Source image not found: {p}")

    print(f"Starting M7.3.8 Photogrammetry Pipeline for {session_id} with {len(image_paths)} images...")

    def progress_callback(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}", flush=True)

    pipeline = PhoneReconstructionPipeline(data_root=ROOT / "data", quality_preset="high")
    result = pipeline.run_pipeline(
        image_paths=image_paths,
        session_id=session_id,
        progress_cb=progress_callback,
    )

    print("\n--- Pipeline Result ---")
    print(f"Success: {result.get('success', False)}")
    print(f"Mesh OBJ: {result.get('mesh_obj_path')}")
    print(f"Mesh PLY: {result.get('mesh_ply_path')}")

    # Copy the new session to CAR360 mirror as well
    car360_target = Path("/Users/vudumulasakethreddy/antygide/CAR360/data/phone_reconstruction") / session_id
    if car360_target.exists():
        shutil.rmtree(car360_target)
    shutil.copytree(target_session, car360_target)
    print(f"Synced session to CAR360 mirror: {car360_target}")

if __name__ == "__main__":
    main()

