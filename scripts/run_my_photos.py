"""Run CLEAR360's own pipeline on real photos at HIGH quality.
python scripts/run_my_photos.py --photos sample_photos [--ref 2.61] [--session session_100]
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from photo_reconstruction.pipeline import PhoneReconstructionPipeline
from run_phone_photogrammetry_experiment import render_reconstruction_offscreen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="sample_photos")
    ap.add_argument("--ref", type=float, default=None, help="known real distance in metres")
    ap.add_argument("--preset", default="high")
    ap.add_argument("--session", default="session_100")
    a = ap.parse_args()

    folder = Path(a.photos)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    photos = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    if len(photos) < 4:
        raise SystemExit(f"Need at least 4 photos in {folder}")
    print(f"{len(photos)} photos | preset={a.preset} | session={a.session}")

    pipe = PhoneReconstructionPipeline(data_root=PROJECT_ROOT / "data", quality_preset=a.preset)
    res = pipe.run_pipeline(
        image_paths=photos,
        session_id=a.session,
        reference_distance_m=a.ref,
        progress_cb=lambda pct, msg: print(f"[{pct:02d}%] {msg}"),
    )

    print("\nsuccess:", res.get("success"))
    print("cameras registered:", len(res.get("camera_poses", {})), "/", len(photos))
    print("sparse:", res.get("sparse_pts_count", 0), "| dense:", res.get("dense_pts_count", 0),
          "| triangles:", res.get("mesh_triangles_count", 0))
    print("\n----- REPORT -----")
    print(json.dumps(res.get("report", {}), indent=2, default=str))

    mesh = res.get("mesh_ply")
    if res.get("success") and mesh and Path(mesh).exists():
        out = PROJECT_ROOT / "docs" / "figures" / f"{a.session}_real.png"
        render_reconstruction_offscreen(Path(mesh), res["camera_poses"], out,
                                        title=f"CLEAR360 real photos ({a.session})")


if __name__ == "__main__":
    main()
