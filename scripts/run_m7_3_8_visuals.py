"""
scripts/run_m7_3_8_visuals.py — Script to generate all 12 M7.3.8 verification figures
====================================================================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_reconstruction.m7_3_8_visual_generator import M738VisualGenerator

def main():
    session_id = "session_066_m7_3_8"
    session_dir = ROOT / "data" / "phone_reconstruction" / session_id
    artifacts_dir = Path("/Users/vudumulasakethreddy/.gemini/antigravity/brain/90a700b0-89df-4c80-ac62-2781194ba2b5")

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    print(f"Generating 12 M7.3.8 verification figures from {session_dir}...")
    generator = M738VisualGenerator(session_dir=session_dir, artifacts_dir=artifacts_dir)
    generated = generator.generate_all()

    for p in generated:
        print(f"Generated: {p.name} ({p.stat().st_size:,} bytes)")

    print("\nAll 12 M7.3.8 verification figures successfully generated.")

if __name__ == "__main__":
    main()

