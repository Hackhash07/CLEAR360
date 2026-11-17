"""
photo_reconstruction/depth_helper.py — Process-Isolated Metric Depth Estimation
=============================================================================

Runs Depth Anything V2 in an isolated Python process to avoid OpenMP runtime
library conflicts with Open3D and pycolmap on macOS ARM64.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class IsolatedDepthEstimator:
    """Estimates dense metric depth maps using an isolated Python subprocess."""

    @classmethod
    def estimate_depths_for_images(
        cls,
        image_paths: List[Path],
        cache_dir: Path,
        project_root: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate dense metric depth for each image in an isolated process.

        Returns
        -------
        Dict[str, np.ndarray] : mapping of filename -> (H, W) float32 metric depth in metres.
        """
        if not image_paths:
            return {}

        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)

        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        manifest_path = c_dir / "depth_request_manifest.json"
        items = [{"path": str(p), "name": p.name} for p in image_paths if p.is_file()]
        with open(manifest_path, "w") as f:
            json.dump({"root": str(project_root), "output_dir": str(c_dir), "items": items}, f)

        worker_code = """
import sys
import json
from pathlib import Path
import cv2
import numpy as np

manifest_file = Path(sys.argv[1])
with open(manifest_file, "r") as f:
    data = json.load(f)

sys.path.insert(0, data["root"])
from depth.estimator import DepthEstimator

out_dir = Path(data["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

estimator = DepthEstimator()

for item in data["items"]:
    img_p = Path(item["path"])
    img_name = item["name"]
    out_npy = out_dir / f"{Path(img_name).stem}_depth.npy"
    if out_npy.exists():
        continue
    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    depth_map, _ = estimator.estimate_depth(img_rgb)
    np.save(str(out_npy), depth_map.astype(np.float32))

print("ISOLATED_DEPTH_SUCCESS")
"""

        try:
            cmd = [sys.executable, "-c", worker_code, str(manifest_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if "ISOLATED_DEPTH_SUCCESS" not in proc.stdout:
                # Subprocess may have failed
                return {}
        except Exception:
            return {}

        results = {}
        for item in items:
            name = item["name"]
            npy_path = c_dir / f"{Path(name).stem}_depth.npy"
            if npy_path.exists():
                # Memory-map cached depth maps: downstream stages access
                # them per view, so eagerly loading every full-resolution map
                # would unnecessarily duplicate the on-disk cache in RAM.
                results[name] = np.load(
                    str(npy_path),
                    mmap_mode="r",
                    allow_pickle=False,
                )

        return results

