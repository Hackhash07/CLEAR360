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
        weights_path: Optional[Path] = None,
        max_depth: float = 80.0,
        timeout_s: int = 600,
        raise_on_error: bool = False,
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
        manifest = {
            "root": str(project_root),
            "output_dir": str(c_dir),
            "items": items,
            "weights_path": str(Path(weights_path).expanduser()) if weights_path is not None else None,
            "max_depth": float(max_depth),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

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

estimator = DepthEstimator(
    weights_path=data.get("weights_path"),
    max_depth=float(data.get("max_depth", 80.0)),
)

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
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_s)),
            )
            if proc.returncode != 0 or "ISOLATED_DEPTH_SUCCESS" not in proc.stdout:
                detail = (proc.stderr or proc.stdout or "No worker output.")[-5000:]
                if raise_on_error:
                    raise RuntimeError(f"Isolated depth inference failed:\\n{detail}")
                return {}
        except subprocess.TimeoutExpired as exc:
            if raise_on_error:
                raise RuntimeError(
                    f"Depth inference exceeded the {timeout_s}-second timeout."
                ) from exc
            return {}
        except Exception:
            if raise_on_error:
                raise
            return {}

        results = {}
        for item in items:
            name = item["name"]
            npy_path = c_dir / f"{Path(name).stem}_depth.npy"
            if npy_path.exists():
                results[name] = np.load(str(npy_path))

        return results

