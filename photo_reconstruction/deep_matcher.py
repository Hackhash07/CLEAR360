"""
photo_reconstruction/deep_matcher.py — Deep Learned Feature Matching with LightGlue
=====================================================================================

Provides neural keypoint extraction and matching using LightGlue (SuperPoint, ALIKED, SIFT).
Significantly boosts feature correspondences on low-texture surfaces (plaster walls, dorm rooms)
and handles extreme wide-angle baselines where classical SIFT fails.

Runs in process-isolated execution on macOS to avoid OpenMP duplicate runtime conflicts with
Open3D and pycolmap.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class DeepMatcher:
    """Neural feature matcher powered by LightGlue with process-isolation."""

    def __init__(
        self,
        feature_type: str = "superpoint",
        max_keypoints: int = 2048,
        device: str = "cpu",
        isolated: bool = True,
    ):
        """
        Parameters
        ----------
        feature_type : str ('superpoint', 'aliked', 'sift')
        max_keypoints : int (default 2048)
        device : str (default 'cpu' to avoid PyTorch MPS strided tensor bugs on macOS)
        isolated : bool (runs via subprocess to avoid OpenMP conflicts with Open3D)
        """
        self.feature_type = feature_type.lower()
        self.max_keypoints = int(max_keypoints)
        self.device = str(device)
        self.isolated = isolated

    @property
    def is_available(self) -> bool:
        """Check whether LightGlue and PyTorch are installed in the environment."""
        has_lg = importlib.util.find_spec("lightglue") is not None
        has_torch = importlib.util.find_spec("torch") is not None
        return has_lg and has_torch

    def match_pair(
        self,
        img_path0: Path,
        img_path1: Path,
        resize: int = 1200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and compute verified matches between two images.

        Parameters
        ----------
        img_path0 : Path
        img_path1 : Path
        resize : int (max dimension for image scaling during extraction)

        Returns
        -------
        kpts0 : (M, 2) float32 coordinates in img0
        kpts1 : (M, 2) float32 coordinates in img1
        confidence : (M,) float32 match confidence scores
        """
        p0 = Path(img_path0)
        p1 = Path(img_path1)

        if not self.is_available or not p0.is_file() or not p1.is_file():
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        if self.isolated:
            return self._match_pair_isolated(p0, p1, resize)
        else:
            return self._match_pair_inproc(p0, p1, resize)

    def _match_pair_isolated(
        self,
        p0: Path,
        p1: Path,
        resize: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run LightGlue in an isolated Python process to avoid OpenMP library collisions."""
        worker_code = """
import os, sys
from pathlib import Path
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, ALIKED, SIFT
from lightglue.utils import load_image, rbd

p0 = Path(sys.argv[1])
p1 = Path(sys.argv[2])
out_npz = Path(sys.argv[3])
feat_type = sys.argv[4].lower()
max_kpts = int(sys.argv[5])
resize_px = int(sys.argv[6])
dev_str = sys.argv[7]

device = torch.device(dev_str)

if feat_type == "superpoint":
    extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
elif feat_type == "aliked":
    extractor = ALIKED(max_num_keypoints=max_kpts).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
else:
    extractor = SIFT(max_num_keypoints=max_kpts).eval().to(device)
    matcher = LightGlue(features="sift").eval().to(device)

im0 = load_image(p0, resize=resize_px).to(device)
im1 = load_image(p1, resize=resize_px).to(device)

with torch.no_grad():
    feats0 = extractor.extract(im0)
    feats1 = extractor.extract(im1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

k0 = feats0["keypoints"].cpu().numpy()
k1 = feats1["keypoints"].cpu().numpy()
m = matches01["matches"].cpu().numpy()
sc = matches01["scores"].cpu().numpy() if "scores" in matches01 else np.ones(len(m), dtype=np.float32)

if len(m) > 0:
    kpts0 = k0[m[:, 0]].astype(np.float32)
    kpts1 = k1[m[:, 1]].astype(np.float32)
    scores = sc.astype(np.float32)
else:
    kpts0 = np.zeros((0, 2), dtype=np.float32)
    kpts1 = np.zeros((0, 2), dtype=np.float32)
    scores = np.zeros(0, dtype=np.float32)

np.savez(out_npz, kpts0=kpts0, kpts1=kpts1, scores=scores)
print("ISOLATED_MATCH_SUCCESS")
"""
        with tempfile.TemporaryDirectory() as td:
            out_npz = Path(td) / "matches.npz"
            env = os.environ.copy()
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            cmd = [
                sys.executable,
                "-c",
                worker_code,
                str(p0),
                str(p1),
                str(out_npz),
                self.feature_type,
                str(self.max_keypoints),
                str(resize),
                self.device,
            ]

            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
                if "ISOLATED_MATCH_SUCCESS" in proc.stdout and out_npz.exists():
                    data = np.load(str(out_npz))
                    return data["kpts0"], data["kpts1"], data["scores"]
            except Exception:
                pass

        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    def _match_pair_inproc(
        self,
        p0: Path,
        p1: Path,
        resize: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """In-process execution (used when process isolation is disabled)."""
        try:
            import torch
            from lightglue import LightGlue, SuperPoint, ALIKED, SIFT
            from lightglue.utils import load_image, rbd

            device = torch.device(self.device)
            if self.feature_type == "superpoint":
                extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(device)
                matcher = LightGlue(features="superpoint").eval().to(device)
            elif self.feature_type == "aliked":
                extractor = ALIKED(max_num_keypoints=self.max_keypoints).eval().to(device)
                matcher = LightGlue(features="aliked").eval().to(device)
            else:
                extractor = SIFT(max_num_keypoints=self.max_keypoints).eval().to(device)
                matcher = LightGlue(features="sift").eval().to(device)

            im0 = load_image(p0, resize=resize).to(device)
            im1 = load_image(p1, resize=resize).to(device)

            with torch.no_grad():
                feats0 = extractor.extract(im0)
                feats1 = extractor.extract(im1)
                matches01 = matcher({"image0": feats0, "image1": feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

            k0 = feats0["keypoints"].cpu().numpy()
            k1 = feats1["keypoints"].cpu().numpy()
            m = matches01["matches"].cpu().numpy()
            sc = matches01["scores"].cpu().numpy() if "scores" in matches01 else np.ones(len(m), dtype=np.float32)

            if len(m) > 0:
                return k0[m[:, 0]].astype(np.float32), k1[m[:, 1]].astype(np.float32), sc.astype(np.float32)
        except Exception:
            pass

        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
