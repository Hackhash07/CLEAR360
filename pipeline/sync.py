"""
CLEAR360 – Frame Synchronisation & Loading Module
===================================================
Provides dataclasses and loaders for multi-camera frame data: RGB images,
ground-truth depth (EXR), world-space position maps (EXR), and semantic masks.

Coordinate Conventions
-----------------------
World / Vehicle frame : X=RIGHT, Y=FORWARD, Z=UP  (vehicle at origin).
Depth convention      : camera Z-depth in metres (NOT Euclidean ray length),
                        stored as float32.

Unprojection formula (for reference – see calibration.py for full detail):
    Given pixel (u, v) and camera-Z depth d (metres):

        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d
        p_cam_hom = [x_cam, y_cam, z_cam, 1.0]
        T_world_cam = np.linalg.inv(T_cam_world)
        p_world = T_world_cam @ p_cam_hom   # → [X, Y, Z, 1]
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional EXR imports
# ---------------------------------------------------------------------------

try:
    import OpenEXR
    import Imath

    _EXR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EXR_AVAILABLE = False
    OpenEXR = None  # type: ignore[assignment]
    Imath = None    # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional OpenCV import
# ---------------------------------------------------------------------------

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]


def _require_exr() -> None:
    """Raise a helpful ImportError if OpenEXR / Imath are not installed."""
    if not _EXR_AVAILABLE:
        raise ImportError(
            "OpenEXR and Imath are required to load EXR depth/position files.\n"
            "Install them with:  pip install OpenEXR Imath"
        )


def _require_cv2() -> None:
    """Raise a helpful ImportError if OpenCV is not installed."""
    if not _CV2_AVAILABLE:
        raise ImportError(
            "OpenCV (cv2) is required to load PNG images.\n"
            "Install it with:  pip install opencv-python"
        )


# ---------------------------------------------------------------------------
# CameraFrameData
# ---------------------------------------------------------------------------


@dataclass
class CameraFrameData:
    """
    All data associated with a single camera at a single frame.

    Attributes
    ----------
    camera_name     : camera identifier ('front', 'rear', 'left', 'right').
    frame_id        : zero-based frame index.
    rgb             : (H, W, 3) uint8 – RGB image (NOT BGR).
    depth           : (H, W) float32 – camera Z-depth in metres.
    position_map    : (H, W, 3) float32 – world XYZ per pixel, or None.
    semantic_mask   : (H, W) uint8 – pass_index values, or None.
    depth_source    : provenance tag, e.g. 'ground_truth_blender' or
                      'ai_depth_anything_v2'.
    """

    camera_name: str
    frame_id: int
    rgb: np.ndarray            # shape (H, W, 3), dtype uint8
    depth: np.ndarray          # shape (H, W),    dtype float32
    position_map: Optional[np.ndarray] = None   # (H, W, 3) float32
    semantic_mask: Optional[np.ndarray] = None  # (H, W) uint8
    depth_source: str = "ground_truth_blender"

    def __repr__(self) -> str:
        h, w = self.depth.shape
        pm = "yes" if self.position_map is not None else "no"
        sm = "yes" if self.semantic_mask is not None else "no"
        return (
            f"CameraFrameData(camera='{self.camera_name}', "
            f"frame={self.frame_id}, shape=({h},{w}), "
            f"position_map={pm}, semantic_mask={sm}, "
            f"depth_source='{self.depth_source}')"
        )


# ---------------------------------------------------------------------------
# FrameSet
# ---------------------------------------------------------------------------


@dataclass
class FrameSet:
    """
    One synchronised 4-camera frame.

    Attributes
    ----------
    frame_id  : zero-based frame index.
    cameras   : dict mapping camera name → CameraFrameData.
    timestamp : optional wall-clock timestamp in seconds.
    """

    frame_id: int
    cameras: dict[str, CameraFrameData] = field(default_factory=dict)
    timestamp: Optional[float] = None

    _REQUIRED_CAMERAS = ("front", "rear", "left", "right")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True if all 4 cameras are present and RGB + depth are non-None."""
        for name in self._REQUIRED_CAMERAS:
            if name not in self.cameras:
                return False
            cfd = self.cameras[name]
            if cfd.rgb is None or cfd.depth is None:
                return False
        return True

    @property
    def camera_names(self) -> list[str]:
        """Sorted list of available camera names."""
        return sorted(self.cameras.keys())

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, camera_name: str) -> CameraFrameData:
        """Return the CameraFrameData for the given camera name.

        Raises KeyError if the camera is not in this FrameSet.
        """
        if camera_name not in self.cameras:
            raise KeyError(
                f"Camera '{camera_name}' not in FrameSet {self.frame_id}. "
                f"Available: {self.camera_names}"
            )
        return self.cameras[camera_name]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the FrameSet and return a list of error strings.

        Checks performed
        ----------------
        1. All 4 required cameras are present.
        2. All RGB arrays have shape (H, W, 3) and dtype uint8.
        3. All depth arrays have shape (H, W) and dtype float32.
        4. All RGB arrays share the same (H, W).
        5. All depth arrays share the same (H, W).
        6. At least 90 % of depth pixels are > 0.
        7. No NaN values in any depth array.
        """
        errors: list[str] = []

        # 1. All cameras present
        for name in self._REQUIRED_CAMERAS:
            if name not in self.cameras:
                errors.append(f"Missing camera: '{name}'")

        present = {n: self.cameras[n] for n in self._REQUIRED_CAMERAS if n in self.cameras}

        # Collect shapes for consistency checks
        rgb_shapes: list[tuple[int, int]] = []
        depth_shapes: list[tuple[int, int]] = []

        for name, cfd in present.items():
            # 2. RGB dtype / ndim
            if cfd.rgb is None:
                errors.append(f"[{name}] RGB is None")
            else:
                if cfd.rgb.dtype != np.uint8:
                    errors.append(
                        f"[{name}] RGB dtype is {cfd.rgb.dtype}, expected uint8"
                    )
                if cfd.rgb.ndim != 3 or cfd.rgb.shape[2] != 3:
                    errors.append(
                        f"[{name}] RGB shape is {cfd.rgb.shape}, expected (H,W,3)"
                    )
                else:
                    rgb_shapes.append((cfd.rgb.shape[0], cfd.rgb.shape[1]))

            # 3. Depth dtype / ndim
            if cfd.depth is None:
                errors.append(f"[{name}] depth is None")
            else:
                if cfd.depth.dtype != np.float32:
                    errors.append(
                        f"[{name}] depth dtype is {cfd.depth.dtype}, expected float32"
                    )
                if cfd.depth.ndim != 2:
                    errors.append(
                        f"[{name}] depth ndim is {cfd.depth.ndim}, expected 2"
                    )
                else:
                    depth_shapes.append((cfd.depth.shape[0], cfd.depth.shape[1]))

                    # 6. ≥ 90 % positive
                    total = cfd.depth.size
                    positive = int(np.sum(cfd.depth > 0))
                    if total > 0 and positive / total < 0.90:
                        pct = 100.0 * positive / total
                        errors.append(
                            f"[{name}] only {pct:.1f}% of depth pixels are > 0 "
                            f"(threshold: 90%)"
                        )

                    # 7. No NaN
                    if np.any(np.isnan(cfd.depth)):
                        errors.append(f"[{name}] depth contains NaN values")

        # 4. Consistent RGB shape
        if len(rgb_shapes) > 1 and len(set(rgb_shapes)) > 1:
            errors.append(f"RGB shapes are inconsistent across cameras: {rgb_shapes}")

        # 5. Consistent depth shape
        if len(depth_shapes) > 1 and len(set(depth_shapes)) > 1:
            errors.append(f"Depth shapes are inconsistent across cameras: {depth_shapes}")

        return errors

    def __repr__(self) -> str:
        return (
            f"FrameSet(frame_id={self.frame_id}, "
            f"cameras={self.camera_names}, "
            f"is_complete={self.is_complete}, "
            f"timestamp={self.timestamp})"
        )


# ---------------------------------------------------------------------------
# FrameSetLoader
# ---------------------------------------------------------------------------


class FrameSetLoader:
    """
    Loads complete 4-camera FrameSets from the data directory produced by the
    CLEAR360 Blender export script (sim/export_frames.py).

    Directory layout expected
    -------------------------
    {data_root}/
        sim_frames/{camera}/frame_{n:04d}.png   – 1280×720 RGB 8-bit PNG
        gt_depth/{camera}/frame_{n:04d}.exr     – float32 EXR, channel 'Z'
        gt_position/{camera}/frame_{n:04d}.exr  – float32 EXR, channels X Y Z
        gt_semantic/{camera}/frame_{n:04d}.png  – uint8 PNG, pixel = pass_index

    Parameters
    ----------
    data_root      : path to the root data directory.
    expected_shape : (H, W) tuple for validation; default (720, 1280).
    """

    CAMERA_NAMES = ["front", "rear", "left", "right"]
    DEPTH_SOURCE_GT = "ground_truth_blender"
    DEPTH_SOURCE_AI = "ai_depth_anything_v2"

    def __init__(
        self,
        data_root: "str | Path",
        expected_shape: tuple[int, int] = (720, 1280),
    ) -> None:
        self._data_root = Path(data_root)
        self._expected_shape = expected_shape  # (H, W)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_frame(
        self,
        frame_id: int,
        load_position: bool = True,
        load_semantic: bool = True,
    ) -> FrameSet:
        """
        Load one complete 4-camera FrameSet for the given frame_id.

        File paths
        ----------
        RGB      : {data_root}/sim_frames/{camera}/frame_{frame_id:04d}.png
        Depth    : {data_root}/gt_depth/{camera}/frame_{frame_id:04d}.exr
        Position : {data_root}/gt_position/{camera}/frame_{frame_id:04d}.exr
        Semantic : {data_root}/gt_semantic/{camera}/frame_{frame_id:04d}.png

        Parameters
        ----------
        frame_id       : zero-based frame index.
        load_position  : whether to load the world-XYZ position map.
        load_semantic  : whether to load the semantic (pass_index) mask.

        Raises
        ------
        FileNotFoundError : if the RGB or depth file for any camera is missing.
        """
        frame_name = f"frame_{frame_id:04d}"
        cameras: dict[str, CameraFrameData] = {}

        for cam in self.CAMERA_NAMES:
            rgb_path = self._data_root / "sim_frames" / cam / f"{frame_name}.png"
            depth_path = self._data_root / "gt_depth" / cam / f"{frame_name}.exr"
            pos_path = self._data_root / "gt_position" / cam / f"{frame_name}.exr"
            sem_path = self._data_root / "gt_semantic" / cam / f"{frame_name}.png"

            # Mandatory files
            if not rgb_path.exists():
                raise FileNotFoundError(
                    f"RGB file not found: {rgb_path}"
                )
            if not depth_path.exists():
                raise FileNotFoundError(
                    f"Depth EXR not found: {depth_path}"
                )

            rgb = self.load_rgb(rgb_path)
            depth = self.load_depth_exr(depth_path)

            position_map: Optional[np.ndarray] = None
            if load_position and pos_path.exists():
                position_map = self.load_position_exr(pos_path)

            semantic_mask: Optional[np.ndarray] = None
            if load_semantic and sem_path.exists():
                semantic_mask = self.load_semantic_png(sem_path)

            cameras[cam] = CameraFrameData(
                camera_name=cam,
                frame_id=frame_id,
                rgb=rgb,
                depth=depth,
                position_map=position_map,
                semantic_mask=semantic_mask,
                depth_source=self.DEPTH_SOURCE_GT,
            )

        return FrameSet(frame_id=frame_id, cameras=cameras)

    def load_rgb(self, path: Path) -> np.ndarray:
        """
        Load a PNG file and return a (H, W, 3) uint8 RGB array.

        Uses OpenCV (cv2.imread) and converts BGR → RGB with cv2.cvtColor.

        Parameters
        ----------
        path : path to the PNG file.

        Returns
        -------
        np.ndarray – shape (H, W, 3), dtype uint8.
        """
        _require_cv2()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2.imread failed to open: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

    def load_depth_exr(self, path: Path) -> np.ndarray:
        """
        Load an EXR depth file and return a (H, W) float32 array in metres.

        EXR loading steps
        -----------------
        1. Open the EXR file with OpenEXR.InputFile.
        2. Read the data window header to determine image dimensions.
        3. Request channel 'Z' as FLOAT (32-bit).
        4. Unpack the raw bytes with struct.unpack.
        5. Reshape to (H, W) and cast to float32.
        6. Clip negative values to 0 (Blender occasionally writes tiny negatives).

        Depth convention: camera Z-depth in metres (NOT Euclidean ray length).

        Parameters
        ----------
        path : path to the EXR file (must contain channel 'Z').

        Returns
        -------
        np.ndarray – shape (H, W), dtype float32.
        """
        _require_exr()
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Request 32-bit float for the Z channel
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw_z = exr_file.channel("Z", pt)

        # Unpack raw bytes into float values
        num_pixels = width * height
        depth_flat = struct.unpack(f"{num_pixels}f", raw_z)
        depth = np.array(depth_flat, dtype=np.float32).reshape((height, width))

        # Clip negative artefacts (can appear in Blender Z-pass near edges)
        depth = np.clip(depth, 0.0, None)
        return depth

    def load_position_exr(self, path: Path) -> np.ndarray:
        """
        Load an EXR world-position file and return a (H, W, 3) float32 array.

        Channels X, Y, Z are read and stacked in order along axis 2:
            result[..., 0] = X  (world right)
            result[..., 1] = Y  (world forward)
            result[..., 2] = Z  (world up)

        Parameters
        ----------
        path : path to the EXR file (must contain channels 'X', 'Y', 'Z').

        Returns
        -------
        np.ndarray – shape (H, W, 3), dtype float32.
        """
        _require_exr()
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        num_pixels = width * height

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = []
        for ch in ("X", "Y", "Z"):
            raw = exr_file.channel(ch, pt)
            flat = struct.unpack(f"{num_pixels}f", raw)
            arr = np.array(flat, dtype=np.float32).reshape((height, width))
            channels.append(arr)

        return np.stack(channels, axis=2)  # (H, W, 3)

    def load_semantic_png(self, path: Path) -> np.ndarray:
        """
        Load a semantic mask PNG and return a (H, W) uint8 pass-index array.

        The PNG pixel values correspond to Blender object pass_index values.

        Parameters
        ----------
        path : path to the PNG file.

        Returns
        -------
        np.ndarray – shape (H, W), dtype uint8.
        """
        _require_cv2()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(
                f"cv2.imread failed to open semantic mask: {path}"
            )
        return gray.astype(np.uint8)

    def available_frames(self) -> list[int]:
        """
        Scan {data_root}/sim_frames/front/ for frame_*.png files and return a
        sorted list of zero-based frame IDs.

        Returns an empty list (no crash) if the directory does not exist.
        """
        scan_dir = self._data_root / "sim_frames" / "front"
        if not scan_dir.exists():
            return []

        frame_ids: list[int] = []
        pattern = re.compile(r"^frame_(\d{4})\.png$")
        for p in scan_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                frame_ids.append(int(m.group(1)))

        return sorted(frame_ids)

    def load_all(
        self,
        max_frames: Optional[int] = None,
        load_position: bool = True,
        load_semantic: bool = True,
    ) -> list[FrameSet]:
        """
        Load all available frames (up to max_frames) and return as a list.

        Parameters
        ----------
        max_frames     : optional upper limit on the number of frames loaded.
        load_position  : forwarded to load_frame.
        load_semantic  : forwarded to load_frame.

        Returns
        -------
        list[FrameSet] – in ascending frame-ID order.
        """
        ids = self.available_frames()
        if max_frames is not None:
            ids = ids[:max_frames]

        framesets: list[FrameSet] = []
        for fid in ids:
            fs = self.load_frame(
                fid,
                load_position=load_position,
                load_semantic=load_semantic,
            )
            framesets.append(fs)

        return framesets

    def __repr__(self) -> str:
        return (
            f"FrameSetLoader(data_root='{self._data_root}', "
            f"expected_shape={self._expected_shape})"
        )
