"""
CLEAR360 – Camera Calibration Module
=====================================
Provides dataclasses and utilities for camera intrinsics, full camera models
(intrinsics + extrinsics), and a store that loads/saves the cameras.yaml
calibration file.

Coordinate Conventions
-----------------------
World / Vehicle frame:
    X = RIGHT,  Y = FORWARD (direction of travel),  Z = UP.
    Vehicle centre at world origin.

Camera frame (OpenCV):
    x = right in image,  y = down in image,  z = forward into scene.

T_cam_world (4×4 float64):
    Transforms a world-frame homogeneous point to camera frame:
        p_cam = T_cam_world @ p_world_hom

Depth convention:
    Camera Z-depth in metres (NOT Euclidean ray length). Stored as float32.

Unprojection formula (pixel → world-frame 3-D point):
    Given pixel (u, v) and camera-Z depth d (metres):

        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d
        p_cam_hom = [x_cam, y_cam, z_cam, 1.0]
        T_world_cam = np.linalg.inv(T_cam_world)
        p_world = T_world_cam @ p_cam_hom   # → [X, Y, Z, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    Attributes
    ----------
    width, height       : image resolution in pixels
    fx, fy              : focal lengths in pixels
    cx, cy              : principal point in pixels
    focal_length_mm     : physical focal length (millimetres)
    sensor_width_mm     : physical sensor width (millimetres)
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    focal_length_mm: float
    sensor_width_mm: float

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    def K(self) -> np.ndarray:
        """Return the 3×3 camera intrinsic matrix K.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def hfov_deg(self) -> float:
        """Horizontal field of view in degrees.

        Formula: 2 * degrees(atan2(width/2, fx))
        This is the full horizontal FOV (not the half-angle).
        """
        return 2.0 * math.degrees(math.atan2(self.width / 2.0, self.fx))

    def vfov_deg(self) -> float:
        """Vertical field of view in degrees.

        Formula: 2 * degrees(atan2(height/2, fy))
        """
        return 2.0 * math.degrees(math.atan2(self.height / 2.0, self.fy))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON/YAML-serialisable dict."""
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "focal_length_mm": self.focal_length_mm,
            "sensor_width_mm": self.sensor_width_mm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        """Construct from a plain dict (inverse of to_dict)."""
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            focal_length_mm=float(d["focal_length_mm"]),
            sensor_width_mm=float(d["sensor_width_mm"]),
        )

    def __repr__(self) -> str:
        return (
            f"CameraIntrinsics(width={self.width}, height={self.height}, "
            f"fx={self.fx:.4f}, fy={self.fy:.4f}, "
            f"cx={self.cx:.4f}, cy={self.cy:.4f}, "
            f"focal_length_mm={self.focal_length_mm}, "
            f"sensor_width_mm={self.sensor_width_mm})"
        )


# ---------------------------------------------------------------------------
# CameraModel
# ---------------------------------------------------------------------------

@dataclass
class CameraModel:
    """
    Full camera model: intrinsics + extrinsics.

    Attributes
    ----------
    name            : camera identifier, e.g. 'front', 'rear', 'left', 'right'
    intrinsics      : CameraIntrinsics
    T_cam_world     : (4,4) float64 – OpenCV world-to-camera transform.
                      Satisfies: p_cam = T_cam_world @ p_world_hom
    position_world  : (3,) float64 – camera origin in world frame.
                      Extracted from T_cam_world as:
                          position_world = -R.T @ t
                      where R = T_cam_world[:3,:3], t = T_cam_world[:3,3]
    depth_convention: provenance tag for the depth representation.
    source          : where the calibration came from.

    Coordinate Conventions
    ----------------------
    World frame : X=RIGHT, Y=FORWARD, Z=UP  (vehicle at origin)
    Camera frame: x=right-in-image, y=down-in-image, z=forward-into-scene

    Unprojection formula
    --------------------
    Given pixel (u, v) and camera-Z depth d (metres):

        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d
        p_cam_hom = [x_cam, y_cam, z_cam, 1.0]
        T_world_cam = np.linalg.inv(T_cam_world)
        p_world = T_world_cam @ p_cam_hom   # → [X, Y, Z, 1]
    """

    name: str
    intrinsics: CameraIntrinsics
    T_cam_world: np.ndarray        # shape (4,4), dtype float64
    position_world: np.ndarray     # shape (3,),  dtype float64
    depth_convention: str = "camera_z_depth_metres"
    source: str = "blender_simulation"

    # ------------------------------------------------------------------
    # Derived properties (computed from T_cam_world)
    # ------------------------------------------------------------------

    @property
    def T_world_cam(self) -> np.ndarray:
        """Camera-to-world transform (4×4).  Inverse of T_cam_world."""
        return np.linalg.inv(self.T_cam_world)

    @property
    def R_cam_world(self) -> np.ndarray:
        """Rotation part of T_cam_world: upper-left 3×3 block."""
        return self.T_cam_world[:3, :3].copy()

    @property
    def t_cam_world(self) -> np.ndarray:
        """Translation part of T_cam_world: right column, shape (3,)."""
        return self.T_cam_world[:3, 3].copy()

    # ------------------------------------------------------------------
    # Projection / unprojection
    # ------------------------------------------------------------------

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Unproject a pixel (u, v) with camera-Z depth d to a world-frame 3-D point.

        Parameters
        ----------
        u, v  : pixel coordinates (column, row). Origin = top-left corner.
        depth : camera Z-depth in metres (NOT Euclidean ray length).

        Returns
        -------
        np.ndarray, shape (3,), dtype float64 – point in world frame [X, Y, Z].

        Unprojection formula
        --------------------
            x_cam = (u - cx) * depth / fx
            y_cam = (v - cy) * depth / fy
            z_cam = depth
            p_cam_hom = [x_cam, y_cam, z_cam, 1.0]
            T_world_cam = np.linalg.inv(T_cam_world)
            p_world = T_world_cam @ p_cam_hom   # → [X, Y, Z, 1]
        """
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        p_cam_hom = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
        p_world_hom = self.T_world_cam @ p_cam_hom   # [X, Y, Z, 1]
        return p_world_hom[:3]

    def project(
        self, point_world: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Project a world-frame 3-D point to pixel coordinates and depth.

        Parameters
        ----------
        point_world : array-like, shape (3,) – point in world frame [X, Y, Z].

        Returns
        -------
        (u, v, depth) where u, v are pixel coordinates and depth is camera
        Z-depth in metres.  Returns (nan, nan, nan) if the point is behind
        the camera (z_cam ≤ 0).

        Projection formula
        ------------------
            p_world_hom = [X, Y, Z, 1.0]
            p_cam_hom   = T_cam_world @ p_world_hom    # → [x, y, z, 1]
            depth = z_cam  (camera Z-depth, NOT Euclidean)
            u = fx * (x_cam / z_cam) + cx
            v = fy * (y_cam / z_cam) + cy
        """
        p_world_hom = np.array(
            [point_world[0], point_world[1], point_world[2], 1.0],
            dtype=np.float64,
        )
        p_cam_hom = self.T_cam_world @ p_world_hom  # [x, y, z, 1]
        z_cam = float(p_cam_hom[2])

        if z_cam <= 0.0:
            return float("nan"), float("nan"), float("nan")

        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        u = fx * (p_cam_hom[0] / z_cam) + cx
        v = fy * (p_cam_hom[1] / z_cam) + cy
        return float(u), float(v), float(z_cam)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON/YAML-serialisable dict."""
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "T_cam_world": self.T_cam_world.tolist(),
            "position_world": self.position_world.tolist(),
            "depth_convention": self.depth_convention,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "CameraModel":
        """Construct a CameraModel from a plain dict (inverse of to_dict)."""
        intrinsics = CameraIntrinsics.from_dict(d["intrinsics"])
        T_cam_world = np.array(d["T_cam_world"], dtype=np.float64)
        if T_cam_world.shape != (4, 4):
            raise ValueError(
                f"Camera '{name}': T_cam_world must be (4,4), got {T_cam_world.shape}"
            )
        position_world = np.array(d["position_world"], dtype=np.float64)
        return cls(
            name=name,
            intrinsics=intrinsics,
            T_cam_world=T_cam_world,
            position_world=position_world,
            depth_convention=d.get("depth_convention", "camera_z_depth_metres"),
            source=d.get("source", "blender_simulation"),
        )

    def __repr__(self) -> str:
        pos = self.position_world
        return (
            f"CameraModel(name='{self.name}', "
            f"position_world=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
            f"depth_convention='{self.depth_convention}', "
            f"source='{self.source}')"
        )


# ---------------------------------------------------------------------------
# CalibrationStore
# ---------------------------------------------------------------------------

class CalibrationStore:
    """
    Loads and holds all 4 camera models from a cameras.yaml file.

    Usage
    -----
        store = CalibrationStore("config/cameras.yaml")
        store.load()
        front_cam = store.get("front")
    """

    CAMERA_NAMES = ["front", "rear", "left", "right"]

    def __init__(self, cameras_yaml_path: "str | Path") -> None:
        self._path = Path(cameras_yaml_path)
        self._cameras: dict[str, CameraModel] = {}

    def load(self) -> None:
        """Load cameras.yaml and populate the internal camera dict."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"cameras.yaml not found at: {self._path}"
            )
        with self._path.open("r") as fh:
            raw: dict = yaml.safe_load(fh)

        cameras_raw: dict = raw.get("cameras", {})
        self._cameras = {}
        for name, cam_dict in cameras_raw.items():
            self._cameras[name] = CameraModel.from_dict(name, cam_dict)

    def get(self, name: str) -> CameraModel:
        """Return the CameraModel for the given camera name.

        Parameters
        ----------
        name : one of 'front', 'rear', 'left', 'right'
        """
        if not self._cameras:
            raise RuntimeError(
                "CalibrationStore is empty. Call load() first."
            )
        if name not in self._cameras:
            raise KeyError(
                f"Camera '{name}' not found. Available: {list(self._cameras.keys())}"
            )
        return self._cameras[name]

    def all_cameras(self) -> dict[str, CameraModel]:
        """Return all camera models as a dict keyed by camera name."""
        return dict(self._cameras)

    def save(self, output_path: "str | Path") -> None:
        """Save all camera models back to YAML.

        Parameters
        ----------
        output_path : destination path for the YAML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cameras_dict = {name: cam.to_dict() for name, cam in self._cameras.items()}
        payload = {"cameras": cameras_dict}

        with output_path.open("w") as fh:
            yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)

    @property
    def camera_names(self) -> list[str]:
        """Return the canonical list of camera names."""
        return list(self.CAMERA_NAMES)

    def __repr__(self) -> str:
        loaded = list(self._cameras.keys()) if self._cameras else []
        return (
            f"CalibrationStore(path='{self._path}', "
            f"cameras_loaded={loaded})"
        )
