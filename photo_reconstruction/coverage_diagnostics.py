"""COLMAP sparse overlap and dense-cloud projection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from photo_reconstruction.scene_orientation import quaternion_to_rotation


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


def parse_cameras_txt(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1].upper()
        width, height = int(parts[2]), int(parts[3])
        params = tuple(float(value) for value in parts[4:])
        cameras[camera_id] = ColmapCamera(
            camera_id, model, width, height, params
        )
    if not cameras:
        raise ValueError(f"No COLMAP cameras found in {path}")
    return cameras


def parse_images_txt(path: str | Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    for raw_line in Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # COLMAP pose records end in a filename; observation records are numeric.
        if len(parts) < 10 or not parts[0].isdigit() or not parts[8].isdigit():
            continue
        try:
            float(parts[-1])
            continue
        except ValueError:
            pass
        try:
            pose = [float(value) for value in parts[1:8]]
            images.append(
                ColmapImage(
                    image_id=int(parts[0]),
                    qw=pose[0], qx=pose[1], qy=pose[2], qz=pose[3],
                    tx=pose[4], ty=pose[5], tz=pose[6],
                    camera_id=int(parts[8]),
                    name=" ".join(parts[9:]),
                )
            )
        except (ValueError, IndexError):
            continue
    if not images:
        raise ValueError(f"No registered COLMAP image poses found in {path}")
    return images


def parse_points3d_tracks(path: str | Path) -> tuple[int, dict[int, int], dict[tuple[int, int], int]]:
    """Return point count, per-image track observations, and pairwise co-visibility."""
    point_count = 0
    per_image: dict[int, int] = {}
    pairwise: dict[tuple[int, int], int] = {}
    for raw_line in Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            # POINT3D_ID X Y Z R G B ERROR [IMAGE_ID POINT2D_IDX ...]
            point_id = int(parts[0])
            int(parts[4]); int(parts[5]); int(parts[6])
            float(parts[7])
            track_values = parts[8:]
            if len(track_values) % 2:
                continue
            image_ids = sorted({int(track_values[index]) for index in range(0, len(track_values), 2)})
        except ValueError:
            continue
        point_count += 1
        for image_id in image_ids:
            per_image[image_id] = per_image.get(image_id, 0) + 1
        for first, second in combinations(image_ids, 2):
            pairwise[(first, second)] = pairwise.get((first, second), 0) + 1
    return point_count, per_image, pairwise


def project_points(
    points_world: np.ndarray,
    camera: ColmapCamera,
    image: ColmapImage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into the original COLMAP camera image.

    The backend currently uses SIMPLE_RADIAL, but common COLMAP pinhole and
    OpenCV camera models are implemented to make the diagnostic reusable.
    Returns (u, v, camera_z, valid_in_front_and_finite).
    """
    points = np.asarray(points_world, dtype=np.float64)
    rotation = quaternion_to_rotation(image.qw, image.qx, image.qy, image.qz)
    translation = np.asarray([image.tx, image.ty, image.tz], dtype=np.float64)
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-8)
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid] = camera_points[valid, 0] / z[valid]
    y[valid] = camera_points[valid, 1] / z[valid]

    params = camera.params
    model = camera.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
        xd, yd = x, y
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        xd, yd = x, y
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        fx = fy = f
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        xd, yd = x * radial, y * radial
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        fx = fy = f
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd, yd = x * radial, y * radial
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    else:
        raise NotImplementedError(
            f"Coverage projection does not yet support COLMAP camera model {model}"
        )

    u = fx * xd + cx
    v = fy * yd + cy
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return u, v, z, valid
