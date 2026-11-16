"""Camera-pose based world-up estimation and export alignment for COLMAP scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a COLMAP Hamilton quaternion into a world-to-camera rotation."""
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid zero-length COLMAP quaternion.")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def estimate_world_up(images_txt: str | Path) -> tuple[np.ndarray, float, int] | None:
    """Estimate scene up from COLMAP image poses.

    COLMAP camera coordinates are +X right, +Y down, +Z forward. With upright
    captures, camera-local -Y transformed into world coordinates is gravity-up.
    A principal-axis estimate tolerates some image-to-image camera roll.

    Returns (unit_up_vector, consistency_ratio, pose_count), or None when the
    registered views do not provide a stable dominant up direction.
    """
    path = Path(images_txt)
    if not path.is_file():
        return None

    ups: list[np.ndarray] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = map(float, parts[1:5])
            rotation = quaternion_to_rotation(qw, qx, qy, qz)
        except (ValueError, OverflowError):
            continue
        up = rotation.T @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        length = float(np.linalg.norm(up))
        if np.isfinite(length) and length > 1e-8:
            ups.append(up / length)

    if len(ups) < 3:
        return None

    vectors = np.vstack(ups)
    covariance = vectors.T @ vectors / len(vectors)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    dominant = eigenvectors[:, order[-1]]
    ratio = float(eigenvalues[order[-1]] / max(float(eigenvalues.sum()), 1e-12))

    # Require a coherent up direction. A low ratio means the phone was rolled
    # substantially between images or the views do not provide reliable gravity.
    if not np.isfinite(ratio) or ratio < 0.60:
        return None

    reference = vectors[0]
    if float(np.dot(dominant, reference)) < 0.0:
        dominant = -dominant
    dominant /= np.linalg.norm(dominant)
    return dominant.astype(np.float64), ratio, len(ups)


def rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest proper rotation that maps source onto target."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        basis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    cross = np.cross(a, b)
    skew = np.array(
        [[0.0, -cross[2], cross[1]],
         [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) / (1.0 + dot)


def align_obj_file(
    path: str | Path, rotation: np.ndarray, center: np.ndarray | None = None
) -> bool:
    """Rotate OBJ positions and normals while preserving independent UV indices."""
    obj_path = Path(path)
    if not obj_path.is_file():
        return False
    lines = obj_path.read_text(encoding="utf-8", errors="replace").splitlines()
    vertex_rows: list[tuple[int, np.ndarray]] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("v "):
            try:
                values = np.asarray([float(v) for v in stripped.split()[1:4]], dtype=np.float64)
                if values.shape == (3,) and np.isfinite(values).all():
                    vertex_rows.append((index, values))
            except ValueError:
                continue
    if not vertex_rows:
        return False

    if center is None:
        center = np.median(np.vstack([v for _, v in vertex_rows]), axis=0)
    else:
        center = np.asarray(center, dtype=np.float64)
    for index, values in vertex_rows:
        transformed = rotation @ (values - center) + center
        prefix = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
        parts = lines[index].lstrip().split()
        remainder = parts[4:]
        lines[index] = prefix + "v " + " ".join(f"{x:.9g}" for x in transformed)
        if remainder:
            lines[index] += " " + " ".join(remainder)

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("vn "):
            continue
        try:
            normal = np.asarray([float(v) for v in stripped.split()[1:4]], dtype=np.float64)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                continue
            transformed = rotation @ normal
            length = float(np.linalg.norm(transformed))
            if length > 1e-10:
                transformed /= length
            prefix = line[: len(line) - len(line.lstrip())]
            lines[index] = prefix + "vn " + " ".join(f"{x:.9g}" for x in transformed)
        except ValueError:
            continue

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True



def obj_median_center(path: str | Path) -> np.ndarray:
    """Read a robust center from OBJ vertex records."""
    vertices: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.lstrip()
            if not stripped.startswith("v "):
                continue
            try:
                values = [float(v) for v in stripped.split()[1:4]]
                if len(values) == 3 and np.isfinite(values).all():
                    vertices.append(values)
            except ValueError:
                continue
    if not vertices:
        raise ValueError(f"No valid vertices found in OBJ: {path}")
    return np.median(np.asarray(vertices, dtype=np.float64), axis=0)


def align_open3d_geometry_file(
    path: str | Path,
    rotation: np.ndarray,
    center: np.ndarray,
    *,
    point_cloud: bool = False,
) -> bool:
    """Apply the world-up rotation to a PLY mesh or point cloud, preserving colors."""
    import open3d as o3d

    file_path = Path(path)
    if not file_path.is_file():
        return False
    if point_cloud:
        geometry = o3d.io.read_point_cloud(str(file_path))
        if len(geometry.points) == 0:
            return False
        geometry.rotate(rotation, center=center)
        return bool(o3d.io.write_point_cloud(str(file_path), geometry, write_ascii=False))
    geometry = o3d.io.read_triangle_mesh(str(file_path))
    if len(geometry.vertices) == 0 or len(geometry.triangles) == 0:
        return False
    geometry.rotate(rotation, center=center)
    return bool(o3d.io.write_triangle_mesh(
        str(file_path), geometry, write_ascii=False, compressed=False
    ))


def align_glb_file(
    path: str | Path,
    rotation: np.ndarray,
    center: np.ndarray,
) -> bool:
    """Apply the scene-up rotation to a GLB while retaining its materials."""
    import trimesh

    file_path = Path(path)
    if not file_path.is_file():
        return False
    loaded = trimesh.load(str(file_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return False
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = center - rotation @ center
        loaded.apply_transform(transform)
        loaded.export(str(file_path), file_type="glb")
        return True
    if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        return False
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    loaded.apply_transform(transform)
    loaded.export(str(file_path), file_type="glb")
    return True
