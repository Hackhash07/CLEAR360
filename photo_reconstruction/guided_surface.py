"""Guided photo-projection modelling for scenes that benefit from manual surface control.

This is a lightweight 2.5D workflow: a full reference photograph becomes the
background surface, while user-drawn regions are duplicated at small depth
offsets to create controllable relief. It is not a substitute for complete
multi-view volumetric reconstruction, but it preserves the source photograph
exactly from the reference viewpoint.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _grid_for_image(width: int, height: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(0, width, step, dtype=np.int32)
    ys = np.arange(0, height, step, dtype=np.int32)
    if xs[-1] != width - 1:
        xs = np.append(xs, width - 1)
    if ys[-1] != height - 1:
        ys = np.append(ys, height - 1)
    return xs, ys


def _layer_faces(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                 vertex_offset: int) -> list[tuple[int, int, int]]:
    """Create front-facing triangles for image-grid cells covered by mask."""
    faces: list[tuple[int, int, int]] = []
    nx = len(xs)
    for iy in range(len(ys) - 1):
        y0, y1 = int(ys[iy]), int(ys[iy + 1])
        cy = min(mask.shape[0] - 1, (y0 + y1) // 2)
        for ix in range(nx - 1):
            x0, x1 = int(xs[ix]), int(xs[ix + 1])
            cx = min(mask.shape[1] - 1, (x0 + x1) // 2)
            if mask[cy, cx] == 0:
                continue
            tl = vertex_offset + iy * nx + ix
            tr = tl + 1
            bl = tl + nx
            br = bl + 1
            # The normal points toward the viewer (negative Y).
            faces.append((tl, bl, br))
            faces.append((tl, br, tr))
    return faces


def build_guided_photo_model(
    reference_image: str | Path,
    regions: list[dict[str, Any]],
    session_dir: str | Path,
    *,
    width_m: float = 2.0,
    grid_step_px: int = 8,
) -> dict[str, Any]:
    """Build a textured OBJ/MTL/PNG 2.5D relief model from one reference image.

    Region points are [x, y] image pixel coordinates. Each region's
    depth_cm is its offset toward the viewer relative to the base photo plane.
    Positive offsets therefore bring that photo region forward.
    """
    source = Path(reference_image).expanduser()
    output = Path(session_dir)
    mesh_dir = output / "mesh"
    originals_dir = output / "original_images"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(f"Reference photograph not found: {source}")
    if width_m <= 0 or grid_step_px < 2:
        raise ValueError("width_m must be positive and grid_step_px must be at least 2.")

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    if width < 64 or height < 64:
        raise ValueError("Reference photograph is too small for guided modelling.")

    texture_path = mesh_dir / "scene_mesh.png"
    image.save(texture_path, format="PNG", optimize=True)

    reference_copy = originals_dir / f"guided_reference{source.suffix.lower() or '.jpg'}"
    if source.resolve() != reference_copy.resolve():
        shutil.copy2(source, reference_copy)

    xs, ys = _grid_for_image(width, height, grid_step_px)
    nx, ny = len(xs), len(ys)
    pixel_grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    uv = np.column_stack((
        pixel_grid[:, 0] / max(1, width - 1),
        1.0 - pixel_grid[:, 1] / max(1, height - 1),
    )).astype(np.float32)

    world_width = float(width_m)
    world_height = world_width * height / width
    x_world = (pixel_grid[:, 0].astype(np.float32) / (width - 1) - 0.5) * world_width
    z_world = (0.5 - pixel_grid[:, 1].astype(np.float32) / (height - 1)) * world_height

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[tuple[int, int, int]] = []
    obj_lines: list[str] = ["mtllib scene_mesh.mtl", "o GuidedPhotoProjection"]
    normal_front = "vn 0 -1 0"
    normal_back = "vn 0 1 0"

    # Base photo plane sits behind all user-defined relief patches.
    all_mask = np.full((height, width), 255, dtype=np.uint8)
    layer_specs = [{"name": "Reference photo", "depth_cm": 0.0, "mask": all_mask}]
    for index, region in enumerate(regions, start=1):
        points = region.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Region {index} needs at least three polygon points.")
        depth_cm = float(region.get("depth_cm", 0.0))
        if not np.isfinite(depth_cm) or depth_cm < 0.0 or depth_cm > 200.0:
            raise ValueError(f"Region {index} depth_cm must be between 0 and 200.")
        polygon = np.asarray(points, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise ValueError(f"Region {index} has invalid polygon coordinates.")
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        if not np.any(mask):
            raise ValueError(f"Region {index} is empty after clipping to the image.")
        layer_specs.append({
            "name": str(region.get("name") or f"Region {index}"),
            "depth_cm": depth_cm,
            "mask": mask,
        })

    for layer_index, layer in enumerate(layer_specs):
        depth_m = float(layer["depth_cm"]) / 100.0
        y_world = np.full_like(x_world, -depth_m, dtype=np.float32)
        vertices = np.column_stack((x_world, y_world, z_world)).astype(np.float32)
        vertex_offset = sum(len(part) for part in vertices_parts)
        vertices_parts.append(vertices)
        faces_parts.extend(_layer_faces(layer["mask"], xs, ys, vertex_offset))

    if not faces_parts:
        raise RuntimeError("No mesh faces were generated from the reference image.")

    vertices = np.vstack(vertices_parts)
    faces = np.asarray(faces_parts, dtype=np.int64)

    # OBJ expects one-based indices. Every generated vertex has the matching UV.
    obj_lines.extend([
        "# Guided CLEAR360 photo-projection relief",
        f"# source_image {source.name}",
        f"# approximate_width_m {world_width:.6f}",
        "usemtl guided_photo",
    ])
    obj_lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
    obj_lines.extend(f"vt {u:.8f} {v:.8f}" for _ in vertices_parts for u, v in uv)
    obj_lines.extend([normal_front, normal_back])
    for a, b, c in faces:
        obj_lines.append(f"f {a + 1}/{a + 1}/1 {b + 1}/{b + 1}/1 {c + 1}/{c + 1}/1")
        obj_lines.append(f"f {c + 1}/{c + 1}/2 {b + 1}/{b + 1}/2 {a + 1}/{a + 1}/2")

    obj_path = mesh_dir / "scene_mesh.obj"
    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    mtl_path = mesh_dir / "scene_mesh.mtl"
    mtl_path.write_text(
        "newmtl guided_photo\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        "d 1.0\n"
        "illum 1\n"
        "map_Kd scene_mesh.png\n",
        encoding="utf-8",
    )

    report = {
        "backend": "Guided Photo Projection (2.5D)",
        "reference_image": reference_copy.name,
        "image_width_px": width,
        "image_height_px": height,
        "approximate_width_m": world_width,
        "grid_step_px": int(grid_step_px),
        "region_count": len(regions),
        "regions": [
            {
                "name": str(region.get("name") or f"Region {idx}"),
                "depth_cm": float(region.get("depth_cm", 0.0)),
                "points": [[float(x), float(y)] for x, y in region.get("points", [])],
            }
            for idx, region in enumerate(regions, start=1)
        ],
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces) * 2),
        "mesh_obj": str(obj_path),
        "texture_png": str(texture_path),
        "note": (
            "This is a guided 2.5D photo-projection model. It preserves the selected "
            "front photograph and adds user-defined depth layers; it does not infer "
            "unseen backside geometry."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"mesh_obj": obj_path, "texture_png": texture_path, "report": report}
