"""Load textured GLB scenes into one VisPy-friendly texture atlas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TexturedGLB:
    """Flattened GLB scene with a combined texture atlas and explicit fallbacks."""

    all_vertices: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    texture: np.ndarray | None
    untextured_vertices: np.ndarray
    untextured_faces: np.ndarray
    untextured_vertex_colors: np.ndarray
    textured_geometry_count: int
    untextured_geometry_count: int

    @property
    def texture_coverage(self) -> float:
        total = len(self.faces) + len(self.untextured_faces)
        return len(self.faces) / total if total else 0.0


def _material_image(material) -> Image.Image | None:
    if material is None:
        return None
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        return None
    try:
        if isinstance(image, Image.Image):
            return image.convert("RGB").copy()
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] < 3:
            return None
        return Image.fromarray(np.asarray(array[:, :, :3], dtype=np.uint8)).convert("RGB")
    except (TypeError, ValueError, OSError):
        return None


def _vertex_colors(visual, vertex_count: int) -> np.ndarray:
    colors = None
    if visual is not None and hasattr(visual, "vertex_colors"):
        try:
            candidate = np.asarray(visual.vertex_colors)
            if candidate.ndim == 2 and candidate.shape[0] == vertex_count and candidate.shape[1] >= 3:
                colors = candidate[:, :4].astype(np.float32)
        except (TypeError, ValueError):
            colors = None
    if colors is None:
        colors = np.tile(
            np.asarray([[0.58, 0.60, 0.64, 1.0]], dtype=np.float32),
            (vertex_count, 1),
        )
    else:
        if colors.max(initial=0.0) > 1.0:
            colors /= 255.0
        if colors.shape[1] == 3:
            colors = np.column_stack(
                [colors, np.ones(vertex_count, dtype=np.float32)]
            )
        colors = np.clip(colors, 0.0, 1.0)
    return np.ascontiguousarray(colors, dtype=np.float32)


def load_textured_glb(glb_path: str | Path) -> TexturedGLB:
    """Load a GLB/GLTF scene and retain per-material embedded base-color maps.

    Trimesh imports glTF UVs into its bottom-left texture convention. Each
    textured geometry is transformed into scene coordinates, its diffuse map
    is packed into a horizontal atlas, and its UVs are remapped into that tile.
    Geometry without a valid embedded image and per-vertex UVs is returned
    separately so it can be shown without pretending it has photo texture.
    """
    import trimesh

    path = Path(glb_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GLB file not found: {path}")
    scene = trimesh.load(str(path), force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        instances = [(np.eye(4, dtype=np.float64), scene)]
    elif isinstance(scene, trimesh.Scene):
        instances = []
        for node_name in list(scene.graph.nodes_geometry):
            transform, geometry_name = scene.graph.get(node_name)
            geometry = scene.geometry.get(geometry_name)
            if isinstance(geometry, trimesh.Trimesh):
                instances.append((np.asarray(transform, dtype=np.float64), geometry))
        if not instances:
            instances = [
                (np.eye(4, dtype=np.float64), geometry)
                for geometry in scene.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
            ]
    else:
        raise ValueError("GLB did not contain a mesh or scene.")

    if not instances:
        raise ValueError("GLB scene contains no triangle meshes.")

    textured_parts = []
    plain_parts = []
    all_vertices = []
    atlas_images: list[Image.Image] = []

    for transform, geometry in instances:
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        faces = np.asarray(geometry.faces, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or len(faces) == 0:
            continue
        if not np.isfinite(vertices).all() or int(faces.max(initial=0)) >= len(vertices):
            raise ValueError("GLB contains non-finite vertices or invalid face indices.")

        homogeneous = np.column_stack(
            [vertices.astype(np.float64), np.ones(len(vertices), dtype=np.float64)]
        )
        transformed = (homogeneous @ transform.T)[:, :3].astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("GLB scene transform produced non-finite vertices.")
        all_vertices.append(transformed)

        visual = getattr(geometry, "visual", None)
        uv = None
        if visual is not None and hasattr(visual, "uv") and visual.uv is not None:
            candidate_uv = np.asarray(visual.uv, dtype=np.float32)
            if candidate_uv.shape == (len(vertices), 2) and np.isfinite(candidate_uv).all():
                uv = candidate_uv

        material = getattr(visual, "material", None) if visual is not None else None
        image = _material_image(material)
        if uv is not None and image is not None:
            textured_parts.append((transformed, faces, uv, image))
            atlas_images.append(image)
        else:
            plain_parts.append(
                (transformed, faces, _vertex_colors(visual, len(vertices)))
            )

    if not all_vertices:
        raise ValueError("GLB contains no usable triangular mesh geometry.")

    textured_vertices: list[np.ndarray] = []
    textured_faces: list[np.ndarray] = []
    remapped_uvs: list[np.ndarray] = []
    texture: np.ndarray | None = None

    if atlas_images:
        total_width = sum(image.width for image in atlas_images)
        max_height = max(image.height for image in atlas_images)
        if total_width <= 0 or max_height <= 0 or total_width * max_height > 120_000_000:
            raise ValueError("Combined GLB texture atlas exceeds the safe viewer size.")
        composite = Image.new("RGB", (total_width, max_height), (32, 32, 32))
        x_offset = 0
        for vertices, faces, uv, image in textured_parts:
            composite.paste(image, (x_offset, max_height - image.height))
            uv_wrapped = uv - np.floor(uv)
            atlas_uv = np.empty_like(uv_wrapped, dtype=np.float32)
            atlas_uv[:, 0] = (
                x_offset / total_width
                + uv_wrapped[:, 0] * (image.width / total_width)
            )
            atlas_uv[:, 1] = uv_wrapped[:, 1] * (image.height / max_height)
            textured_vertices.append(vertices)
            textured_faces.append(faces + sum(len(part) for part in textured_vertices[:-1]))
            remapped_uvs.append(atlas_uv)
            x_offset += image.width
        texture = np.asarray(composite, dtype=np.uint8)

    plain_vertices: list[np.ndarray] = []
    plain_faces: list[np.ndarray] = []
    plain_colors: list[np.ndarray] = []
    vertex_offset = 0
    for vertices, faces, colors in plain_parts:
        plain_vertices.append(vertices)
        plain_faces.append(faces + vertex_offset)
        plain_colors.append(colors)
        vertex_offset += len(vertices)

    return TexturedGLB(
        all_vertices=np.ascontiguousarray(np.vstack(all_vertices), dtype=np.float32),
        vertices=(
            np.ascontiguousarray(np.vstack(textured_vertices), dtype=np.float32)
            if textured_vertices else np.empty((0, 3), dtype=np.float32)
        ),
        faces=(
            np.ascontiguousarray(np.vstack(textured_faces), dtype=np.uint32)
            if textured_faces else np.empty((0, 3), dtype=np.uint32)
        ),
        uvs=(
            np.ascontiguousarray(np.vstack(remapped_uvs), dtype=np.float32)
            if remapped_uvs else np.empty((0, 2), dtype=np.float32)
        ),
        texture=texture,
        untextured_vertices=(
            np.ascontiguousarray(np.vstack(plain_vertices), dtype=np.float32)
            if plain_vertices else np.empty((0, 3), dtype=np.float32)
        ),
        untextured_faces=(
            np.ascontiguousarray(np.vstack(plain_faces), dtype=np.uint32)
            if plain_faces else np.empty((0, 3), dtype=np.uint32)
        ),
        untextured_vertex_colors=(
            np.ascontiguousarray(np.vstack(plain_colors), dtype=np.float32)
            if plain_colors else np.empty((0, 4), dtype=np.float32)
        ),
        textured_geometry_count=len(textured_parts),
        untextured_geometry_count=len(plain_parts),
    )
