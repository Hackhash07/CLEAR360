"""Load Wavefront OBJ geometry and texture atlases with independent OBJ indices."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_obj_index(text: str, count: int) -> int:
    value = int(text)
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ValueError(f"OBJ index {value} is outside an array of length {count}.")
    return index


@dataclass(frozen=True)
class TexturedOBJ:
    """Textured OBJ data plus geometry for faces that have no usable UVs.

    Iteration intentionally yields the original six return values so existing
    callers that unpacked the loader result remain source-compatible.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None
    uvs: np.ndarray
    texture: np.ndarray
    atlas_path: Path
    untextured_vertices: np.ndarray
    untextured_faces: np.ndarray
    textured_triangle_count: int
    untextured_triangle_count: int

    @property
    def total_triangle_count(self) -> int:
        return self.textured_triangle_count + self.untextured_triangle_count

    @property
    def texture_coverage(self) -> float:
        """Fraction of output triangles backed by valid texture coordinates."""
        total = self.total_triangle_count
        return self.textured_triangle_count / total if total else 0.0

    def _legacy_values(self):
        return (
            self.vertices,
            self.faces,
            self.normals,
            self.uvs,
            self.texture,
            self.atlas_path,
        )

    def __iter__(self):
        return iter(self._legacy_values())

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index):
        return self._legacy_values()[index]


def _read_material_textures(obj_path: Path) -> dict[str, Path]:
    """Resolve each MTL diffuse map relative to its material file."""
    result: dict[str, Path] = {}
    default_material = "__default__"
    with obj_path.open("r", encoding="utf-8", errors="replace") as obj_file:
        material_libraries = []
        for raw_line in obj_file:
            if not raw_line.lstrip().lower().startswith("mtllib "):
                continue
            try:
                tokens = shlex.split(raw_line.strip(), comments=True)
            except ValueError:
                continue
            material_libraries.extend(tokens[1:])

    for library in material_libraries:
        mtl_path = obj_path.parent / library
        if not mtl_path.is_file():
            continue
        current_material = default_material
        for raw_line in mtl_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                tokens = shlex.split(raw_line, comments=True)
            except ValueError:
                continue
            if not tokens:
                continue
            key = tokens[0].lower()
            if key == "newmtl" and len(tokens) > 1:
                current_material = " ".join(tokens[1:])
            elif key == "map_kd" and len(tokens) > 1:
                texture_path = (mtl_path.parent / tokens[-1]).expanduser()
                if texture_path.is_file():
                    result[current_material] = texture_path

    if not result:
        fallback = obj_path.with_name("scene_mesh.png")
        if fallback.is_file():
            result[default_material] = fallback
    return result


def _make_combined_texture(
    material_textures: dict[str, Path],
) -> tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]:
    """Pack per-material diffuse maps horizontally into one UV-addressable atlas.

    Each image is bottom-aligned in the composite. UV coordinates are remapped
    from the original material's [0, 1] range into its horizontal atlas tile.
    """
    if not material_textures:
        raise FileNotFoundError("No diffuse texture was referenced by the OBJ material file.")

    loaded: dict[str, Image.Image] = {}
    for material, path in material_textures.items():
        with Image.open(path) as image:
            loaded[material] = image.convert("RGB").copy()

    total_width = sum(image.width for image in loaded.values())
    max_height = max(image.height for image in loaded.values())
    if total_width <= 0 or max_height <= 0:
        raise ValueError("Texture atlas has invalid dimensions.")
    if total_width * max_height > 120_000_000:
        raise ValueError(
            "Combined texture atlases are too large for the desktop viewer; "
            "reduce the OpenMVS texture size and reconstruct."
        )

    composite = Image.new("RGB", (total_width, max_height), (32, 32, 32))
    remap: dict[str, tuple[float, float, float, float]] = {}
    x_offset = 0
    for material, image in loaded.items():
        composite.paste(image, (x_offset, max_height - image.height))
        remap[material] = (
            x_offset / total_width,
            image.width / total_width,
            0.0,
            image.height / max_height,
        )
        x_offset += image.width

    return np.asarray(composite, dtype=np.uint8), remap


def load_textured_obj(obj_file: str | Path) -> TexturedOBJ:
    """Load textured OBJ faces while preserving faces without UVs separately.

    OBJ position, texture-coordinate, and normal indices are independent.
    Textured corners are expanded by their (v, vt, vn, material) tuple, so
    texture seams do not pair UV coordinates with the wrong geometric vertex.

    A polygon is treated as untextured when any corner omits its UV index or
    when its material has no resolvable diffuse map. Its triangles are returned
    separately for a neutral, unlit geometry fallback. No fabricated UV index
    is ever assigned to an untextured corner.
    """
    path = Path(obj_file)
    if not path.is_file():
        raise FileNotFoundError(f"OBJ file not found: {path}")

    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    source_normals: list[list[float]] = []
    face_records: list[tuple[str, list[str]]] = []
    current_material = "__default__"

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            command = parts[0].lower()
            if command == "v" and len(parts) >= 4:
                positions.append([float(v) for v in parts[1:4]])
            elif command == "vt" and len(parts) >= 3:
                texcoords.append([float(v) for v in parts[1:3]])
            elif command == "vn" and len(parts) >= 4:
                source_normals.append([float(v) for v in parts[1:4]])
            elif command == "usemtl":
                current_material = " ".join(parts[1:]) or "__default__"
            elif command == "f" and len(parts) >= 4:
                face_records.append((current_material, parts[1:]))

    if not positions or not face_records:
        raise ValueError("OBJ must contain vertices and polygon faces.")

    material_textures = _read_material_textures(path)
    if not material_textures:
        raise FileNotFoundError(f"No texture atlas was found for {path.name}.")
    texture, uv_remap = _make_combined_texture(material_textures)

    expanded_vertices: list[list[float]] = []
    expanded_uvs: list[list[float]] = []
    expanded_normals: list[list[float]] = []
    expanded_faces: list[list[int]] = []
    corner_map: dict[tuple[int, int, int | None, str], int] = {}
    normals_complete = True

    plain_vertices: list[list[float]] = []
    plain_faces: list[list[int]] = []
    plain_vertex_map: dict[int, int] = {}

    textured_triangle_count = 0
    untextured_triangle_count = 0

    def parse_corner(token: str) -> tuple[int, int | None, int | None]:
        components = token.split("/")
        if len(components) > 3 or not components[0]:
            raise ValueError(f"Invalid OBJ face corner: {token!r}")

        vertex_index = _resolve_obj_index(components[0], len(positions))
        uv_index: int | None = None
        normal_index: int | None = None

        if len(components) >= 2 and components[1]:
            if not texcoords:
                raise ValueError(
                    f"OBJ face corner {token!r} references UVs, but the OBJ has no vt records."
                )
            uv_index = _resolve_obj_index(components[1], len(texcoords))
        if len(components) >= 3 and components[2]:
            normal_index = _resolve_obj_index(components[2], len(source_normals))

        return vertex_index, uv_index, normal_index

    for material, tokens in face_records:
        corners = [parse_corner(token) for token in tokens]
        if len(corners) < 3:
            continue

        material_key = material if material in uv_remap else "__default__"
        if material_key not in uv_remap:
            if len(uv_remap) == 1:
                material_key = next(iter(uv_remap))
            else:
                material_key = ""

        is_textured = (
            all(uv_index is not None for _, uv_index, _ in corners)
            and bool(material_key)
        )
        triangle_count = len(corners) - 2

        if not is_textured:
            polygon_indices: list[int] = []
            for vertex_index, _, _ in corners:
                if vertex_index not in plain_vertex_map:
                    plain_vertex_map[vertex_index] = len(plain_vertices)
                    plain_vertices.append(positions[vertex_index])
                polygon_indices.append(plain_vertex_map[vertex_index])
            for index in range(1, len(polygon_indices) - 1):
                plain_faces.append(
                    [polygon_indices[0], polygon_indices[index], polygon_indices[index + 1]]
                )
            untextured_triangle_count += triangle_count
            continue

        polygon_indices = []
        for vertex_index, uv_index, normal_index in corners:
            assert uv_index is not None
            if normal_index is None:
                normals_complete = False

            key = (vertex_index, uv_index, normal_index, material_key)
            if key in corner_map:
                polygon_indices.append(corner_map[key])
                continue

            atlas_u0, atlas_uw, _, atlas_vh = uv_remap[material_key]
            source_u, source_v = texcoords[uv_index]
            new_uv = [
                atlas_u0 + float(source_u) * atlas_uw,
                float(source_v) * atlas_vh,
            ]
            new_index = len(expanded_vertices)
            expanded_vertices.append(positions[vertex_index])
            expanded_uvs.append(new_uv)
            if normal_index is not None:
                expanded_normals.append(source_normals[normal_index])
            corner_map[key] = new_index
            polygon_indices.append(new_index)

        for index in range(1, len(polygon_indices) - 1):
            expanded_faces.append(
                [polygon_indices[0], polygon_indices[index], polygon_indices[index + 1]]
            )
        textured_triangle_count += triangle_count

    if not expanded_faces:
        raise ValueError(
            "OBJ contains no faces with complete UV coordinates and a resolvable diffuse material."
        )

    normals = None
    if normals_complete and len(expanded_normals) == len(expanded_vertices):
        normals_array = np.asarray(expanded_normals, dtype=np.float32)
        lengths = np.linalg.norm(normals_array, axis=1, keepdims=True)
        normals = normals_array / np.maximum(lengths, 1e-8)

    return TexturedOBJ(
        vertices=np.asarray(expanded_vertices, dtype=np.float32),
        faces=np.asarray(expanded_faces, dtype=np.uint32),
        normals=normals,
        uvs=np.asarray(expanded_uvs, dtype=np.float32),
        texture=texture,
        atlas_path=next(iter(material_textures.values())),
        untextured_vertices=np.asarray(plain_vertices, dtype=np.float32).reshape(-1, 3),
        untextured_faces=np.asarray(plain_faces, dtype=np.uint32).reshape(-1, 3),
        textured_triangle_count=textured_triangle_count,
        untextured_triangle_count=untextured_triangle_count,
    )
