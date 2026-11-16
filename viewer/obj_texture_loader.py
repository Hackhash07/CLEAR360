"""Load Wavefront OBJ geometry and texture atlases with independent OBJ indices."""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_obj_index(text: str, count: int) -> int:
    value = int(text)
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ValueError(f"OBJ index {value} is outside an array of length {count}.")
    return index


def _read_material_textures(obj_path: Path) -> dict[str, Path]:
    """Resolve each MTL diffuse map relative to its material file."""
    result: dict[str, Path] = {}
    default_material = "__default__"
    with obj_path.open("r", encoding="utf-8", errors="replace") as obj_file:
        material_libraries = [
            shlex.split(line.strip(), comments=True)[1]
            for line in obj_file
            if line.lstrip().lower().startswith("mtllib ")
            and len(shlex.split(line.strip(), comments=True)) > 1
        ]

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


def load_textured_obj(
    obj_file: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, Path]:
    """Return expanded vertices/faces/normals/UVs plus a combined RGB texture.

    OBJ position, texture-coordinate, and normal indices are independent. This
    loader expands corners by their (v, vt, vn, material) tuple, so texture seams
    do not cause UV coordinates to be paired with the wrong geometric vertex.
    Polygonal faces are triangulated as a fan.
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

    if not positions or not texcoords or not face_records:
        raise ValueError("OBJ must contain vertices, UV coordinates and polygon faces.")

    material_textures = _read_material_textures(path)
    if "__default__" not in material_textures and not any(
        material in material_textures for material, _ in face_records
    ):
        raise FileNotFoundError(f"No texture atlas was found for the materials in {path.name}.")
    # An OBJ without explicit usemtl records can still use the fallback atlas.
    default_texture = material_textures.get("__default__")
    texture, uv_remap = _make_combined_texture(material_textures)

    expanded_vertices: list[list[float]] = []
    expanded_uvs: list[list[float]] = []
    expanded_normals: list[list[float]] = []
    expanded_faces: list[list[int]] = []
    corner_map: dict[tuple[int, int, int | None, str], int] = {}
    normals_complete = True

    def expanded_index(material: str, token: str) -> int:
        nonlocal normals_complete
        components = token.split("/")
        if len(components) < 2 or not components[0] or not components[1]:
            raise ValueError("A textured OBJ face is missing a vertex or UV index.")
        vertex_index = _resolve_obj_index(components[0], len(positions))
        uv_index = _resolve_obj_index(components[1], len(texcoords))
        normal_index: int | None = None
        if len(components) >= 3 and components[2]:
            normal_index = _resolve_obj_index(components[2], len(source_normals))
        else:
            normals_complete = False

        material_key = material if material in uv_remap else "__default__"
        if material_key not in uv_remap:
            # If there is exactly one texture, it can safely serve every face.
            if len(uv_remap) == 1:
                material_key = next(iter(uv_remap))
            else:
                raise ValueError(f"No diffuse texture found for OBJ material '{material}'.")
        key = (vertex_index, uv_index, normal_index, material_key)
        if key in corner_map:
            return corner_map[key]

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
        return new_index

    for material, tokens in face_records:
        polygon = [expanded_index(material, token) for token in tokens]
        for index in range(1, len(polygon) - 1):
            expanded_faces.append([polygon[0], polygon[index], polygon[index + 1]])

    if not expanded_faces:
        raise ValueError("OBJ did not contain any valid triangle faces.")

    normals = None
    if normals_complete and len(expanded_normals) == len(expanded_vertices):
        normals_array = np.asarray(expanded_normals, dtype=np.float32)
        lengths = np.linalg.norm(normals_array, axis=1, keepdims=True)
        normals = normals_array / np.maximum(lengths, 1e-8)

    return (
        np.asarray(expanded_vertices, dtype=np.float32),
        np.asarray(expanded_faces, dtype=np.uint32),
        normals,
        np.asarray(expanded_uvs, dtype=np.float32),
        texture,
        next(iter(material_textures.values())),
    )
