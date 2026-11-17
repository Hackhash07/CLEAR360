"""Utilities for making OpenMVS OBJ/MTL exports self-contained."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path


def copy_mtl_diffuse_maps(
    source_mtl: str | Path,
    destination_dir: str | Path,
    *,
    output_name: str | None = None,
) -> Path:
    """Copy every diffuse image referenced by an MTL and rewrite its map_Kd paths.

    Wavefront MTL paths are relative to the MTL file. OpenMVS 2.4.0's OBJ
    exporter writes JPEG maps beside the OBJ; copying only GLB PNG sidecars
    leaves a syntactically valid but visually incomplete OBJ export.
    """
    source_mtl_path = Path(source_mtl).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    if not source_mtl_path.is_file():
        raise FileNotFoundError(f"Material file not found: {source_mtl_path}")

    destination.mkdir(parents=True, exist_ok=True)
    rewritten: list[str] = []
    copied_maps: list[Path] = []

    for raw_line in source_mtl_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = raw_line.lstrip()
        try:
            tokens = shlex.split(stripped, comments=True)
        except ValueError as exc:
            raise ValueError(f"Malformed MTL line: {raw_line}") from exc

        if not tokens or tokens[0].lower() != "map_kd":
            rewritten.append(raw_line)
            continue
        if len(tokens) < 2:
            raise ValueError(f"MTL map_Kd line has no filename: {raw_line}")

        # MTL map options (for example -s or -o) precede the final filename.
        source_texture = Path(tokens[-1]).expanduser()
        if not source_texture.is_absolute():
            source_texture = source_mtl_path.parent / source_texture
        source_texture = source_texture.resolve()
        if not source_texture.is_file():
            raise FileNotFoundError(
                f"MTL diffuse map does not exist: {source_texture} "
                f"(referenced by {source_mtl_path.name})"
            )

        target_texture = destination / source_texture.name
        if source_texture != target_texture.resolve():
            if target_texture.exists() and target_texture.resolve() != source_texture:
                # Avoid silently replacing a different material with the same basename.
                target_texture = destination / (
                    f"{source_mtl_path.stem}_{len(copied_maps):02d}_{source_texture.name}"
                )
            shutil.copy2(source_texture, target_texture)

        tokens[-1] = target_texture.name
        indent = raw_line[: len(raw_line) - len(stripped)]
        rewritten.append(
            indent + " ".join(shlex.quote(token) for token in tokens)
        )
        copied_maps.append(target_texture)

    if not copied_maps:
        raise ValueError(f"No map_Kd diffuse textures found in {source_mtl_path}")

    output_mtl = destination / (output_name or source_mtl_path.name)
    output_mtl.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    # Fail fast if path rewriting ever leaves a dangling material reference.
    for texture_path in copied_maps:
        if not texture_path.is_file():
            raise FileNotFoundError(f"Copied diffuse map is missing: {texture_path}")
    return output_mtl



def export_obj_materials_to_glb(
    obj_path: str | Path,
    glb_path: str | Path,
) -> dict[str, int]:
    """Convert an already-textured OBJ/MTL package to GLB without rerunning TextureMesh.

    OpenMVS TextureMesh performs the expensive view-to-texture assignment before
    writing an export. Running it once for OBJ and again for GLB repeats that
    calculation. This converts the exact OBJ geometry, UVs and MTL diffuse maps
    into GLB using trimesh, preserving a single OpenMVS texture-generation pass.
    """
    import trimesh

    source_obj = Path(obj_path).expanduser().resolve()
    destination_glb = Path(glb_path).expanduser().resolve()
    if not source_obj.is_file():
        raise FileNotFoundError(f"Textured OBJ not found: {source_obj}")

    loaded = trimesh.load(str(source_obj), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise ValueError("Textured OBJ did not load as a mesh or scene.")

    if not scene.geometry:
        raise ValueError("Textured OBJ contains no mesh geometry.")

    vertex_count = sum(
        len(geometry.vertices)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )
    face_count = sum(
        len(geometry.faces)
        for geometry in scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )
    if vertex_count == 0 or face_count == 0:
        raise ValueError("Textured OBJ contains no usable vertices or faces.")

    textured_geometry = 0
    for geometry in scene.geometry.values():
        visual = getattr(geometry, "visual", None)
        material = getattr(visual, "material", None)
        if material is not None and (
            getattr(material, "image", None) is not None
            or getattr(material, "baseColorTexture", None) is not None
        ):
            textured_geometry += 1
    if textured_geometry == 0:
        raise ValueError(
            "Textured OBJ has no readable diffuse image; refusing to create an untextured GLB."
        )

    destination_glb.parent.mkdir(parents=True, exist_ok=True)
    exported = scene.export(file_obj=str(destination_glb), file_type="glb")
    if isinstance(exported, bytes):
        destination_glb.write_bytes(exported)
    if not destination_glb.is_file() or destination_glb.stat().st_size == 0:
        raise RuntimeError("OBJ-to-GLB conversion did not create a non-empty GLB file.")

    # Re-open the GLB to catch broken geometry or texture embedding before
    # reporting a successful full-quality export.
    check = trimesh.load(str(destination_glb), force="scene", process=False)
    if isinstance(check, trimesh.Trimesh):
        check_scene = trimesh.Scene(check)
    elif isinstance(check, trimesh.Scene):
        check_scene = check
    else:
        raise RuntimeError("Converted GLB could not be reopened as a mesh scene.")
    if not check_scene.geometry:
        raise RuntimeError("Converted GLB contains no mesh geometry.")
    check_vertices = sum(
        len(geometry.vertices)
        for geometry in check_scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )
    check_faces = sum(
        len(geometry.faces)
        for geometry in check_scene.geometry.values()
        if isinstance(geometry, trimesh.Trimesh)
    )
    if check_vertices == 0 or check_faces == 0:
        raise RuntimeError("Converted GLB contains no usable vertices or faces.")

    check_textures = sum(
        1
        for geometry in check_scene.geometry.values()
        if getattr(getattr(geometry, "visual", None), "material", None) is not None
        and getattr(
            getattr(geometry.visual, "material", None),
            "baseColorTexture",
            None,
        ) is not None
    )
    if check_textures == 0:
        raise RuntimeError("Converted GLB lost all embedded diffuse textures.")

    return {
        "obj_vertices": int(vertex_count),
        "obj_faces": int(face_count),
        "glb_vertices": int(check_vertices),
        "glb_faces": int(check_faces),
        "textured_geometry_count": int(textured_geometry),
        "glb_textured_geometry_count": int(check_textures),
    }
