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
