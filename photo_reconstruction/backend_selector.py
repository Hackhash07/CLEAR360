"""Automatic reconstruction-backend selection for CLEAR360."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

from photo_reconstruction.openmvs_tools import (
    EXECUTABLE_NAMES,
    configured_openmvs_root,
    find_all_openmvs_executables,
)


OPENMVS_EXECUTABLES = EXECUTABLE_NAMES


def resolve_reconstruction_backend(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Return ('openmvs'|'native', human-readable reason) without user choices.

    Backend detection and the reconstruction worker share the same executable
    resolver, including configured roots, common bin/build layouts, and PATH.
    This prevents selecting OpenMVS in the UI when the worker cannot launch it.
    """
    env = os.environ if environ is None else environ
    openmvs_dir = configured_openmvs_root(env)

    configured_colmap = env.get("CLEAR360_COLMAP_BIN", "").strip()
    colmap_path = None
    if configured_colmap:
        candidate = Path(configured_colmap).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            colmap_path = str(candidate.resolve())
    if colmap_path is None:
        colmap_path = shutil.which("colmap", path=env.get("PATH"))

    tools = find_all_openmvs_executables(openmvs_dir, path=env.get("PATH"))
    missing = [name for name, executable in tools.items() if executable is None]
    if colmap_path and not missing:
        return "openmvs", "COLMAP + OpenMVS detected automatically"

    details = []
    if not colmap_path:
        details.append("COLMAP not found")
    if missing:
        details.append("missing OpenMVS tools: " + ", ".join(missing))
    reason = "; ".join(details) if details else "OpenMVS tools unavailable"
    return "native", f"using built-in CLEAR360 engine ({reason})"
