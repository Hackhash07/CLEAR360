"""Automatic reconstruction-backend selection for CLEAR360."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping


OPENMVS_EXECUTABLES = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)


def _find_executable(
    name: str,
    openmvs_dir: Path | None,
    environ: Mapping[str, str],
) -> str | None:
    """Find an executable in the configured OpenMVS folder or PATH."""
    if openmvs_dir is not None:
        candidate = openmvs_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name, path=environ.get("PATH"))


def resolve_reconstruction_backend(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Return ('openmvs'|'native', human-readable reason) without user choices.

    COLMAP + OpenMVS is selected only when COLMAP and every required OpenMVS
    executable are present. Otherwise the existing CLEAR360 pipeline is used.
    """
    env = os.environ if environ is None else environ
    configured_dir = env.get("CLEAR360_OPENMVS_DIR", "").strip()
    openmvs_dir = Path(configured_dir).expanduser() if configured_dir else None

    configured_colmap = env.get("CLEAR360_COLMAP_BIN", "").strip()
    colmap_path = None
    if configured_colmap:
        candidate = Path(configured_colmap).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            colmap_path = str(candidate)
    if colmap_path is None:
        colmap_path = shutil.which("colmap", path=env.get("PATH"))

    missing = [
        name for name in OPENMVS_EXECUTABLES
        if _find_executable(name, openmvs_dir, env) is None
    ]
    if colmap_path and not missing:
        return "openmvs", "COLMAP + OpenMVS detected"

    details = []
    if not colmap_path:
        details.append("COLMAP not found")
    if missing:
        details.append("missing OpenMVS tools: " + ", ".join(missing))
    reason = "; ".join(details) if details else "OpenMVS tools unavailable"
    return "native", f"using built-in CLEAR360 engine ({reason})"
