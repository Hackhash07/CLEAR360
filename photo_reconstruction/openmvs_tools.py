"""Consistent OpenMVS executable discovery for the GUI and reconstruction worker."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping


EXECUTABLE_NAMES = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)


def configured_openmvs_root(environ: Mapping[str, str] | None = None) -> Path | None:
    """Resolve the configured or conventional per-user OpenMVS install folder."""
    env = os.environ if environ is None else environ
    configured = env.get("CLEAR360_OPENMVS_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    default = Path.home() / "Applications" / "OpenMVS-2.4.0"
    return default if default.is_dir() else None


def _candidate_paths(root: Path, name: str) -> list[Path]:
    """List common layouts before considering a recursive search."""
    relative_locations = (
        Path(name),
        Path("bin") / name,
        Path("build") / "bin" / name,
        Path("build") / name,
        Path("usr") / "local" / "bin" / name,
        Path("install") / "bin" / name,
    )
    return [root / rel for rel in relative_locations]


def find_openmvs_executable(
    name: str,
    openmvs_dir: Path | str | None = None,
    path: str | None = None,
) -> str | None:
    """Find an executable in common install layouts, a nested build, or PATH."""
    root = Path(openmvs_dir).expanduser() if openmvs_dir else configured_openmvs_root()
    if root is not None and root.is_dir():
        for candidate in _candidate_paths(root, name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

        # Some CMake installs place targets under a generated platform/config
        # directory. Search by exact executable name as a final folder lookup.
        try:
            for candidate in root.rglob(name):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate.resolve())
        except OSError:
            pass

    return shutil.which(name, path=path)


def find_all_openmvs_executables(
    openmvs_dir: Path | str | None = None,
    path: str | None = None,
) -> dict[str, str | None]:
    """Resolve all required OpenMVS tools using exactly the same search rules."""
    return {
        name: find_openmvs_executable(name, openmvs_dir=openmvs_dir, path=path)
        for name in EXECUTABLE_NAMES
    }
