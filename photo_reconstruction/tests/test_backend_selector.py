from pathlib import Path

from photo_reconstruction.backend_selector import (
    OPENMVS_EXECUTABLES,
    resolve_reconstruction_backend,
)


def _make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def test_selects_openmvs_when_all_tools_are_available(tmp_path: Path):
    openmvs_dir = tmp_path / "openmvs"
    colmap = tmp_path / "colmap" / "colmap"
    _make_executable(colmap)
    for name in OPENMVS_EXECUTABLES:
        _make_executable(openmvs_dir / name)

    backend, reason = resolve_reconstruction_backend(
        {
            "CLEAR360_OPENMVS_DIR": str(openmvs_dir),
            "CLEAR360_COLMAP_BIN": str(colmap),
            "PATH": "",
        }
    )

    assert backend == "openmvs"
    assert "detected" in reason.lower()


def test_falls_back_to_native_when_openmvs_tool_is_missing(tmp_path: Path):
    openmvs_dir = tmp_path / "openmvs"
    colmap = tmp_path / "colmap" / "colmap"
    _make_executable(colmap)
    for name in OPENMVS_EXECUTABLES[:-1]:
        _make_executable(openmvs_dir / name)

    backend, reason = resolve_reconstruction_backend(
        {
            "CLEAR360_OPENMVS_DIR": str(openmvs_dir),
            "CLEAR360_COLMAP_BIN": str(colmap),
            "PATH": "",
        }
    )

    assert backend == "native"
    assert "TextureMesh" in reason


def test_falls_back_to_native_when_colmap_is_missing(tmp_path: Path):
    openmvs_dir = tmp_path / "openmvs"
    for name in OPENMVS_EXECUTABLES:
        _make_executable(openmvs_dir / name)

    backend, reason = resolve_reconstruction_backend(
        {
            "CLEAR360_OPENMVS_DIR": str(openmvs_dir),
            "CLEAR360_COLMAP_BIN": "",
            "PATH": "",
        }
    )

    assert backend == "native"
    assert "COLMAP not found" in reason



def test_selects_openmvs_when_tools_are_inside_bin_folder(tmp_path: Path):
    openmvs_root = tmp_path / "OpenMVS-2.4.0"
    colmap = tmp_path / "colmap" / "colmap"
    _make_executable(colmap)
    for name in OPENMVS_EXECUTABLES:
        _make_executable(openmvs_root / "bin" / name)

    backend, reason = resolve_reconstruction_backend(
        {
            "CLEAR360_OPENMVS_DIR": str(openmvs_root),
            "CLEAR360_COLMAP_BIN": str(colmap),
            "PATH": "",
        }
    )

    assert backend == "openmvs"
    assert "detected" in reason.lower()


def test_finds_openmvs_tool_in_nested_build_directory(tmp_path: Path):
    from photo_reconstruction.openmvs_tools import find_openmvs_executable

    openmvs_root = tmp_path / "OpenMVS-2.4.0"
    executable = openmvs_root / "build" / "apps" / "DensifyPointCloud"
    _make_executable(executable)

    found = find_openmvs_executable(
        "DensifyPointCloud",
        openmvs_dir=openmvs_root,
        path="",
    )

    assert found == str(executable.resolve())
