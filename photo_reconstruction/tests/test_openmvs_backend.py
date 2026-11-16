"""Unit tests for the optional COLMAP + OpenMVS backend."""

from pathlib import Path

import pytest

from photo_reconstruction.openmvs_backend import OpenMVSBackend


def test_openmvs_requires_six_photos_before_tool_lookup(tmp_path: Path) -> None:
    backend = OpenMVSBackend(
        openmvs_dir=tmp_path,
        colmap_bin=str(tmp_path / "missing-colmap"),
    )
    photos = [tmp_path / f"photo_{index}.jpg" for index in range(5)]

    with pytest.raises(ValueError, match="at least 6 overlapping photographs"):
        backend.run(photos, session_dir=tmp_path)


def test_openmvs_reports_missing_colmap_for_valid_photo_count(tmp_path: Path) -> None:
    backend = OpenMVSBackend(
        openmvs_dir=tmp_path,
        colmap_bin=str(tmp_path / "missing-colmap"),
    )
    photos = [tmp_path / f"photo_{index}.jpg" for index in range(6)]

    with pytest.raises(FileNotFoundError, match="COLMAP executable not found"):
        backend.run(photos, session_dir=tmp_path)


def test_openmvs_resolves_executable_from_configured_directory(tmp_path: Path) -> None:
    binary = tmp_path / "DensifyPointCloud"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    backend = OpenMVSBackend(openmvs_dir=tmp_path)
    assert backend._binary("DensifyPointCloud") == str(binary)
