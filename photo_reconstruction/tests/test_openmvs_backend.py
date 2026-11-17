"""Unit tests for the optional COLMAP + OpenMVS backend."""

from pathlib import Path

import pytest

from photo_reconstruction.openmvs_backend import OpenMVSBackend


def test_colmap_camera_mode_shares_calibration_for_consistent_lens_sets():
    from photo_reconstruction.openmvs_backend import resolve_colmap_camera_mode

    mode = resolve_colmap_camera_mode({
        "camera": {"mixed_lenses_detected": False, "warnings": []}
    })

    assert mode["single_camera"] == "1"
    assert mode["single_camera_per_image"] == "0"
    assert mode["strategy"] == "shared_camera"


def test_colmap_camera_mode_separates_mixed_lens_calibrations():
    from photo_reconstruction.openmvs_backend import resolve_colmap_camera_mode

    mode = resolve_colmap_camera_mode({
        "camera": {
            "mixed_lenses_detected": True,
            "warnings": ["Mixed camera lenses detected across session."],
        }
    })

    assert mode["single_camera"] == "0"
    assert mode["single_camera_per_image"] == "1"
    assert mode["strategy"] == "per_image_camera"
    assert "Mixed camera lenses" in mode["reason"]


def test_colmap_camera_mode_separates_different_image_resolutions():
    from photo_reconstruction.openmvs_backend import resolve_colmap_camera_mode

    mode = resolve_colmap_camera_mode({
        "camera": {"mixed_lenses_detected": False, "warnings": []},
        "image_dimensions": {
            "dimensions": [
                {"width": 4032, "height": 3024},
                {"width": 1920, "height": 1080},
            ]
        },
    })

    assert mode["single_camera"] == "0"
    assert mode["single_camera_per_image"] == "1"
    assert mode["strategy"] == "per_image_camera"
    assert "Different image dimensions" in mode["reason"]


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

def test_openmvs_registration_summary_reports_registered_and_failed_uploads(tmp_path: Path):
    from photo_reconstruction.openmvs_backend import _colmap_registration_summary

    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "1 1 0 0 0 0 0 0 1 photo_01.jpg\n"
        "1 2 3 4 5 6 7 8 9 10\n"
        "2 1 0 0 0 0 0 0 1 photo_02.jpg\n"
        "1 2 3 4 5 6 7 8 9 10\n",
        encoding="utf-8",
    )
    points3d_txt = tmp_path / "points3D.txt"
    points3d_txt.write_text(
        "1 0 0 1 255 0 0 0.2 1 0 2 1\n"
        "2 0 1 1 0 255 0 0.4 1 1 2 2\n",
        encoding="utf-8",
    )

    summary = _colmap_registration_summary(
        images_txt,
        [
            tmp_path / "photo_01.jpg",
            tmp_path / "photo_02.jpg",
            tmp_path / "photo_03.jpg",
        ],
        points3d_txt,
    )

    assert summary["uploaded_count"] == 3
    assert summary["registered_count"] == 2
    assert summary["failed_images"] == ["photo_03.jpg"]
    assert summary["sparse_point_count"] == 2
    assert summary["mean_reprojection_error_px"] == 0.3
    assert summary["registration_completeness_percentage"] == 66.7
    assert summary["camera_overlap"]["component_count"] == 2
    assert summary["camera_overlap"]["supported_pair_count"] == 0
    assert summary["camera_overlap"]["warnings"]

def test_openmvs_camera_graph_connects_views_with_fifteen_shared_tracks(tmp_path: Path):
    from photo_reconstruction.openmvs_backend import _colmap_registration_summary

    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "1 1 0 0 0 0 0 0 1 photo_01.jpg\n"
        "1 2 3 4 5 6 7 8 9 10\n"
        "2 1 0 0 0 0 0 0 1 photo_02.jpg\n"
        "1 2 3 4 5 6 7 8 9 10\n",
        encoding="utf-8",
    )
    points3d_txt = tmp_path / "points3D.txt"
    points3d_txt.write_text(
        "".join(
            f"{index} 0 0 1 255 0 0 0.2 1 {index} 2 {index}\n"
            for index in range(1, 16)
        ),
        encoding="utf-8",
    )

    summary = _colmap_registration_summary(
        images_txt,
        [tmp_path / "photo_01.jpg", tmp_path / "photo_02.jpg"],
        points3d_txt,
    )

    overlap = summary["camera_overlap"]
    assert overlap["component_count"] == 1
    assert overlap["supported_pair_count"] == 1
    assert overlap["supported_pairs"][0]["shared_3d_tracks"] == 15
    assert overlap["warnings"] == []


def test_colmap_cli_uses_current_feature_extraction_option_groups():
    from photo_reconstruction.openmvs_backend import _colmap_feature_extraction_options

    options = _colmap_feature_extraction_options(max_image_size=1600, max_num_features=2500)

    assert options == [
        "--FeatureExtraction.max_image_size", "1600",
        "--SiftExtraction.max_num_features", "2500",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
        "--FeatureExtraction.use_gpu", "0",
    ]
    assert "--SiftExtraction.max_image_size" not in options
    assert "--SiftExtraction.use_gpu" not in options


def test_colmap_cli_uses_current_feature_matching_option_group():
    from photo_reconstruction.openmvs_backend import _colmap_feature_matching_options

    assert _colmap_feature_matching_options() == [
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.guided_matching", "1",
    ]



def test_colmap_feature_extraction_uses_robust_sift_options():
    from photo_reconstruction.openmvs_backend import _colmap_feature_extraction_options

    options = _colmap_feature_extraction_options(max_image_size=1600, max_num_features=6000)
    pairs = dict(zip(options[0::2], options[1::2]))

    assert pairs["--FeatureExtraction.max_image_size"] == "1600"
    assert pairs["--SiftExtraction.max_num_features"] == "6000"
    assert pairs["--SiftExtraction.estimate_affine_shape"] == "1"
    assert pairs["--SiftExtraction.domain_size_pooling"] == "1"
    assert pairs["--FeatureExtraction.use_gpu"] == "0"
    assert "--SiftExtraction.max_image_size" not in options


def test_colmap_matching_uses_geometry_guided_sift_matching():
    from photo_reconstruction.openmvs_backend import _colmap_feature_matching_options

    options = _colmap_feature_matching_options()
    pairs = dict(zip(options[0::2], options[1::2]))

    assert pairs["--FeatureMatching.use_gpu"] == "0"
    assert pairs["--FeatureMatching.guided_matching"] == "1"


def test_quality_fusion_requires_three_supporting_views_but_draft_allows_two():
    from photo_reconstruction.openmvs_backend import _openmvs_fusion_minimum_views

    assert _openmvs_fusion_minimum_views("standard") == 3
    assert _openmvs_fusion_minimum_views("high") == 3
    assert _openmvs_fusion_minimum_views("maximum") == 3
    assert _openmvs_fusion_minimum_views("draft") == 2
    assert _openmvs_fusion_minimum_views("quick") == 2
