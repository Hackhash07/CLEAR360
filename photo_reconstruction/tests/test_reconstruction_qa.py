from pathlib import Path
import subprocess
import sys

import numpy as np
import trimesh

from photo_reconstruction.quality_report import QualityReportGenerator
from photo_reconstruction.reconstruction_qa import (
    audit_camera_overlap_from_track_ids,
    audit_obj_materials,
    audit_reconstruction_outputs,
)


def _write_open_quad(path: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    mesh.export(path)


def test_open_scene_surface_is_not_mistaken_for_a_failed_solid(tmp_path: Path):
    mesh_path = tmp_path / "scene.ply"
    _write_open_quad(mesh_path)

    report = audit_reconstruction_outputs(mesh_path)

    assert report["status"] == "healthy"
    assert report["score"] >= 90.0
    assert report["geometry"]["surface_type"] == "open_surface"
    assert report["policy"]["watertight_mesh_required"] is False
    assert report["checks"]["open_surface_allowed"] is True


def test_disconnected_parts_are_reported_without_automatic_deletion(tmp_path: Path):
    mesh_path = tmp_path / "two_parts.ply"
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )
    mesh.export(mesh_path)

    report = audit_reconstruction_outputs(mesh_path)

    assert report["geometry"]["connected_components"] == 2
    assert len(mesh.faces) == 2
    assert any("disconnected components" in warning for warning in report["warnings"])
    assert report["policy"]["automatic_component_deletion"] is False


def test_camera_overlap_graph_reports_connected_and_disconnected_views():
    tracks = {
        "camera_a.jpg": set(range(1, 21)),
        "camera_b.jpg": set(range(1, 41)),
        "camera_c.jpg": set(range(21, 61)),
        "camera_isolated.jpg": {100, 101},
    }

    report = audit_camera_overlap_from_track_ids(tracks)

    assert report["available"] is True
    assert report["component_count"] == 2
    assert report["status"] == "disconnected"
    assert report["supported_pair_count"] == 2
    assert report["isolated_cameras"] == ["camera_isolated.jpg"]
    assert report["warnings"]


def test_single_registered_camera_is_insufficient_for_overlap():
    report = audit_camera_overlap_from_track_ids(
        {"camera_a.jpg": set(range(1, 21))}
    )

    assert report["available"] is True
    assert report["status"] == "insufficient"
    assert report["component_count"] == 1
    assert report["warnings"]


def test_camera_overlap_is_unavailable_without_sparse_track_evidence():
    report = audit_camera_overlap_from_track_ids(
        {"camera_a.jpg": set(), "camera_b.jpg": set()}
    )

    assert report["available"] is False
    assert report["status"] == "unavailable"
    assert report["component_count"] is None


def test_obj_material_and_diffuse_map_references_are_checked(tmp_path: Path):
    obj = tmp_path / "scene.obj"
    mtl = tmp_path / "scene.mtl"
    texture = tmp_path / "diffuse.png"
    texture.write_bytes(b"test texture")

    obj.write_text("mtllib scene.mtl\n", encoding="utf-8")
    mtl.write_text(
        "newmtl material_0\n"
        "Kd 1.0 1.0 1.0\n"
        "map_Kd diffuse.png\n",
        encoding="utf-8",
    )

    valid = audit_obj_materials(obj)
    assert valid["all_references_resolve"] is True
    assert valid["diffuse_maps"] == [str(texture.resolve())]

    texture.unlink()
    invalid = audit_obj_materials(obj)
    assert invalid["all_references_resolve"] is False
    assert invalid["missing_diffuse_maps"]


def test_reconstruction_qa_records_unresolved_obj_texture(tmp_path: Path):
    mesh_path = tmp_path / "scene.ply"
    obj_path = tmp_path / "scene.obj"
    _write_open_quad(mesh_path)
    obj_path.write_text("mtllib missing.mtl\n", encoding="utf-8")

    report = audit_reconstruction_outputs(mesh_path, obj_path=obj_path)

    assert report["status"] == "review"
    assert report["geometry_status"] == "healthy"
    assert report["checks"]["obj_material_references_resolve"] is False
    assert any("diffuse-texture references" in warning for warning in report["warnings"])


def test_reconstruction_report_preserves_backend_independent_mesh_quality():
    mesh_quality = {
        "schema_version": 1,
        "status": "review",
        "score": 84.5,
        "warnings": ["Inspect disconnected components."],
    }
    camera_overlap = {"available": True, "status": "connected", "warnings": []}
    photo_set = {"status": "review", "can_reconstruct": True, "warnings": ["One blurry image."]}
    report = QualityReportGenerator.generate_report(
        session_id="session_test",
        uploaded_count=6,
        registered_count=5,
        failed_images=["frame_006.jpg"],
        sparse_point_count=1200,
        dense_point_count=25000,
        mesh_triangle_count=45000,
        reprojection_error_px=0.9,
        processing_time_s=12.4,
        scale_info={"is_calibrated": False},
        mesh_quality=mesh_quality,
        camera_overlap=camera_overlap,
        photo_set=photo_set,
    )

    assert report["mesh_quality"] == mesh_quality
    assert report["camera_overlap"] == camera_overlap
    assert report["photo_set"] == photo_set
    assert report["diagnostics"]["photo_set_warnings"] == photo_set["warnings"]
    assert report["image_counts"]["registered"] == 5

def _report_with_quality(mesh_quality=None, camera_overlap=None, photo_set=None):
    return QualityReportGenerator.generate_report(
        session_id="session_grade",
        uploaded_count=4,
        registered_count=4,
        failed_images=[],
        sparse_point_count=1000,
        dense_point_count=10000,
        mesh_triangle_count=20000,
        reprojection_error_px=0.5,
        processing_time_s=30.0,
        scale_info={"is_calibrated": False},
        mesh_quality=mesh_quality,
        camera_overlap=camera_overlap,
        photo_set=photo_set,
    )


def test_optimal_grade_requires_healthy_mesh_and_connected_camera_graph():
    report = _report_with_quality(
        mesh_quality={"status": "healthy"},
        camera_overlap={"available": True, "status": "connected"},
        photo_set={"status": "ready"},
    )
    assert report["reconstruction_grade"] == "OPTIMAL"


def test_export_or_geometry_review_prevents_optimal_grade():
    report = _report_with_quality(
        mesh_quality={"status": "review", "geometry_status": "healthy"},
        camera_overlap={"available": True, "status": "connected"},
        photo_set={"status": "ready"},
    )
    assert report["reconstruction_grade"] == "ACCEPTABLE"


def test_poor_mesh_quality_marks_reconstruction_failed():
    report = _report_with_quality(
        mesh_quality={"status": "poor"},
        camera_overlap={"available": True, "status": "connected"},
        photo_set={"status": "ready"},
    )
    assert report["reconstruction_grade"] == "FAILED"


def test_review_photo_set_prevents_optimal_grade():
    report = _report_with_quality(
        mesh_quality={"status": "healthy"},
        camera_overlap={"available": True, "status": "connected"},
        photo_set={"status": "review", "warnings": ["Exact duplicates detected."]},
    )
    assert report["reconstruction_grade"] == "ACCEPTABLE"


def test_missing_reprojection_residuals_cannot_receive_optimal_grade():
    report = QualityReportGenerator.generate_report(
        session_id="session_without_residuals",
        uploaded_count=6,
        registered_count=6,
        failed_images=[],
        sparse_point_count=0,
        dense_point_count=1000,
        mesh_triangle_count=2000,
        reprojection_error_px=None,
        processing_time_s=12.4,
        scale_info={"is_calibrated": False},
    )

    assert report["reconstruction_grade"] == "ACCEPTABLE"
    assert report["geometric_metrics"]["mean_reprojection_error_px"] is None

def test_lightweight_qa_import_does_not_load_the_desktop_or_ml_runtime():
    project_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        "import photo_reconstruction.reconstruction_qa; "
        "assert 'open3d' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'PySide6' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
