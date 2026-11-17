"""
photo_reconstruction/tests/test_phone_pipeline_regression.py — M7.3 Bug Fix Regression Suite
============================================================================================

Verifies:
  1. GUI-to-pipeline API compatibility (acceptance of quality_preset, etc.).
  2. Every quality preset ('draft', 'standard', 'high', 'maximum', 'quick') configuration and validation.
  3. Four-photo input reconstruction.
  4. Eleven-photo input verification (all 11 ingested, no silent dropping).
  5. Invalid image paths and empty inputs.
  6. Missing optional dependencies (OpenCV fallback when pycolmap is unavailable).
  7. Pipeline exceptions handled gracefully by background worker.
  8. Worker error reporting via Qt signals.
  9. UI button re-enabling, status updates, and photograph preservation.
 10. Successful reconstruction artifact generation (PLY, OBJ, report.json, poses).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from PySide6 import QtWidgets, QtCore

from photo_reconstruction.pipeline import (
    PhoneReconstructionPipeline,
    QUALITY_PRESETS,
    validate_quality_preset,
)
from photo_reconstruction.sfm_engine import SfMEngine
from viewer.phone_panel import PhoneReconstructionPanel, ReconstructionWorker


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QCoreApplication / QApplication instance exists for Qt signals/widgets."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# -----------------------------------------------------------------------------
# 1. GUI-to-Pipeline API Compatibility
# -----------------------------------------------------------------------------
def test_1_gui_to_pipeline_api_compatibility():
    """Verify run_pipeline signature accepts quality_preset and all GUI arguments without TypeError."""
    pipeline = PhoneReconstructionPipeline(quality_preset="standard")
    assert pipeline.quality_preset == "standard"

    # Verify run_pipeline accepts quality_preset as keyword argument without TypeError
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = pipeline.run_pipeline(
            image_paths=[],
            session_id="test_api_compat",
            reference_distance_m=None,
            progress_cb=lambda pct, msg: None,
            quality_preset="draft",
        )
        assert res["success"] is False
        assert res["quality_preset"] == "draft"
        assert pipeline.quality_preset == "draft"


# -----------------------------------------------------------------------------
# 2. Every Quality Preset Validation & Mapping
# -----------------------------------------------------------------------------
def test_2_every_quality_preset():
    """Verify all 4 primary presets ('draft', 'standard', 'high', 'maximum') plus 'quick'."""
    presets = ["draft", "quick", "standard", "high", "maximum"]
    for p in presets:
        val = validate_quality_preset(p)
        assert val in QUALITY_PRESETS
        cfg = QUALITY_PRESETS[val]
        assert "max_image_size" in cfg and cfg["max_image_size"] > 0
        assert "sift_max_features" in cfg and cfg["sift_max_features"] > 0
        assert "voxel_size" in cfg and cfg["voxel_size"] > 0.0
        assert "poisson_depth" in cfg and cfg["poisson_depth"] >= 7
        assert "cv_sift_features" in cfg and cfg["cv_sift_features"] > 0

    # Verify Draft has lower density/size and Maximum has highest
    assert QUALITY_PRESETS["draft"]["max_image_size"] < QUALITY_PRESETS["maximum"]["max_image_size"]
    assert QUALITY_PRESETS["draft"]["sift_max_features"] < QUALITY_PRESETS["maximum"]["sift_max_features"]
    assert QUALITY_PRESETS["draft"]["voxel_size"] > QUALITY_PRESETS["maximum"]["voxel_size"]

    # Verify invalid preset raises ValueError
    with pytest.raises(ValueError):
        validate_quality_preset("ultra_unsupported")

    with pytest.raises(TypeError):
        validate_quality_preset(12345)  # Non-string


# -----------------------------------------------------------------------------
# 3. Four-Photo Input
# -----------------------------------------------------------------------------
def test_3_four_photo_input():
    """Verify reconstruction with 4 orbital photographs and unscaled preservation."""
    dir_4orb = Path("data/phone_datasets/orbital_4photos")
    if not dir_4orb.exists():
        pytest.skip("Test dataset orbital_4photos not found.")

    photos = sorted(list(dir_4orb.glob("*.jpg")))
    assert len(photos) >= 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir), quality_preset="draft")
        res = pipeline.run_pipeline(
            image_paths=photos[:4],
            session_id="session_test_4photo",
            reference_distance_m=None,  # Unscaled
            quality_preset="draft",
        )
        assert res["success"] is True
        assert res["quality_preset"] == "draft"
        assert res["mesh_ply"].exists()
        assert res["report"]["scale_calibration"]["is_calibrated"] is False


# -----------------------------------------------------------------------------
# 4. Eleven-Photo Input Verification (No Silently Dropped Images)
# -----------------------------------------------------------------------------
def test_4_eleven_photo_input():
    """Verify that all 11 original image paths reach the pipeline and none are dropped."""
    dir_12 = Path("data/phone_datasets/orbital_12photos")
    if not dir_12.exists():
        pytest.skip("Test dataset orbital_12photos not found.")

    photos = sorted(list(dir_12.glob("*.jpg")))[:11]
    assert len(photos) == 11, f"Expected 11 photos, found {len(photos)}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir))
        session_dir = Path(tmp_dir) / "phone_reconstruction" / "session_11_test"

        # Track progress callbacks
        progress_events = []

        res = pipeline.run_pipeline(
            image_paths=photos,
            session_id="session_11_test",
            reference_distance_m=None,
            progress_cb=lambda pct, msg: progress_events.append((pct, msg)),
            quality_preset="draft",
        )

        assert res["success"] is True
        # Verify all 11 images are in original_images directory
        copied_images = list((session_dir / "original_images").glob("*.jpg"))
        assert len(copied_images) == 11, f"Expected 11 copied images, found {len(copied_images)}"
        assert res["report"]["image_counts"]["uploaded"] == 11
        assert len(progress_events) > 0


# -----------------------------------------------------------------------------
# 5. Invalid Image Paths & Missing Files
# -----------------------------------------------------------------------------
def test_5_invalid_image_paths():
    """Verify handling of nonexistent files, non-images, and empty inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir))

        # 1. Empty list
        res_empty = pipeline.run_pipeline([])
        assert res_empty["success"] is False
        assert "at least 2" in res_empty["report"]["notes"].lower() or "fewer than 2" in res_empty["report"]["notes"].lower()

        # 2. Nonexistent paths
        bad_paths = [Path("/invalid/dir/photo1.jpg"), Path("/invalid/dir/photo2.jpg")]
        res_bad = pipeline.run_pipeline(bad_paths)
        assert res_bad["success"] is False
        assert res_bad["report"]["image_counts"]["uploaded"] == 2
        assert res_bad["report"]["image_counts"]["registered"] == 0


# -----------------------------------------------------------------------------
# 6. Missing Optional Dependencies (Mocking pycolmap unavailable)
# -----------------------------------------------------------------------------
def test_6_missing_optional_dependencies(monkeypatch):
    """Verify that when pycolmap is unavailable, SfMEngine falls back to OpenCV SfM."""
    monkeypatch.setattr("photo_reconstruction.sfm_engine.HAS_PYCOLMAP", False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sfm = SfMEngine(work_dir=Path(tmp_dir) / "work", quality_preset="draft")
        dir_4orb = Path("data/phone_datasets/orbital_4photos")
        if not dir_4orb.exists():
            pytest.skip("Test dataset not found.")

        # Test running fallback
        res = sfm.run_reconstruction(dir_4orb)
        assert "camera_poses" in res
        assert "sparse_points" in res
        assert res["registered_count"] >= 2


# -----------------------------------------------------------------------------
# 7. Pipeline Exceptions Handled by Worker
# -----------------------------------------------------------------------------
def test_7_pipeline_exceptions(qapp, monkeypatch):
    """Verify that unexpected exceptions in the pipeline are caught by ReconstructionWorker."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir))

        # Monkeypatch run_pipeline to simulate unexpected exception
        def mock_failing_pipeline(*args, **kwargs):
            raise RuntimeError("Simulated unexpected pipeline failure")

        monkeypatch.setattr(pipeline, "run_pipeline", mock_failing_pipeline)

        worker = ReconstructionWorker(
            pipeline=pipeline,
            image_paths=[Path("img1.jpg"), Path("img2.jpg")],
            quality_preset="standard",
            reference_distance_m=None,
        )

        caught_errors = []
        worker.failed.connect(lambda err, tb: caught_errors.append((err, tb)))

        # Run worker synchronously
        worker.run()

        assert len(caught_errors) == 1
        err_msg, tb_str = caught_errors[0]
        assert "Simulated unexpected pipeline failure" in err_msg
        assert "Traceback" in tb_str or "RuntimeError" in tb_str


# -----------------------------------------------------------------------------
# 8. Worker Error Reporting via Qt Signals
# -----------------------------------------------------------------------------
def test_8_worker_error_reporting(qapp):
    """Verify worker emits failed signal with informative diagnostics when SfM fails."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir))

        # Run worker with 0 valid images
        worker = ReconstructionWorker(
            pipeline=pipeline,
            image_paths=[Path("/nonexistent1.jpg"), Path("/nonexistent2.jpg")],
            quality_preset="standard",
            reference_distance_m=None,
        )

        failed_payloads = []
        success_payloads = []
        worker.failed.connect(lambda err, tb: failed_payloads.append((err, tb)))
        worker.finished_success.connect(lambda res: success_payloads.append(res))

        worker.run()

        assert len(failed_payloads) == 1
        assert len(success_payloads) == 0
        err, _ = failed_payloads[0]
        assert "fewer than 2" in err.lower() or "required" in err.lower()


# -----------------------------------------------------------------------------
# 9. UI Button Re-enabling and Photo Preservation
# -----------------------------------------------------------------------------
def test_9_button_reenabling(qapp, monkeypatch):
    """Verify UI buttons are re-enabled, status updated, and photos preserved on failure and success."""
    # Prevent blocking modal dialogs during automated test
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        panel = PhoneReconstructionPanel(data_root=Path(tmp_dir))

        # Add mock photos to panel
        dummy_photos = [Path("/tmp/mock_a.jpg"), Path("/tmp/mock_b.jpg")]
        panel.selected_image_paths = list(dummy_photos)

        # Simulate start state
        panel.btn_start.setEnabled(False)
        panel.btn_add_photos.setEnabled(False)
        panel.lbl_status.setText("Initializing photogrammetry engine...")

        # 1. Test Failure recovery
        panel._on_reconstruction_failed("Mock error occurred", "Traceback info")
        assert panel.btn_start.isEnabled() is True
        assert panel.btn_add_photos.isEnabled() is True
        assert panel.prog_bar.value() == 0
        assert "Mock error occurred" in panel.lbl_status.text()
        # Ensure images were preserved!
        assert len(panel.selected_image_paths) == 2

        # 2. Test Success recovery
        panel.btn_start.setEnabled(False)
        panel.btn_add_photos.setEnabled(False)
        mock_success_res = {
            "success": True,
            "session_id": "session_success_test",
            "mesh_ply": Path(tmp_dir) / "mesh.ply",
        }
        panel._on_reconstruction_success(mock_success_res)
        assert panel.btn_start.isEnabled() is True
        assert panel.btn_add_photos.isEnabled() is True
        assert panel.prog_bar.value() == 100
        assert "Reconstruction successful" in panel.lbl_status.text()
        assert len(panel.selected_image_paths) == 2


# -----------------------------------------------------------------------------
# 10. Successful Reconstruction Completion Artifacts
# -----------------------------------------------------------------------------
def test_10_successful_reconstruction_completion():
    """Verify all expected 3D artifacts (PLY, OBJ, report.json, poses) are generated."""
    dir_4orb = Path("data/phone_datasets/orbital_4photos")
    if not dir_4orb.exists():
        pytest.skip("Test dataset orbital_4photos not found.")

    photos = sorted(list(dir_4orb.glob("*.jpg")))[:4]

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = PhoneReconstructionPipeline(data_root=Path(tmp_dir))
        res = pipeline.run_pipeline(
            image_paths=photos,
            session_id="session_artifacts_test",
            reference_distance_m=None,
            quality_preset="draft",
        )

        assert res["success"] is True
        session_dir = res["session_dir"]
        assert (session_dir / "mesh" / "scene_mesh.ply").exists()
        assert (session_dir / "mesh" / "scene_mesh.obj").exists()
        assert (session_dir / "dense_cloud" / "dense_cloud.ply").exists()
        assert (session_dir / "report.json").exists()
        assert len(res["camera_poses"]) >= 2
        assert res["quality_preset"] == "draft"



# -----------------------------------------------------------------------------
# 11. OpenMVS Worker Input Manifest JSON Scope Regression
# -----------------------------------------------------------------------------
def test_openmvs_worker_writes_input_manifest_before_backend(
    qapp, monkeypatch, tmp_path
):
    """The OpenMVS worker must serialize its input manifest before starting tools."""
    import json

    from PIL import Image
    from photo_reconstruction import openmvs_backend

    image_paths = []
    for index in range(6):
        path = tmp_path / f"openmvs_input_{index}.jpg"
        Image.new("RGB", (48, 48), (index * 30, 80, 140)).save(path, format="JPEG")
        image_paths.append(path)

    manifests = []

    class FakeOpenMVSBackend:
        def __init__(self, **kwargs):
            pass

        def run(self, image_paths, session_dir, **kwargs):
            manifest_path = Path(session_dir) / "input_manifest.json"
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            raise RuntimeError("Fake OpenMVS backend reached after manifest write")

    monkeypatch.setattr(openmvs_backend, "OpenMVSBackend", FakeOpenMVSBackend)

    pipeline = PhoneReconstructionPipeline(data_root=tmp_path)
    worker = ReconstructionWorker(
        pipeline=pipeline,
        image_paths=image_paths,
        quality_preset="standard",
        reference_distance_m=None,
        backend="openmvs",
    )
    failures = []
    worker.failed.connect(lambda message, tb: failures.append((message, tb)))

    worker.run()

    assert len(manifests) == 1
    assert manifests[0]["uploaded_count"] == 6
    assert manifests[0]["sfm_ready_count"] == 6
    assert len(manifests[0]["images"]) == 6
    assert len(failures) == 1
    assert "Fake OpenMVS backend reached" in failures[0][0]
    assert "cannot access local variable 'json'" not in failures[0][0]
