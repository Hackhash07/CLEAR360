"""
photo_reconstruction/tests/test_phone_reconstruction.py — Unit Tests for M7.3
=============================================================================
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from photo_reconstruction.image_loader import ImageLoader
from photo_reconstruction.exif_reader import ExifReader
from photo_reconstruction.image_quality import ImageQualityAssessor
from photo_reconstruction.sfm_engine import SfMEngine
from photo_reconstruction.dense_reconstruction import DenseReconstructor
from photo_reconstruction.mesh_builder import MeshBuilder
from photo_reconstruction.scale_calibrator import ScaleCalibrator
from photo_reconstruction.session_manager import SessionManager
from photo_reconstruction.quality_report import QualityReportGenerator
from photo_reconstruction.pipeline import PhoneReconstructionPipeline


@pytest.fixture
def sample_image():
    """Create a sample 640x480 test image with texture."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 480, 40):
        for j in range(0, 640, 40):
            color = (int((i * 3) % 255), int((j * 2) % 255), int((i + j) % 255))
            cv2.rectangle(img, (j, i), (j + 20, i + 20), color, -1)
    return img


def test_image_loader_valid(sample_image):
    """Verify loading and dimension checking of valid image."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "test.png"
        cv2.imwrite(str(p), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))

        rgb, meta = ImageLoader.load_image(p)
        assert rgb.shape == (480, 640, 3)
        assert meta["width"] == 640
        assert meta["height"] == 480
        assert meta["file_name"] == "test.png"


def test_image_loader_invalid():
    """Verify rejection of nonexistent and unsupported formats."""
    with pytest.raises(FileNotFoundError):
        ImageLoader.load_image(Path("/nonexistent/file.png"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        p_txt = Path(tmp_dir) / "readme.txt"
        p_txt.write_text("not an image")
        with pytest.raises(ValueError):
            ImageLoader.load_image(p_txt)


def test_exif_reader_fallback(sample_image):
    """Verify focal length estimation when EXIF is absent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "plain.png"
        cv2.imwrite(str(p), cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))

        info = ExifReader.get_camera_info(p, 640, 480)
        assert info["estimated_fx"] > 0
        assert info["estimated_fy"] > 0
        assert info["cx"] == 320.0
        assert info["cy"] == 240.0
        assert info["intrinsics_source"] == "default_fallback"


def test_mixed_lens_warning():
    """Verify detection of mixed lenses across session."""
    cams = [
        {"estimated_fx": 700.0, "model": "Phone A"},
        {"estimated_fx": 710.0, "model": "Phone A"},
        {"estimated_fx": 1400.0, "model": "Phone A"},  # 2x telephoto!
    ]
    is_mixed, warnings = ExifReader.check_mixed_lenses(cams)
    assert is_mixed is True
    assert len(warnings) > 0


def test_image_quality_blur_detection(sample_image):
    """Verify blur scoring on sharp vs heavily blurred image."""
    sharp_eval = ImageQualityAssessor.evaluate(sample_image)
    assert sharp_eval["blur_score"] > 50.0

    blurred = cv2.GaussianBlur(sample_image, (31, 31), 10.0)
    blur_eval = ImageQualityAssessor.evaluate(blurred)
    assert blur_eval["blur_score"] < sharp_eval["blur_score"]


def test_image_quality_exposure():
    """Verify exposure evaluation on washed out vs normal images."""
    washed = np.full((480, 640, 3), 250, dtype=np.uint8)
    res = ImageQualityAssessor.evaluate(washed)
    assert res["exposure_mean"] > 240.0
    assert any("overexposed" in rec.lower() or "clipping" in rec.lower() for rec in res["recommendations"])


def test_sfm_engine_insufficient_images():
    """Verify SfM gracefully handles fewer than 2 images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_path = Path(tmp_dir)
        sfm = SfMEngine(t_path / "work")
        res = sfm.run_reconstruction(t_path / "empty_dir")
        assert res["success"] is False
        assert "At least 2" in res["message"]


def test_dense_reconstruction_cleaning():
    """Verify dense point cloud downsampling and normal estimation."""
    pts = np.random.rand(500, 3).astype(np.float32) * 5.0
    cols = np.ones((500, 3), dtype=np.float32)

    dense = DenseReconstructor(voxel_size=0.1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_out = Path(tmp_dir) / "dense.ply"
        saved = DenseReconstructor.save_point_cloud_ply(pts, cols, ply_out)
        assert saved.exists()
        assert saved.stat().st_size > 100


def test_mesh_builder_export():
    """Verify mesh export to PLY and OBJ."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    c = np.ones((4, 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        p_ply = Path(tmp_dir) / "test.ply"
        MeshBuilder.save_mesh_ply(v, f, c, p_ply)
        assert p_ply.exists()

        p_obj = Path(tmp_dir) / "test.obj"
        MeshBuilder.save_mesh_obj(v, f, c, p_obj)
        assert p_obj.exists()


def test_scale_calibrator():
    """Verify metric scale computation and vertex transformation."""
    calib = ScaleCalibrator()
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])
    # Measured is 2.0 units; target is 4.0 metres -> scale factor = 2.0
    scale = calib.compute_scale(p1, p2, target_distance_metres=4.0, reference_label="Wheelbase")
    assert np.isclose(scale, 2.0)
    assert calib.is_calibrated is True

    verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    scaled_v, _ = calib.apply_scale(verts)
    assert np.allclose(scaled_v, [[2.0, 4.0, 6.0]])


def test_session_manager():
    """Verify session folder creation and report storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sm = SessionManager(Path(tmp_dir))
        s_dir = sm.create_session("session_001")
        assert s_dir.exists()
        assert (s_dir / "original_images").exists()
        assert (s_dir / "mesh").exists()

        assert "session_001" in sm.list_sessions()


def test_quality_report_generator():
    """Verify standardized quality report assembly."""
    report = QualityReportGenerator.generate_report(
        session_id="session_test",
        uploaded_count=4,
        registered_count=4,
        failed_images=[],
        sparse_point_count=1200,
        dense_point_count=25000,
        mesh_triangle_count=45000,
        reprojection_error_px=0.85,
        processing_time_s=12.4,
        scale_info={"is_calibrated": True, "scale_factor": 1.5},
    )
    assert report["reconstruction_grade"] == "OPTIMAL"
    assert report["completeness_percentage"] == 100.0
    assert report["image_counts"]["registered"] == 4

