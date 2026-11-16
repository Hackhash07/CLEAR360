from pathlib import Path

import numpy as np
from PIL import Image

from photo_reconstruction.guided_surface import build_guided_photo_model


def test_guided_photo_model_writes_textured_obj_package(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    yy, xx = np.mgrid[0:80, 0:120]
    rgb = np.dstack((xx * 2, yy * 3, (xx + yy) % 255)).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(image_path)

    session_dir = tmp_path / "session_001"
    result = build_guided_photo_model(
        image_path,
        [
            {
                "name": "fountain",
                "points": [[30, 15], [90, 15], [105, 65], [20, 65]],
                "depth_cm": 8.0,
            }
        ],
        session_dir,
        width_m=2.0,
        grid_step_px=8,
    )

    obj_path = result["mesh_obj"]
    mtl_path = obj_path.with_suffix(".mtl")
    texture_path = result["texture_png"]

    assert obj_path.is_file()
    assert mtl_path.is_file()
    assert texture_path.is_file()
    assert (session_dir / "report.json").is_file()

    obj_text = obj_path.read_text(encoding="utf-8")
    assert "mtllib scene_mesh.mtl" in obj_text
    assert "usemtl guided_photo" in obj_text
    assert "vt " in obj_text
    assert "f " in obj_text
    assert "map_Kd scene_mesh.png" in mtl_path.read_text(encoding="utf-8")

    face_count = sum(line.startswith("f ") for line in obj_text.splitlines())
    assert face_count == result["report"]["triangle_count"]
    assert result["report"]["region_count"] == 1
    assert result["report"]["vertex_count"] > 0


def test_guided_photo_model_rejects_invalid_regions(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(image_path)

    try:
        build_guided_photo_model(
            image_path,
            [{"name": "bad", "points": [[5, 5], [30, 30]], "depth_cm": 5}],
            tmp_path / "session_002",
        )
    except ValueError as exc:
        assert "at least three polygon points" in str(exc)
    else:
        raise AssertionError("Expected an invalid polygon to be rejected.")
