from pathlib import Path

import numpy as np

from photo_reconstruction.scene_orientation import (
    align_obj_file,
    estimate_world_up,
    quaternion_to_rotation,
    rotation_from_to,
)


def test_identity_camera_pose_reports_camera_up_as_negative_world_y(tmp_path: Path):
    images_txt = tmp_path / "images.txt"
    rows = []
    for image_id in range(1, 5):
        rows.extend([
            f"{image_id} 1 0 0 0 0 0 0 1 image_{image_id}.jpg",
            # Point observations have many numeric fields but are not pose rows.
            "1 2 3 4 5 6 7 8 9 10 11 12",
        ])
    images_txt.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = estimate_world_up(images_txt)

    assert result is not None
    up, consistency, pose_count = result
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0], atol=1e-6)
    assert consistency > 0.99
    assert pose_count == 4


def test_rotation_from_to_maps_estimated_up_to_positive_z():
    source = np.array([0.0, -1.0, 0.0])
    rotation = rotation_from_to(source, np.array([0.0, 0.0, 1.0]))
    aligned = rotation @ source
    np.testing.assert_allclose(aligned, [0.0, 0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) > 0.999


def test_alignment_rotates_obj_vertices_and_normals_but_not_uvs(tmp_path: Path):
    obj = tmp_path / "scene.obj"
    obj.write_text(
        "mtllib scene.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "vn 0 -1 0\n"
        "f 1/1/1 2/2/1 3/3/1\n",
        encoding="utf-8",
    )
    rotation = rotation_from_to(np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert align_obj_file(obj, rotation, center=np.zeros(3))

    lines = obj.read_text(encoding="utf-8").splitlines()
    vertices = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines if line.startswith("v ")])
    uvs = [line for line in lines if line.startswith("vt ")]
    normal = np.asarray([float(v) for v in next(line for line in lines if line.startswith("vn ")).split()[1:4]])

    np.testing.assert_allclose(vertices, [[0, 0, 0], [1, 0, 0], [0, 0, -1]], atol=1e-7)
    assert len(uvs) == 3
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-7)


def test_invalid_pose_quaternion_is_rejected():
    try:
        quaternion_to_rotation(0, 0, 0, 0)
    except ValueError as exc:
        assert "zero-length" in str(exc)
    else:
        raise AssertionError("Expected zero quaternion to be rejected.")
