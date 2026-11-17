import numpy as np
import pytest

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.texture_arrays import build_unwelded_uv_arrays


def _camera(name: str, translation: tuple[float, float, float]) -> PhoneCameraModel:
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, 3] = np.asarray(translation, dtype=np.float64)
    return PhoneCameraModel(
        name=name,
        model_name="PINHOLE",
        width=1000,
        height=800,
        params=np.array([500.0, 500.0, 500.0, 400.0]),
        fx=500.0,
        fy=500.0,
        cx=500.0,
        cy=400.0,
        T_world_cam=world_from_camera,
        T_cam_world=np.linalg.inv(world_from_camera),
    )


def _legacy_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    best_camera_indices: np.ndarray,
    camera_names: list[str],
    camera_models: dict[str, PhoneCameraModel],
    camera_cell_offsets: dict[int, tuple[int, int, float, float]],
    *,
    total_atlas_width: int,
    total_atlas_height: int,
    cell_width: int,
    cell_height: int,
    neutral_u: float,
    neutral_v: float,
):
    """Small reference implementation matching the former per-face loop."""
    verts = vertices.astype(np.float64)
    face_count = len(faces)
    new_vertices = np.zeros((face_count * 3, 3), dtype=np.float32)
    new_uvs = np.zeros((face_count * 3, 2), dtype=np.float32)
    new_faces = np.arange(face_count * 3, dtype=np.uint32).reshape(face_count, 3)

    for face_index in range(face_count):
        camera_index = int(best_camera_indices[face_index])
        points = verts[faces[face_index]]
        new_vertices[face_index * 3:face_index * 3 + 3] = points
        if camera_index < 0 or camera_index not in camera_cell_offsets:
            new_uvs[face_index * 3:face_index * 3 + 3] = np.array(
                [[neutral_u, neutral_v]] * 3, dtype=np.float32
            )
            continue

        x_offset, y_offset, scale_x, scale_y = camera_cell_offsets[camera_index]
        pixels, _, _ = camera_models[camera_names[camera_index]].project_world_to_image(
            points, check_bounds=False
        )
        u_atlas = (x_offset + pixels[:, 0] * scale_x) / float(total_atlas_width)
        v_atlas = (y_offset + pixels[:, 1] * scale_y) / float(total_atlas_height)
        u_min = x_offset / float(total_atlas_width)
        u_max = (x_offset + cell_width) / float(total_atlas_width)
        v_min = y_offset / float(total_atlas_height)
        v_max = (y_offset + cell_height) / float(total_atlas_height)
        u_atlas = np.clip(u_atlas, u_min + 1e-4, u_max - 1e-4)
        v_atlas = np.clip(v_atlas, v_min + 1e-4, v_max - 1e-4)
        new_uvs[face_index * 3:face_index * 3 + 3] = np.column_stack(
            [u_atlas, v_atlas]
        )

    return new_vertices, new_faces, new_uvs


def test_vectorized_uv_arrays_match_legacy_per_face_math():
    rng = np.random.default_rng(2026)
    vertices = rng.normal(size=(24, 3)).astype(np.float32)
    vertices[:, 2] += 8.0
    faces = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
         [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
        dtype=np.uint32,
    )
    best_camera_indices = np.array([0, 1, -1, 0, 1, -1, 1, 0], dtype=np.int32)
    camera_names = ["camera_a.jpg", "camera_b.jpg"]
    camera_models = {
        camera_names[0]: _camera(camera_names[0], (0.0, 0.0, 0.0)),
        camera_names[1]: _camera(camera_names[1], (0.35, -0.1, 0.0)),
    }
    camera_cell_offsets = {
        0: (0, 0, 0.5, 0.5),
        1: (512, 0, 0.5, 0.5),
    }
    kwargs = dict(
        total_atlas_width=1024,
        total_atlas_height=512,
        cell_width=512,
        cell_height=512,
        neutral_u=0.03125,
        neutral_v=0.0625,
    )

    actual = build_unwelded_uv_arrays(
        vertices, faces, best_camera_indices, camera_names,
        camera_models, camera_cell_offsets, **kwargs
    )
    expected = _legacy_reference(
        vertices, faces, best_camera_indices, camera_names,
        camera_models, camera_cell_offsets, **kwargs
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1e-7)


def test_vectorized_uv_arrays_keep_unobserved_faces_on_neutral_swatch():
    vertices = np.array(
        [[0, 0, 5], [1, 0, 5], [0, 1, 5]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    result = build_unwelded_uv_arrays(
        vertices,
        faces,
        np.array([-1], dtype=np.int32),
        [],
        {},
        {},
        total_atlas_width=512,
        total_atlas_height=512,
        cell_width=512,
        cell_height=512,
        neutral_u=0.1,
        neutral_v=0.2,
    )
    np.testing.assert_array_equal(result[0], vertices)
    np.testing.assert_array_equal(result[1], np.array([[0, 1, 2]], dtype=np.uint32))
    np.testing.assert_array_equal(
        result[2],
        np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype=np.float32),
    )


def test_vectorized_uv_arrays_validate_face_indices():
    with pytest.raises(ValueError, match="outside the vertices"):
        build_unwelded_uv_arrays(
            np.zeros((3, 3), dtype=np.float32),
            np.array([[0, 1, 3]], dtype=np.uint32),
            np.array([-1], dtype=np.int32),
            [],
            {},
            {},
            total_atlas_width=512,
            total_atlas_height=512,
            cell_width=512,
            cell_height=512,
            neutral_u=0.1,
            neutral_v=0.2,
        )


def test_vectorized_uv_arrays_validate_camera_index_shape():
    with pytest.raises(ValueError, match="one entry per face"):
        build_unwelded_uv_arrays(
            np.zeros((3, 3), dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
            np.array([], dtype=np.int32),
            [],
            {},
            {},
            total_atlas_width=512,
            total_atlas_height=512,
            cell_width=512,
            cell_height=512,
            neutral_u=0.1,
            neutral_v=0.2,
        )
