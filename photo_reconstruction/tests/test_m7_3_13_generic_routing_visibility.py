"""Regression tests for generic scene routing and photogrammetric visibility."""

import json

import numpy as np

from photo_reconstruction.pipeline import resolve_scene_profile
from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine


def _front_facing_triangle():
    vertices = np.array(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [0.0, 0.2, 2.0],
        ],
        dtype=np.float32,
    )
    # Winding gives a -Z normal, pointing back toward a camera at the origin.
    faces = np.array([[0, 2, 1]], dtype=np.uint32)
    camera = PhoneCameraModel(
        name="view.jpg",
        model_name="PINHOLE",
        width=200,
        height=200,
        params=np.array([200.0, 200.0, 100.0, 100.0]),
        fx=200.0,
        fy=200.0,
        cx=100.0,
        cy=100.0,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )
    image = np.full((200, 200, 3), 180, dtype=np.uint8)
    return vertices, faces, camera, image


def test_generic_scene_profile_is_default(tmp_path):
    assert resolve_scene_profile(tmp_path) == "generic"


def test_fountain_profile_requires_explicit_opt_in(tmp_path):
    (tmp_path / "scene_profile.json").write_text(
        json.dumps({"profile": "architectural_fountain"}), encoding="utf-8"
    )
    assert resolve_scene_profile(tmp_path) == "architectural_fountain"


def test_unknown_scene_profile_falls_back_to_generic(tmp_path):
    (tmp_path / "scene_profile.json").write_text(
        json.dumps({"profile": "bedroom_special_case"}), encoding="utf-8"
    )
    assert resolve_scene_profile(tmp_path) == "generic"


def test_texture_visibility_rejects_triangle_crossing_image_boundary():
    engine = VisibilityAwareTextureEngine(atlas_size=512)
    vertices, _, camera, image = _front_facing_triangle()
    vertices[0, 0] = -1.2  # Projects outside the image; centroid remains inside.
    faces = np.array([[0, 2, 1]], dtype=np.uint32)

    _, _, _, _, report = engine.texture_mesh(
        vertices=vertices,
        faces=faces,
        camera_models={"view.jpg": camera},
        images={"view.jpg": image},
    )

    assert report.textured_faces == 0
    assert report.untextured_faces == 1


def test_texture_visibility_does_not_rescale_inconsistent_depth():
    engine = VisibilityAwareTextureEngine(atlas_size=512, occlusion_threshold=0.20)
    vertices, faces, camera, image = _front_facing_triangle()

    # Mesh lies at Z=2m, while the already-registered observed depth is 1m.
    # The face is occluded and must not become visible through per-face scale fitting.
    depth = np.full((200, 200), 1.0, dtype=np.float32)
    _, _, _, _, report = engine.texture_mesh(
        vertices=vertices,
        faces=faces,
        camera_models={"view.jpg": camera},
        images={"view.jpg": image},
        depth_maps={"view.jpg": depth},
    )

    assert report.textured_faces == 0
    assert report.untextured_faces == 1


def test_texture_visibility_accepts_consistent_axial_depth():
    engine = VisibilityAwareTextureEngine(atlas_size=512, occlusion_threshold=0.20)
    vertices, faces, camera, image = _front_facing_triangle()
    depth = np.full((200, 200), 2.0, dtype=np.float32)

    _, _, _, _, report = engine.texture_mesh(
        vertices=vertices,
        faces=faces,
        camera_models={"view.jpg": camera},
        images={"view.jpg": image},
        depth_maps={"view.jpg": depth},
    )

    assert report.textured_faces == 1
    assert report.untextured_faces == 0
