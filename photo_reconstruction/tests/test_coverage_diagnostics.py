from pathlib import Path

import numpy as np

from photo_reconstruction.coverage_diagnostics import (
    ColmapCamera,
    ColmapImage,
    parse_images_txt,
    parse_points3d_tracks,
    project_points,
)


def test_parse_images_skips_numeric_observation_rows(tmp_path: Path):
    path = tmp_path / "images.txt"
    path.write_text(
        "1 1 0 0 0 0 0 0 1 fountain_0003.jpg\n"
        "1 2 3 4 5 6 7 8 9 10 11 12\n"
        "2 1 0 0 0 0 0 0 1 fountain_0004.jpg\n"
        "1 2 3 4 5 6 7 8 9 10 11 12\n",
        encoding="utf-8",
    )
    images = parse_images_txt(path)
    assert [image.name for image in images] == ["fountain_0003.jpg", "fountain_0004.jpg"]


def test_sparse_tracks_report_pairwise_shared_observations(tmp_path: Path):
    path = tmp_path / "points3D.txt"
    path.write_text(
        "1 0 0 1 255 0 0 0.2 1 0 2 1 3 2\n"
        "2 0 1 1 0 255 0 0.3 1 1 2 2\n",
        encoding="utf-8",
    )
    count, per_image, pairs = parse_points3d_tracks(path)
    assert count == 2
    assert per_image == {1: 2, 2: 2, 3: 1}
    assert pairs[(1, 2)] == 2
    assert pairs[(1, 3)] == 1


def test_project_simple_radial_camera_center_point():
    camera = ColmapCamera(
        camera_id=1, model="SIMPLE_RADIAL", width=100, height=80,
        params=(50.0, 50.0, 40.0, -0.1),
    )
    image = ColmapImage(
        image_id=1, qw=1, qx=0, qy=0, qz=0,
        tx=0, ty=0, tz=0, camera_id=1, name="frame.jpg",
    )
    points = np.array([[0.0, 0.0, 2.0], [0.4, 0.0, 2.0], [0.0, 0.0, -1.0]])
    u, v, valid = project_points(points, camera, image)
    np.testing.assert_allclose([u[0], v[0]], [50.0, 40.0], atol=1e-7)
    assert valid.tolist() == [True, True, False]
