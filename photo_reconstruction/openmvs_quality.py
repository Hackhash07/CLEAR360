"""Version-pinned, memory-conscious OpenMVS 2.4.0 quality profiles."""

from __future__ import annotations


def densify_common_options(
    *,
    resolution_level: int = 0,
    max_resolution: int = 1600,
    max_threads: int = 1,
    number_views: int = 5,
    postprocess_dmaps: int = 2,
) -> list[str]:
    """Common CPU dense-stereo parameters verified against OpenMVS 2.4.0.

    On the user's CPU-only 2.4.0 build, defaults are five source views and
    three PatchMatch iterations; geometric consistency defaults to two.
    Keep at least two agreeing views during fusion to avoid noisy one-view
    geometry. postprocess-dmaps=2 only interpolates short gaps whose boundary
    depths are sufficiently similar; it does not fill large unknown regions.
    """
    if resolution_level < 0 or max_resolution <= 0 or max_threads < 1:
        raise ValueError("Invalid dense reconstruction resource settings.")
    if number_views < 0 or postprocess_dmaps not in (0, 1, 2, 3, 4, 5, 6, 7):
        raise ValueError("Invalid dense view or depth-map post-processing setting.")
    return [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--resolution-level", str(resolution_level),
        "--max-resolution", str(max_resolution),
        "--min-resolution", "640",
        "--sub-resolution-levels", "2",
        "--number-views", str(number_views),
        "--number-views-fuse", "2",
        "--iters", "3",
        "--geometric-iters", "2",
        "--postprocess-dmaps", str(postprocess_dmaps),
        "--tower-mode", "0",
    ]


def reconstruct_mesh_options(
    *, max_threads: int = 1, remove_spurious: float = 20.0
) -> list[str]:
    """Mesh cleanup; lower remove-spurious factors remove more long-edge faces."""
    if max_threads < 1 or remove_spurious < 0:
        raise ValueError("Invalid mesh cleanup settings.")
    return [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--remove-spurious", f"{remove_spurious:g}",
        "--close-holes", "30",
        "--smooth", "2",
    ]


def refine_mesh_options(
    *, max_threads: int = 1, max_views: int = 6
) -> list[str]:
    """Refine at native undistorted resolution using all six available images."""
    if max_threads < 1 or max_views < 1:
        raise ValueError("Invalid refinement settings.")
    return [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--resolution-level", "0",
        "--min-resolution", "640",
        "--max-views", str(max_views),
        "--decimate", "1",
        "--close-holes", "30",
        "--scales", "2",
        "--reduce-memory", "1",
    ]


def texture_mesh_options(
    *, max_threads: int = 1, max_texture_size: int = 2048
) -> list[str]:
    """Create texture patches with OpenMVS seam-leveling enabled."""
    if max_threads < 1 or max_texture_size < 256:
        raise ValueError("Invalid texture export settings.")
    return [
        "--max-threads", str(max_threads),
        "--resolution-level", "0",
        "--min-resolution", "640",
        "--max-texture-size", str(max_texture_size),
        "--virtual-face-images", "3",
        "--cost-smoothness-ratio", "0.7",
        "--close-holes", "0",
        "--sharpness-weight", "0.5",
        "--ignore-mask-label=-1",
        "--local-seam-leveling", "1",
        "--global-seam-leveling", "1",
    ]
