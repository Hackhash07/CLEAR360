"""Version-pinned, memory-conscious OpenMVS 2.4.0 quality profiles."""

from __future__ import annotations


def densify_common_options(
    *,
    resolution_level: int = 0,
    max_resolution: int = 1600,
    max_threads: int = 1,
    number_views: int = 5,
    postprocess_dmaps: int = 10,
) -> list[str]:
    """Common CPU dense-stereo parameters verified against OpenMVS 2.4.0.

    The profile keeps five source views, three PatchMatch iterations and two
    geometric-consistency iterations. OpenMVS 2.4.0 defines postprocess-dmaps
    as a bit mask: 1=remove small speckles, 2=interpolate short gaps,
    4=confidence recalibration fused into CUDA estimation, and 8=force the
    confidence-recalibration pass on CPU. The quality profile uses 10 (2+8):
    interpolate short gaps and run CPU confidence recalibration so weak depth
    estimates can be rejected before fusion. This costs an extra full-resolution
    CPU pass, so callers may explicitly use 2 for a faster draft preview.
    """
    if resolution_level < 0 or max_resolution <= 0 or max_threads < 1:
        raise ValueError("Invalid dense reconstruction resource settings.")
    if number_views < 0 or not 0 <= postprocess_dmaps <= 15:
        raise ValueError("Invalid dense view or depth-map post-processing bit mask.")
    return [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--resolution-level", str(resolution_level),
        "--max-resolution", str(max_resolution),
        "--min-resolution", "640",
        "--sub-resolution-levels", "2",
        "--number-views", str(number_views),
        "--iters", "3",
        "--geometric-iters", "2",
        "--postprocess-dmaps", str(postprocess_dmaps),
        "--tower-mode", "0",
    ]


def reconstruct_mesh_options(
    *,
    max_threads: int = 1,
    remove_spurious: float = 20.0,
    free_space_support: bool = False,
) -> list[str]:
    """Mesh cleanup with optional recovery of weakly represented surfaces.

    OpenMVS 2.4.0 documents free-space support as using free-space evidence
    to reconstruct weakly represented surfaces. It is opt-in because it can
    trade some precision for completeness and needs visual comparison.
    """
    if max_threads < 1 or remove_spurious < 0:
        raise ValueError("Invalid mesh cleanup settings.")
    options = [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--remove-spurious", f"{remove_spurious:g}",
        "--close-holes", "30",
        "--smooth", "2",
    ]
    if free_space_support:
        # OpenMVS 2.4.0 declares this as a typed bool option, not a zero-arg flag.
        options.extend(["--free-space-support", "1"])
    return options


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



def densify_fusion_options(*, minimum_views: int = 2) -> list[str]:
    """Options used only by the fusion pass (not by depth-map estimation)."""
    if minimum_views < 1:
        raise ValueError("Fusion requires at least one supporting view.")
    return [
        "--number-views-fuse", str(minimum_views),
        "--fusion-mode", "0",
    ]
