"""Version-pinned, memory-conscious OpenMVS 2.4.0 quality profiles."""

from __future__ import annotations


def densify_common_options(
    *,
    resolution_level: int = 0,
    max_resolution: int = 1600,
    max_threads: int = 1,
    number_views: int = 5,
    postprocess_dmaps: int = 10,
    iterations: int = 3,
    geometric_iterations: int = 2,
    min_resolution: int = 640,
    sub_resolution_levels: int = 2,
) -> list[str]:
    """Common CPU dense-stereo parameters verified against OpenMVS 2.4.0.

    The defaults retain five source views, three PatchMatch iterations and
    two geometric-consistency iterations for High/Maximum runs. OpenMVS 2.4.0
    defines postprocess-dmaps as a bit mask: 1=remove small speckles,
    2=interpolate short gaps, 4=confidence recalibration fused into CUDA
    estimation, and 8=force the confidence-recalibration pass on CPU.
    High/Maximum use 10 (2+8), including the extra full-resolution CPU pass.
    The Apple-Silicon High profile opts for 2 to skip that CPU pass after
    benchmarking trade-offs; Maximum keeps the original full-quality setting.
    Balanced Standard explicitly uses 2 to retain short-gap interpolation
    without that additional CPU confidence-recalibration pass.
    """
    if (
        resolution_level < 0 or max_resolution <= 0 or max_threads < 1
        or iterations < 1 or geometric_iterations < 0
        or min_resolution <= 0 or sub_resolution_levels < 0
    ):
        raise ValueError("Invalid dense reconstruction resource settings.")
    if number_views < 0 or not 0 <= postprocess_dmaps <= 15:
        raise ValueError("Invalid dense view or depth-map post-processing bit mask.")
    return [
        "--archive-type", "2",
        "--max-threads", str(max_threads),
        "--resolution-level", str(resolution_level),
        "--max-resolution", str(max_resolution),
        "--min-resolution", str(min_resolution),
        "--sub-resolution-levels", str(sub_resolution_levels),
        "--number-views", str(number_views),
        "--iters", str(iterations),
        "--geometric-iters", str(geometric_iterations),
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
    *,
    max_threads: int = 1,
    max_texture_size: int = 2048,
    virtual_face_images: int = 3,
    local_seam_leveling: bool = True,
    global_seam_leveling: bool = True,
) -> list[str]:
    """Create OpenMVS texture patches with configurable seam handling.

    Defaults preserve the existing Standard/High/Maximum behavior. The
    Apple-Silicon High workaround can disable both seam passes on affected
    OpenMVS 2.4.0 builds, where local seam leveling leaves black patch edges
    and global seam leveling can corrupt the full texture.
    """
    if (
        max_threads < 1
        or max_texture_size < 256
        or virtual_face_images < 1
    ):
        raise ValueError("Invalid texture export settings.")
    return [
        "--max-threads", str(max_threads),
        "--resolution-level", "0",
        "--min-resolution", "640",
        "--max-texture-size", str(max_texture_size),
        "--virtual-face-images", str(virtual_face_images),
        "--cost-smoothness-ratio", "0.7",
        "--close-holes", "0",
        "--sharpness-weight", "0.5",
        "--ignore-mask-label=-1",
        "--local-seam-leveling", "1" if local_seam_leveling else "0",
        "--global-seam-leveling", "1" if global_seam_leveling else "0",
    ]



def densify_fusion_options(*, minimum_views: int = 2) -> list[str]:
    """Options used only by the fusion pass (not by depth-map estimation)."""
    if minimum_views < 1:
        raise ValueError("Fusion requires at least one supporting view.")
    return [
        "--number-views-fuse", str(minimum_views),
        "--fusion-mode", "0",
    ]



def apple_silicon_high_profile() -> dict[str, int]:
    """Runtime-oriented High profile for Apple Silicon CPU-only OpenMVS runs.

    COLMAP keeps the existing 2400 px / 8192-feature SIFT settings. Dense
    stereo uses 1920 px, four source views, one PatchMatch iteration and
    one geometric-consistency iteration (the OpenMVS default), and skips the
    additional full-resolution CPU confidence-recalibration pass.
    Surface meshing, texturing, orientation, exports and QA remain enabled,
    while RefineMesh is skipped by the backend to reduce mesh and texture cost.
    This is a deliberate geometry-detail/runtime trade-off; use Maximum for
    the original full-quality dense solve and full mesh refinement.
    """
    return {
        "feature_size": 8192,
        "max_image_size": 2400,
        "max_resolution": 1920,
        "resolution_level": 0,
        "dense_views": 4,
        "dense_iterations": 1,
        "geometric_iterations": 1,
        "min_dense_resolution": 560,
        "sub_resolution_levels": 1,
        "postprocess_dmaps": 2,
    }


def resolve_openmvs_thread_count(
    *,
    platform_system: str,
    platform_machine: str,
    cpu_count: int,
    total_memory_bytes: int | None,
    override: str | None = None,
) -> int:
    """Choose a conservative OpenMVS thread limit for Apple Silicon memory.

    CLEAR360_OPENMVS_THREADS can override the automatic choice. On Apple
    Silicon, the default is 2 threads below 12 GiB, 3 below 24 GiB, and 4
    above that. Other platforms retain the legacy one-thread default unless
    explicitly overridden.
    """
    available_cpus = max(1, int(cpu_count or 1))
    override_text = (override or "").strip()
    if override_text:
        try:
            requested = int(override_text)
        except ValueError as exc:
            raise ValueError(
                "CLEAR360_OPENMVS_THREADS must be a positive integer."
            ) from exc
        if requested < 1:
            raise ValueError("CLEAR360_OPENMVS_THREADS must be a positive integer.")
        return min(requested, available_cpus)

    is_apple_silicon = (
        platform_system == "Darwin"
        and platform_machine.lower() in {"arm64", "aarch64"}
    )
    if not is_apple_silicon:
        return 1

    gib = 1024 ** 3
    memory = int(total_memory_bytes or 0)
    if memory <= 0 or memory < 12 * gib:
        return min(2, available_cpus)
    if memory < 24 * gib:
        return min(3, available_cpus)
    return min(4, available_cpus)
