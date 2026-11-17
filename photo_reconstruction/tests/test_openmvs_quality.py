from photo_reconstruction.openmvs_quality import (
    densify_common_options,
    densify_fusion_options,
    reconstruct_mesh_options,
    refine_mesh_options,
    texture_mesh_options,
)


def _value(options: list[str], key: str) -> str:
    return options[options.index(key) + 1]


def test_densify_quality_profile_enables_cpu_depth_confidence_recalibration():
    options = densify_common_options()
    assert _value(options, "--number-views") == "5"
    assert _value(options, "--iters") == "3"
    assert _value(options, "--geometric-iters") == "2"
    assert "--number-views-fuse" not in options
    assert _value(options, "--postprocess-dmaps") == "10"
    assert _value(options, "--resolution-level") == "0"
    assert _value(options, "--max-resolution") == "1600"


def test_mesh_cleanup_uses_default_spurious_factor_not_aggressive_factor():
    options = reconstruct_mesh_options()
    assert _value(options, "--remove-spurious") == "20"
    assert _value(options, "--close-holes") == "30"
    assert _value(options, "--smooth") == "2"


def test_refine_uses_native_resolution_and_six_views():
    options = refine_mesh_options()
    assert _value(options, "--resolution-level") == "0"
    assert _value(options, "--min-resolution") == "640"
    assert _value(options, "--max-views") == "6"


def test_texture_exports_use_seam_leveling():
    options = texture_mesh_options()
    assert _value(options, "--global-seam-leveling") == "1"
    assert _value(options, "--local-seam-leveling") == "1"



def test_reconstruct_mesh_free_space_support_is_explicit_opt_in():
    default_options = reconstruct_mesh_options()
    experimental_options = reconstruct_mesh_options(free_space_support=True)
    assert "--free-space-support" not in default_options
    assert experimental_options[experimental_options.index("--free-space-support") + 1] == "1"



def test_fusion_options_define_minimum_views_exactly_once():
    options = densify_fusion_options(minimum_views=2)
    assert options.count("--number-views-fuse") == 1
    assert _value(options, "--number-views-fuse") == "2"
    assert options.count("--fusion-mode") == 1
    assert _value(options, "--fusion-mode") == "0"



def test_draft_densify_profile_can_keep_cpu_cost_lower():
    options = densify_common_options(postprocess_dmaps=2)
    assert _value(options, "--postprocess-dmaps") == "2"


def test_densify_postprocess_bitmask_validation_rejects_out_of_range_values():
    import pytest

    with pytest.raises(ValueError, match="bit mask"):
        densify_common_options(postprocess_dmaps=16)


def test_fast_preview_densify_profile_reduces_pixel_and_iteration_budget():
    options = densify_common_options(
        resolution_level=1,
        max_resolution=640,
        max_threads=2,
        number_views=3,
        postprocess_dmaps=2,
        iterations=1,
        geometric_iterations=1,
        min_resolution=320,
        sub_resolution_levels=1,
    )
    assert _value(options, "--resolution-level") == "1"
    assert _value(options, "--max-resolution") == "640"
    assert _value(options, "--max-threads") == "2"
    assert _value(options, "--number-views") == "3"
    assert _value(options, "--iters") == "1"
    assert _value(options, "--geometric-iters") == "1"
    assert _value(options, "--postprocess-dmaps") == "2"
    assert _value(options, "--min-resolution") == "320"
    assert _value(options, "--sub-resolution-levels") == "1"
