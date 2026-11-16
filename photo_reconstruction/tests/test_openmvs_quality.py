from photo_reconstruction.openmvs_quality import (
    densify_common_options,
    reconstruct_mesh_options,
    refine_mesh_options,
    texture_mesh_options,
)


def _value(options: list[str], key: str) -> str:
    return options[options.index(key) + 1]


def test_densify_profile_restores_cpu_defaults_and_safe_gap_interpolation():
    options = densify_common_options()
    assert _value(options, "--number-views") == "5"
    assert _value(options, "--iters") == "3"
    assert _value(options, "--geometric-iters") == "2"
    assert _value(options, "--number-views-fuse") == "2"
    assert _value(options, "--postprocess-dmaps") == "2"
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
