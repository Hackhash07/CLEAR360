"""Small, dependency-light diagnostics for photogrammetry texture atlases."""

from __future__ import annotations

import math

import numpy as np


def texture_color_statistics(
    image: np.ndarray, *, max_sample_pixels: int = 1_000_000
) -> dict[str, object]:
    """Summarize RGB brightness without rewriting or classifying dark pixels.

    "Near black" is defined as an RGB channel sum below 12. This is only a
    diagnostic signal: genuine dark paint, tyres, shadows and trim are valid
    photographic content and must never be recolored automatically.
    """
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Texture diagnostics require an image with at least 3 color channels.")
    if pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise ValueError("Texture diagnostics require a non-empty image.")
    if max_sample_pixels < 1:
        raise ValueError("max_sample_pixels must be positive.")

    step = max(
        1,
        int(math.ceil(math.sqrt((pixels.shape[0] * pixels.shape[1]) / max_sample_pixels))),
    )
    sample = np.asarray(pixels[::step, ::step, :3], dtype=np.uint8)
    sample_u16 = sample.astype(np.uint16, copy=False)
    near_black = np.sum(sample_u16, axis=2) < 12
    mean_rgb = np.mean(sample.astype(np.float32), axis=(0, 1))
    return {
        "mean_rgb": [round(float(value), 2) for value in mean_rgb],
        "near_black_texel_percentage": round(float(np.mean(near_black) * 100.0), 2),
        "sampled_pixels": int(sample.shape[0] * sample.shape[1]),
        "sample_step": step,
        "near_black_definition": "sum(R,G,B) < 12",
    }
