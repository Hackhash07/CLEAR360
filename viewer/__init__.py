"""
viewer/ — Interactive 3D Surround Environment Viewer for CLEAR360.

Public objects are imported lazily so lightweight utilities such as the OBJ
texture loader do not require Open3D, Qt, or the OpenGL viewer stack merely
because the viewer package is imported.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "SceneData": ("viewer.data_loader", "SceneData"),
    "ViewerDataLoader": ("viewer.data_loader", "ViewerDataLoader"),
    "VehicleModel3D": ("viewer.vehicle_model", "VehicleModel3D"),
    "Clear360Viewport": ("viewer.viewport", "Clear360Viewport"),
    "ViewerControlPanel": ("viewer.controls", "ViewerControlPanel"),
    "ViewerInfoPanel": ("viewer.info_panel", "ViewerInfoPanel"),
    "Clear360ViewerWindow": ("viewer.app", "Clear360ViewerWindow"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public viewer symbols on first access."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
