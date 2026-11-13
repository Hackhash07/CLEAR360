"""
viewer/ — Interactive 3D Surround Environment Viewer for CLEAR360
"""

from viewer.data_loader import SceneData, ViewerDataLoader
from viewer.vehicle_model import VehicleModel3D
from viewer.viewport import Clear360Viewport
from viewer.controls import ViewerControlPanel
from viewer.info_panel import ViewerInfoPanel

__all__ = [
    "SceneData",
    "ViewerDataLoader",
    "VehicleModel3D",
    "Clear360Viewport",
    "ViewerControlPanel",
    "ViewerInfoPanel",
    "Clear360ViewerWindow",
]


def __getattr__(name: str):
    if name == "Clear360ViewerWindow":
        from viewer.app import Clear360ViewerWindow

        return Clear360ViewerWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
