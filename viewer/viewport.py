"""
viewer/viewport.py — Interactive 3D Surround Viewport for CLEAR360

Embeds a high-performance OpenGL VisPy SceneCanvas within PySide6.
Features:
  - Perspective-based 3D surround camera with free turntable rotation, pan, zoom
  - Preset automotive viewpoints: Chase 3D (default), Front, Rear, Left, Right, Top Perspective
  - High-performance multi-layer rendering (>60 FPS on Apple Silicon):
      * RGB / Semantic Point Clouds
      * Volumetric Occupancy Grid (Occupied voxels in red, free space in cyan)
      * 3D Hyundai Creta SUV model with headlights, taillights, side mirrors, wheels
      * 3D Vehicle Clearance Envelope wireframe (body + mirrors extending to +/- 1.05 m)
      * 3D Obstacle Bounding Boxes (lime green wireframes)
      * Selected Obstacle Highlight Box (gold/electric yellow wireframe)
      * Dynamic 3D Clearance Measurement Line connecting vehicle boundary to closest obstacle point
      * Ground reference grid and coordinate axes (+X Right Red, +Y Forward Green, +Z Up Blue)
  - Screen-space mouse picking to select obstacles directly in the 3D viewport
"""

from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import cv2
from PySide6 import QtWidgets, QtCore, QtGui
import vispy.scene
from vispy.scene import visuals

from viewer.data_loader import SceneData
from viewer.vehicle_model import VehicleModel3D


class Clear360Viewport(QtWidgets.QWidget):
    """
    Modular 3D OpenGL Viewport widget embedding a VisPy SceneCanvas into PySide6.
    """

    obstacle_selected = QtCore.Signal(int)  # Emitted when an obstacle is selected (by ID)

    CAMERA_PRESETS = {
        "chase": {"elevation": 18.0, "azimuth": -90.0, "distance": 7.5, "center": (0.0, 1.2, 0.7)},
        "front": {"elevation": 14.0, "azimuth": 90.0, "distance": 7.5, "center": (0.0, 1.8, 0.7)},
        "rear": {"elevation": 14.0, "azimuth": -90.0, "distance": 7.5, "center": (0.0, -1.8, 0.7)},
        "left": {"elevation": 14.0, "azimuth": 180.0, "distance": 8.0, "center": (0.0, 0.0, 0.7)},
        "right": {"elevation": 14.0, "azimuth": 0.0, "distance": 8.0, "center": (0.0, 0.0, 0.7)},
        "top": {"elevation": 89.5, "azimuth": -90.0, "distance": 14.0, "center": (0.0, 0.0, 0.0)},
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.vehicle_model = VehicleModel3D()
        self.current_scene: Optional[SceneData] = None
        self.selected_obstacle_id: Optional[int] = None

        # Primary Visualization Mode: "surface" (default), "driver", "pointcloud", "semantic_occupancy"
        self.vis_mode = "surface"
        self.texture_filter = None

        # View settings state
        self.point_cloud_mode = "rgb"  # "rgb", "semantic", "confidence", "none"
        self.point_size = 2.5
        self.show_occupied_voxels = False
        self.show_free_voxels = False
        self.show_unknown_voxels = False
        self.show_vehicle = True
        self.show_envelope = True
        self.show_bboxes = True
        self.show_clearance_line = True
        self.show_grid = True

        self._init_ui()
        self._init_visuals()
        self.set_camera_preset("chase")

    def _init_ui(self):
        """Initialize VisPy SceneCanvas and embed into widget layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create VisPy canvas with interactive keys & mouse
        self.canvas = vispy.scene.SceneCanvas(
            keys=None,
            show=False,
            bgcolor="#121318",  # Sleek automotive dark background
            parent=self,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.up = "+z"
        self.view.camera.fov = 50.0

        # Embed native Qt widget
        layout.addWidget(self.canvas.native)

        # Connect mouse click for 3D obstacle picking
        self.canvas.events.mouse_press.connect(self._on_mouse_press)

    def _init_visuals(self):
        """Construct visual scene nodes in the VisPy scene graph."""
        # 1. Ground reference grid
        grid_pts, grid_cols = self.vehicle_model.get_ground_grid_lines(size=15.0, step=1.0)
        self.visual_grid = visuals.Line(pos=grid_pts, color=grid_cols, connect="segments", width=1.0)
        self.view.add(self.visual_grid)

        # 2. Coordinate axes at vehicle origin
        axes_pts, axes_cols = self.vehicle_model.get_axes_lines(length=1.5)
        self.visual_axes = visuals.Line(pos=axes_pts, color=axes_cols, connect="segments", width=3.0)
        self.view.add(self.visual_axes)

        # 3. Vehicle 3D Mesh (Creta SUV)
        v_mesh, f_mesh, c_mesh = self.vehicle_model.get_mesh_data()
        self.visual_vehicle = visuals.Mesh(vertices=v_mesh, faces=f_mesh, vertex_colors=c_mesh)
        self.view.add(self.visual_vehicle)

        # 4. Vehicle Clearance Envelope Wireframe
        env_pts, env_cols = self.vehicle_model.get_clearance_envelope_lines()
        self.visual_envelope = visuals.Line(pos=env_pts, color=env_cols, connect="segments", width=1.5)
        self.view.add(self.visual_envelope)

        # 4b. Photorealistic 3D Surface Mesh (M7.1)
        dummy_v = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32)
        dummy_f = np.array([[0, 1, 2]], dtype=np.uint32)
        dummy_c = np.ones((3, 4), dtype=np.float32)
        self.visual_surface_mesh = visuals.Mesh(
            vertices=dummy_v,
            faces=dummy_f,
            vertex_colors=dummy_c,
            shading="smooth",
        )
        self.visual_surface_mesh.visible = False
        self.view.add(self.visual_surface_mesh)

        # 4c. Photorealistic Textured 3D Mesh with UV Atlas (M7.2 Driver View)
        self.visual_textured_mesh = visuals.Mesh(
            vertices=dummy_v,
            faces=dummy_f,
            color="white",
            shading="smooth",
        )
        self.visual_textured_mesh.visible = False
        self.view.add(self.visual_textured_mesh)

        # 5. Point Cloud (RGB / Semantic)
        self.visual_cloud = visuals.Markers()
        self.visual_cloud.set_data(np.zeros((0, 3), dtype=np.float32), size=self.point_size)
        self.view.add(self.visual_cloud)

        # 6. Occupancy Voxels (Occupied)
        self.visual_occ_occupied = visuals.Markers()
        self.visual_occ_occupied.set_data(np.zeros((0, 3), dtype=np.float32), size=4.0)
        self.view.add(self.visual_occ_occupied)

        # 7. Occupancy Voxels (Free space)
        self.visual_occ_free = visuals.Markers()
        self.visual_occ_free.set_data(np.zeros((0, 3), dtype=np.float32), size=2.0)
        self.view.add(self.visual_occ_free)

        # 7b. Occupancy Voxels (Unknown space)
        self.visual_occ_unknown = visuals.Markers()
        self.visual_occ_unknown.set_data(np.zeros((0, 3), dtype=np.float32), size=2.0)
        self.view.add(self.visual_occ_unknown)

        # 8. All Obstacle Bounding Boxes (lime green wireframes)
        self.visual_bboxes = visuals.Line(connect="segments", width=1.5)
        self.view.add(self.visual_bboxes)

        # 9. Selected Obstacle Bounding Box (highlighted gold wireframe)
        self.visual_selected_bbox = visuals.Line(connect="segments", width=3.5)
        self.view.add(self.visual_selected_bbox)

        # 10. Dynamic Clearance Measurement Line & Markers
        self.visual_clearance_line = visuals.Line(connect="segments", width=3.0)
        self.view.add(self.visual_clearance_line)
        self.visual_clearance_markers = visuals.Markers()
        self.view.add(self.visual_clearance_markers)

    # -------------------------------------------------------------------------
    # Scene Data Updating & Visualization Modes (M7.1)
    # -------------------------------------------------------------------------
    def request_update(self):
        """Request immediate redraw on VisPy scene canvas and Qt native widget."""
        self.canvas.update()
        if hasattr(self.canvas, "native") and self.canvas.native is not None:
            self.canvas.native.update()

    def set_scene_data(self, scene_data: SceneData):
        """Update viewport with new frame / source data."""
        self.current_scene = scene_data
        self._update_textured_mesh()
        self._update_surface_mesh()
        self._update_point_cloud()
        self._update_occupancy()
        self._update_obstacle_bboxes()
        self._apply_mode_visibilities()

        # If previous selection exists, re-highlight or select closest obstacle
        if self.selected_obstacle_id is not None:
            self.select_obstacle(self.selected_obstacle_id)
        elif scene_data.obstacles:
            # Auto-select closest obstacle by default
            self.select_obstacle(scene_data.obstacles[0]["obstacle_id"])

        self.request_update()

    def set_visualization_mode(self, mode: str):
        """
        Switch primary visualization mode:
          'driver': Clean Photorealistic Driver View (M7.2)
          'surface': Photorealistic 3D Surface Mesh (M7.1)
          'pointcloud': Point Cloud Debug Mode
          'semantic_occupancy': Semantic / Occupancy Grid Mode (Engineering View)
        """
        self.vis_mode = mode
        self._apply_mode_visibilities()
        self.request_update()

    def _apply_mode_visibilities(self):
        """Enforce visibility rules for active visualization mode."""
        if self.vis_mode == "driver":
            has_tex = (
                self.current_scene is not None
                and self.current_scene.textured_mesh_vertices is not None
                and len(self.current_scene.textured_mesh_vertices) > 0
            )
            has_surf = (
                self.current_scene is not None
                and self.current_scene.mesh_vertices is not None
                and len(self.current_scene.mesh_vertices) > 0
            )
            self.visual_textured_mesh.visible = has_tex
            self.visual_surface_mesh.visible = (not has_tex) and has_surf
            self.visual_vehicle.visible = self.show_vehicle

            # Driver View: Clean automotive cockpit view
            # If meshes exist, hide point cloud; if no mesh exists, show point cloud as fallback
            if not has_tex and not has_surf:
                self.visual_cloud.visible = (self.point_cloud_mode != "none")
                self.visual_grid.visible = self.show_grid
            else:
                self.visual_cloud.visible = False
                self.visual_grid.visible = False

            self.visual_occ_occupied.visible = False
            self.visual_occ_free.visible = False
            self.visual_occ_unknown.visible = False
            self.visual_bboxes.visible = False
            self.visual_selected_bbox.visible = False
            self.visual_clearance_line.visible = False
            self.visual_clearance_markers.visible = False
            self.visual_envelope.visible = False
            self.visual_axes.visible = False
            self.canvas.bgcolor = "#0a0a0e"

        elif self.vis_mode == "surface":
            self.visual_textured_mesh.visible = False
            has_mesh = (
                self.current_scene is not None
                and self.current_scene.mesh_vertices is not None
                and len(self.current_scene.mesh_vertices) > 0
            )
            self.visual_surface_mesh.visible = has_mesh
            self.visual_vehicle.visible = self.show_vehicle
            self.visual_cloud.visible = not has_mesh
            self.visual_occ_occupied.visible = False
            self.visual_occ_free.visible = False
            self.visual_occ_unknown.visible = False
            self.visual_bboxes.visible = False
            self.visual_selected_bbox.visible = False
            self.visual_clearance_line.visible = False
            self.visual_clearance_markers.visible = False
            self.visual_envelope.visible = self.show_envelope
            self.visual_grid.visible = self.show_grid
            self.visual_axes.visible = self.show_grid
            self.canvas.bgcolor = "#121318"

        elif self.vis_mode == "pointcloud":
            self.visual_textured_mesh.visible = False
            self.visual_surface_mesh.visible = False
            self.visual_cloud.visible = (self.point_cloud_mode != "none")
            self.visual_occ_occupied.visible = False
            self.visual_occ_free.visible = False
            self.visual_occ_unknown.visible = False
            self.visual_bboxes.visible = self.show_bboxes
            self.visual_clearance_line.visible = self.show_clearance_line
            self.visual_clearance_markers.visible = self.show_clearance_line
            self.visual_envelope.visible = self.show_envelope
            self.visual_grid.visible = self.show_grid
            self.visual_axes.visible = self.show_grid
            self.canvas.bgcolor = "#121318"

        elif self.vis_mode in ("semantic_occupancy", "engineering"):
            self.visual_textured_mesh.visible = False
            self.visual_surface_mesh.visible = False
            self.visual_cloud.visible = False
            self.visual_occ_occupied.visible = True
            self.visual_occ_free.visible = True
            self.visual_occ_unknown.visible = self.show_unknown_voxels
            self.visual_bboxes.visible = self.show_bboxes
            self.visual_clearance_line.visible = self.show_clearance_line
            self.visual_clearance_markers.visible = self.show_clearance_line
            self.visual_envelope.visible = self.show_envelope
            self.visual_grid.visible = self.show_grid
            self.visual_axes.visible = self.show_grid
            self.canvas.bgcolor = "#121318"

    def _update_textured_mesh(self):
        """Update 3D textured mesh vertices, faces, UVs, and texture atlas (M7.2)."""
        if (
            self.current_scene is None
            or self.current_scene.textured_mesh_vertices is None
            or len(self.current_scene.textured_mesh_vertices) < 3
            or self.current_scene.textured_mesh_faces is None
            or len(self.current_scene.textured_mesh_faces) < 1
            or self.current_scene.textured_mesh_uvs is None
            or self.current_scene.textured_mesh_texture is None
        ):
            self.visual_textured_mesh.visible = False
            return

        verts = self.current_scene.textured_mesh_vertices
        faces = self.current_scene.textured_mesh_faces
        uvs = self.current_scene.textured_mesh_uvs
        tex = self.current_scene.textured_mesh_texture

        self.visual_textured_mesh.set_data(vertices=verts, faces=faces, color="white")

        try:
            from vispy.visuals.filters import TextureFilter
            if self.texture_filter is None:
                self.texture_filter = TextureFilter(texture=tex, texcoords=uvs)
                self.visual_textured_mesh.attach(self.texture_filter)
            else:
                self.texture_filter.texture = tex
                self.texture_filter.texcoords = uvs
        except Exception as e:
            print(f"Warning: TextureFilter update error: {e}")

    def _update_surface_mesh(self):
        """Update 3D surface mesh vertices, faces, and vertex colors."""
        if (
            self.current_scene is None
            or self.current_scene.mesh_vertices is None
            or len(self.current_scene.mesh_vertices) < 3
            or self.current_scene.mesh_faces is None
            or len(self.current_scene.mesh_faces) < 1
        ):
            self.visual_surface_mesh.visible = False
            return

        verts = self.current_scene.mesh_vertices
        faces = self.current_scene.mesh_faces
        cols = self.current_scene.mesh_colors

        if cols is not None and cols.shape[1] == 3:
            alpha = np.ones((len(cols), 1), dtype=np.float32)
            cols = np.hstack([cols, alpha])

        self.visual_surface_mesh.set_data(vertices=verts, faces=faces, vertex_colors=cols)
        self.visual_surface_mesh.visible = (self.vis_mode == "surface")

    def _update_point_cloud(self):
        """Update point cloud visual based on current mode."""
        if self.current_scene is None or self.point_cloud_mode == "none":
            self.visual_cloud.visible = False
            return

        self.visual_cloud.visible = True
        if self.point_cloud_mode == "rgb":
            pts = self.current_scene.points_rgb
            cols = self.current_scene.colors_rgb
        elif self.point_cloud_mode == "confidence":
            pts = self.current_scene.points_rgb
            cols = self.current_scene.colors_confidence
        else:  # "semantic"
            pts = self.current_scene.points_semantic
            cols = self.current_scene.colors_semantic

        if len(pts) > 0:
            # Ensure RGBA
            if cols.shape[1] == 3:
                alpha = np.ones((len(cols), 1), dtype=np.float32)
                cols = np.hstack([cols, alpha])
            self.visual_cloud.set_data(pts, edge_color=None, face_color=cols, size=self.point_size)
        else:
            self.visual_cloud.set_data(np.zeros((0, 3), dtype=np.float32), size=self.point_size)

    def _update_occupancy(self):
        """Update occupancy grid visuals."""
        if self.current_scene is None:
            self.visual_occ_occupied.visible = False
            self.visual_occ_free.visible = False
            self.visual_occ_unknown.visible = False
            return

        self.visual_occ_occupied.visible = self.show_occupied_voxels
        self.visual_occ_free.visible = self.show_free_voxels
        self.visual_occ_unknown.visible = self.show_unknown_voxels

        if self.show_occupied_voxels and len(self.current_scene.occupancy_occupied_points) > 0:
            occ_pts = self.current_scene.occupancy_occupied_points
            occ_cols = np.tile(np.array([1.0, 0.25, 0.20, 0.85], dtype=np.float32), (len(occ_pts), 1))
            self.visual_occ_occupied.set_data(occ_pts, edge_color=None, face_color=occ_cols, size=4.0)

        if self.show_free_voxels and len(self.current_scene.occupancy_free_points) > 0:
            free_pts = self.current_scene.occupancy_free_points
            free_cols = np.tile(np.array([0.15, 0.80, 1.0, 0.15], dtype=np.float32), (len(free_pts), 1))
            self.visual_occ_free.set_data(free_pts, edge_color=None, face_color=free_cols, size=2.5)

        if self.show_unknown_voxels and len(self.current_scene.occupancy_unknown_points) > 0:
            unk_pts = self.current_scene.occupancy_unknown_points
            unk_cols = np.tile(np.array([0.45, 0.48, 0.55, 0.15], dtype=np.float32), (len(unk_pts), 1))
            self.visual_occ_unknown.set_data(unk_pts, edge_color=None, face_color=unk_cols, size=2.0)

    def _update_obstacle_bboxes(self):
        """Construct 3D wireframe bounding boxes for all detected obstacles."""
        if self.current_scene is None or not self.show_bboxes or not self.current_scene.obstacles:
            self.visual_bboxes.visible = False
            return

        line_pts: List[np.ndarray] = []
        for obs in self.current_scene.obstacles:
            if "bbox" in obs:
                b_min = np.array(obs["bbox"]["min_point"], dtype=np.float32)
                b_max = np.array(obs["bbox"]["max_point"], dtype=np.float32)
            else:
                c = np.array(obs.get("centroid", [0, 0, 0]), dtype=np.float32)
                d = np.array(obs.get("dimensions", [1, 1, 1]), dtype=np.float32)
                b_min = c - d / 2.0
                b_max = c + d / 2.0
            corners = self._get_box_corners(b_min, b_max)
            self._append_box_edges(corners, line_pts)

        if line_pts:
            pos = np.vstack(line_pts)
            # Lime green with transparency
            cols = np.tile(np.array([0.25, 1.0, 0.40, 0.85], dtype=np.float32), (len(pos), 1))
            self.visual_bboxes.set_data(pos=pos, color=cols)
            self.visual_bboxes.visible = True
        else:
            self.visual_bboxes.visible = False

    @staticmethod
    def _get_box_corners(b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
        x0, y0, z0 = b_min
        x1, y1, z1 = b_max
        return np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float32)

    @staticmethod
    def _append_box_edges(corners: np.ndarray, line_pts: List[np.ndarray]):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i1, i2 in edges:
            line_pts.append(corners[i1])
            line_pts.append(corners[i2])

    # -------------------------------------------------------------------------
    # Obstacle Selection & Clearance Line
    # -------------------------------------------------------------------------
    def select_obstacle(self, obstacle_id: Optional[int]):
        """Highlight selected obstacle wireframe and draw clearance measurement line."""
        self.selected_obstacle_id = obstacle_id

        if obstacle_id is None or self.current_scene is None or not self.current_scene.obstacles:
            self.visual_selected_bbox.visible = False
            self.visual_clearance_line.visible = False
            self.visual_clearance_markers.visible = False
            self.canvas.update()
            return

        # Locate obstacle in scene data
        matched_obs = next((o for o in self.current_scene.obstacles if o["obstacle_id"] == obstacle_id), None)
        if matched_obs is None:
            self.visual_selected_bbox.visible = False
            self.visual_clearance_line.visible = False
            self.visual_clearance_markers.visible = False
            self.canvas.update()
            return

        # 1. Selected Bounding Box (Gold wireframe)
        if "bbox" in matched_obs:
            b_min = np.array(matched_obs["bbox"]["min_point"], dtype=np.float32)
            b_max = np.array(matched_obs["bbox"]["max_point"], dtype=np.float32)
        else:
            c = np.array(matched_obs.get("centroid", [0, 0, 0]), dtype=np.float32)
            d = np.array(matched_obs.get("dimensions", [1, 1, 1]), dtype=np.float32)
            b_min = c - d / 2.0
            b_max = c + d / 2.0
        corners = self._get_box_corners(b_min, b_max)
        sel_line_pts: List[np.ndarray] = []
        self._append_box_edges(corners, sel_line_pts)

        sel_pos = np.vstack(sel_line_pts)
        sel_cols = np.tile(np.array([1.0, 0.90, 0.10, 1.0], dtype=np.float32), (len(sel_pos), 1))
        self.visual_selected_bbox.set_data(pos=sel_pos, color=sel_cols)
        self.visual_selected_bbox.visible = self.show_bboxes

        # 2. Dynamic Clearance Line & Endpoints
        p_veh, p_obs, dist = self.vehicle_model.compute_clearance_endpoints(b_min, b_max)
        clr_pts = np.vstack([p_veh, p_obs]).astype(np.float32)
        clr_cols = np.array([
            [1.0, 0.20, 0.70, 1.0],  # Neon pink / magenta
            [1.0, 0.20, 0.70, 1.0],
        ], dtype=np.float32)
        self.visual_clearance_line.set_data(pos=clr_pts, color=clr_cols)
        self.visual_clearance_line.visible = self.show_clearance_line

        # Endpoint markers
        marker_cols = np.array([
            [0.0, 0.90, 1.0, 1.0],  # Cyan at vehicle contact point
            [1.0, 0.85, 0.1, 1.0],  # Gold at obstacle contact point
        ], dtype=np.float32)
        self.visual_clearance_markers.set_data(clr_pts, edge_color=None, face_color=marker_cols, size=9.0)
        self.visual_clearance_markers.visible = self.show_clearance_line

        self.canvas.update()

    # -------------------------------------------------------------------------
    # Mouse Interaction / Click Picking
    # -------------------------------------------------------------------------
    def _on_mouse_press(self, event):
        """Handle mouse click in 3D viewport for obstacle selection."""
        if event.button != 1:  # Left click only
            return
        if self.current_scene is None or not self.current_scene.obstacles:
            return

        # Extract centroids of all obstacles
        centroids = np.array([o["centroid"] for o in self.current_scene.obstacles], dtype=np.float32)
        # Map centroids to screen coordinates
        sc_pts = self.view.scene.transform.map(centroids)

        click_x, click_y = event.pos
        # Note: on Retina displays, scene transform coordinates match canvas size
        dists = np.hypot(sc_pts[:, 0] - click_x, sc_pts[:, 1] - click_y)
        closest_idx = int(np.argmin(dists))

        if dists[closest_idx] < 60.0:  # Click tolerance in pixels
            selected_id = self.current_scene.obstacles[closest_idx]["obstacle_id"]
            self.select_obstacle(selected_id)
            self.obstacle_selected.emit(selected_id)

    # -------------------------------------------------------------------------
    # Layer Visibility Controls
    # -------------------------------------------------------------------------
    def set_point_cloud_mode(self, mode: str):
        """Set point cloud mode: 'rgb', 'semantic', or 'none'."""
        self.point_cloud_mode = mode
        self._update_point_cloud()
        self.canvas.update()

    def set_point_size(self, size: float):
        """Set point cloud marker size."""
        self.point_size = max(0.5, float(size))
        self._update_point_cloud()
        self.canvas.update()

    def set_occupancy_visibility(self, occupied: bool, free: bool, unknown: bool = False):
        """Toggle occupied, free, and unknown voxel grid layers."""
        self.show_occupied_voxels = occupied
        self.show_free_voxels = free
        self.show_unknown_voxels = unknown
        self._update_occupancy()
        self.canvas.update()

    def set_vehicle_visible(self, visible: bool):
        """Toggle 3D vehicle model visibility."""
        self.show_vehicle = visible
        self.visual_vehicle.visible = visible
        self.canvas.update()

    def set_envelope_visible(self, visible: bool):
        """Toggle 3D vehicle clearance envelope wireframe."""
        self.show_envelope = visible
        self.visual_envelope.visible = visible
        self.canvas.update()

    def set_bboxes_visible(self, visible: bool):
        """Toggle obstacle 3D bounding boxes."""
        self.show_bboxes = visible
        self.visual_bboxes.visible = visible
        if self.selected_obstacle_id is not None:
            self.visual_selected_bbox.visible = visible
        self.canvas.update()

    def set_clearance_line_visible(self, visible: bool):
        """Toggle dynamic clearance measurement line."""
        self.show_clearance_line = visible
        if self.selected_obstacle_id is not None:
            self.visual_clearance_line.visible = visible
            self.visual_clearance_markers.visible = visible
        self.canvas.update()

    def set_grid_visible(self, visible: bool):
        """Toggle ground reference grid and axes."""
        self.show_grid = visible
        self.visual_grid.visible = visible
        self.visual_axes.visible = visible
        self.canvas.update()

    # -------------------------------------------------------------------------
    # Camera Presets & Navigation
    # -------------------------------------------------------------------------
    def set_camera_preset(self, preset_name: str):
        """Switch to standard automotive camera preset."""
        name = preset_name.lower()
        if name == "reset":
            name = "chase"
        if name not in self.CAMERA_PRESETS:
            return

        cfg = self.CAMERA_PRESETS[name]
        self.view.camera.elevation = cfg["elevation"]
        self.view.camera.azimuth = cfg["azimuth"]
        self.view.camera.distance = cfg["distance"]
        self.view.camera.center = cfg["center"]
        self.request_update()

    def render_offscreen(self) -> np.ndarray:
        """Render high-resolution frame image (RGBA uint8)."""
        return self.canvas.render()

    def load_external_mesh(self, mesh_path: str):
        """
        Load an external photogrammetry mesh from PLY or a textured GLB.
        The GLB path is preferred for OpenMVS outputs because it embeds its atlas.
        """
        from pathlib import Path
        import open3d as o3d

        path = Path(mesh_path)
        if not path.exists():
            self.canvas.native.setToolTip(f"Reconstruction output not found: {path}")
            return

        if path.suffix.lower() == ".obj":
            # Parse face-corner (v/vt/vn) indices together. OBJ permits
            # independent position and UV indices, especially at texture seams.
            try:
                from vispy.visuals.filters import TextureFilter
                from viewer.obj_texture_loader import load_textured_obj

                verts, faces, normals, uvs, texture_rgb, atlas_path = load_textured_obj(path)
                # OBJ texture coordinates use a bottom-left origin. VisPy's
                # image texture array is row-major from the top, so flip once.
                texture_rgb = np.ascontiguousarray(np.flipud(texture_rgb))
                mesh_kwargs = {
                    "vertices": verts,
                    "faces": faces,
                    "color": "white",
                }
                if normals is not None and normals.shape == verts.shape:
                    mesh_kwargs["vertex_normals"] = normals
                self.visual_textured_mesh.set_data(**mesh_kwargs)
                if self.texture_filter is None:
                    self.texture_filter = TextureFilter(
                        texture=texture_rgb, texcoords=uvs
                    )
                    self.visual_textured_mesh.attach(self.texture_filter)
                else:
                    self.texture_filter.texture = texture_rgb
                    self.texture_filter.texcoords = uvs

                self.visual_textured_mesh.visible = True
                self.visual_surface_mesh.visible = False
                bounds_min, bounds_max = np.percentile(verts, [1.0, 99.0], axis=0)
                center = (bounds_min + bounds_max) * 0.5
                extent = float(np.linalg.norm(bounds_max - bounds_min))
                self.view.camera.center = tuple(center)
                self.view.camera.distance = max(3.0, extent * 1.8)
                self.view.camera.elevation = 20.0
                self.view.camera.azimuth = -45.0
                self._hide_non_scene_overlays()
                self.request_update()
                print(
                    "Loaded textured OBJ with independent UV indices: "
                    f"{len(verts):,} expanded vertices, {len(faces):,} triangles, "
                    f"atlas={atlas_path.name}"
                )
                return
            except Exception as exc:
                print(f"Notice: OBJ texture loading failed; using geometry fallback: {exc}")
                self.canvas.native.setToolTip(
                    f"Texture atlas could not be loaded: {exc}"
                )
            # If the material/UV package is invalid, still show the geometry.
        if path.suffix.lower() in {".glb", ".gltf"}:
            try:
                import trimesh
                loaded = trimesh.load(str(path), force="scene", process=False)
                if isinstance(loaded, trimesh.Scene):
                    # Flatten the scene graph so node transforms are applied before display.
                    try:
                        loaded = loaded.to_geometry()
                    except Exception:
                        geometries = []
                        for node_name in loaded.graph.nodes_geometry:
                            transform, geometry_name = loaded.graph.get(node_name)
                            geometry = loaded.geometry[geometry_name].copy()
                            geometry.apply_transform(transform)
                            geometries.append(geometry)
                        loaded = trimesh.util.concatenate(geometries)
                mesh = loaded
                if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                    raise ValueError("GLB contains no mesh geometry.")
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.uint32)
                visual = getattr(mesh, "visual", None)
                colors = None
                if visual is not None and hasattr(visual, "vertex_colors"):
                    vc = np.asarray(visual.vertex_colors)
                    if len(vc) == len(verts):
                        colors = vc.astype(np.float32) / (255.0 if vc.max(initial=0) > 1.0 else 1.0)
                if colors is None or colors.shape[1] < 3:
                    colors = np.ones((len(verts), 4), dtype=np.float32) * 0.78
                elif colors.shape[1] == 3:
                    colors = np.hstack([colors[:, :3], np.ones((len(colors), 1), dtype=np.float32)])
                self.visual_surface_mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors)
                self.visual_surface_mesh.visible = True
                self.visual_textured_mesh.visible = False
                center = np.mean(verts, axis=0)
                extent = float(np.max(np.linalg.norm(verts - center, axis=1)))
                self.view.camera.center = tuple(center)
                self.view.camera.distance = max(4.0, extent * 2.2)
                self.view.camera.elevation = 25.0
                self.view.camera.azimuth = -45.0
                self._hide_non_scene_overlays()
                self.request_update()
                return
            except Exception as exc:
                print(f"Notice: GLB preview fallback: {exc}")
                fallback_ply = path.with_suffix(".ply")
                if fallback_ply.exists():
                    path = fallback_ply
                else:
                    self.canvas.native.setToolTip(
                        "GLB preview requires trimesh. Install it with: pip install trimesh"
                    )
                    return

        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            # Fast Preview exports a dense PLY point cloud, not a triangle mesh.
            # Detect and display that explicitly rather than silently showing
            # an empty surface visual.
            point_cloud = o3d.io.read_point_cloud(str(path))
            if len(point_cloud.points) == 0:
                return
            points = np.asarray(point_cloud.points, dtype=np.float32)
            point_colors = np.asarray(point_cloud.colors, dtype=np.float32)
            if len(point_colors) != len(points):
                point_colors = np.ones((len(points), 3), dtype=np.float32) * 0.78
            point_rgba = np.hstack([
                point_colors[:, :3],
                np.ones((len(point_colors), 1), dtype=np.float32),
            ])
            self.visual_cloud.set_data(
                points, edge_color=point_rgba, face_color=point_rgba, size=1.8
            )
            self.visual_cloud.visible = True
            self.visual_surface_mesh.visible = False
            self.visual_textured_mesh.visible = False
            center = np.median(points, axis=0)
            extent = float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))
            self.view.camera.center = tuple(center)
            self.view.camera.distance = max(3.0, extent * 2.2)
            self.view.camera.elevation = 25.0
            self.view.camera.azimuth = -45.0
            self._hide_non_scene_overlays()
            # The generic helper hides automotive point-cloud overlays too;
            # explicitly re-enable the photogrammetry cloud after cleanup.
            self.visual_cloud.visible = True
            self.canvas.native.setToolTip(
                f"Fast Preview point cloud: {len(points):,} points (not a textured mesh)"
            )
            self.request_update()
            print(f"Loaded Fast Preview point cloud: {len(points):,} points")
            return

        self.visual_cloud.visible = False
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        cols = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if len(cols) == 0:
            cols = np.ones((len(verts), 4), dtype=np.float32) * 0.75
        elif cols.shape[1] == 3:
            cols = np.hstack([cols, np.ones((len(cols), 1), dtype=np.float32)])

        self.visual_surface_mesh.set_data(vertices=verts, faces=faces, vertex_colors=cols)
        self.visual_surface_mesh.visible = True
        self.visual_textured_mesh.visible = False
        self.visual_vehicle.visible = False

        # Preserve the native CLEAR360 Python backend's textured OBJ/PNG output.
        session_dir = path.parent.parent
        atlas_png = session_dir / "mesh" / "scene_mesh.png"
        obj_file = session_dir / "mesh" / "scene_mesh.obj"
        if atlas_png.exists() and obj_file.exists():
            try:
                from vispy.visuals.filters import TextureFilter
                tex_bgr = cv2.imread(str(atlas_png))
                if tex_bgr is not None:
                    tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
                    obj_v, obj_vt, obj_vn, obj_f = [], [], [], []
                    with obj_file.open("r", encoding="utf-8", errors="replace") as f_obj:
                        for line in f_obj:
                            if line.startswith("v "):
                                obj_v.append([float(x) for x in line.split()[1:4]])
                            elif line.startswith("vt "):
                                obj_vt.append([float(x) for x in line.split()[1:3]])
                            elif line.startswith("vn "):
                                obj_vn.append([float(x) for x in line.split()[1:4]])
                            elif line.startswith("f "):
                                parts = line.split()[1:4]
                                obj_f.append([int(pt.split("/")[0]) - 1 for pt in parts])
                    if len(obj_v) > 0 and len(obj_vt) == len(obj_v) and len(obj_f) > 0:
                        t_verts = np.asarray(obj_v, dtype=np.float32)
                        t_faces = np.asarray(obj_f, dtype=np.uint32)
                        t_uvs = np.asarray(obj_vt, dtype=np.float32)
                        t_norms = np.asarray(obj_vn, dtype=np.float32) if len(obj_vn) == len(obj_v) else None
                        self.visual_textured_mesh.set_data(
                            vertices=t_verts, faces=t_faces, vertex_normals=t_norms, color="white"
                        )
                        if self.texture_filter is None:
                            self.texture_filter = TextureFilter(texture=tex_rgb, texcoords=t_uvs)
                            self.visual_textured_mesh.attach(self.texture_filter)
                        else:
                            self.texture_filter.texture = tex_rgb
                            self.texture_filter.texcoords = t_uvs
                        self.visual_textured_mesh.visible = True
                        self.visual_surface_mesh.visible = False
            except Exception as exc:
                print(f"Notice: Native texture loading fallback to vertex colors: {exc}")

        dense_p = session_dir / "dense_cloud" / "dense_cloud.ply"
        if dense_p.exists():
            pcd = o3d.io.read_point_cloud(str(dense_p))
            if len(pcd.points) > 0:
                p_pts = np.asarray(pcd.points, dtype=np.float32)
                p_cols = np.asarray(pcd.colors, dtype=np.float32)
                if len(p_cols) == len(p_pts):
                    p_rgba = np.hstack([p_cols, np.ones((len(p_cols), 1), dtype=np.float32)])
                else:
                    p_rgba = np.ones((len(p_pts), 4), dtype=np.float32) * 0.8
                self.visual_cloud.set_data(p_pts, edge_color=p_rgba, face_color=p_rgba, size=1.5)
        self.visual_cloud.visible = False

        center = np.mean(verts, axis=0)
        extent = float(np.max(np.linalg.norm(verts - center, axis=1)))
        self.view.camera.center = tuple(center)
        self.view.camera.distance = max(4.0, extent * 2.2)
        self.view.camera.elevation = 25.0
        self.view.camera.azimuth = -45.0
        self._hide_non_scene_overlays()
        self.request_update()

    def _hide_non_scene_overlays(self):
        """Hide automotive overlays while inspecting an independent photogrammetry scene."""
        self.visual_vehicle.visible = False
        self.visual_cloud.visible = False
        self.visual_occ_occupied.visible = False
        self.visual_occ_free.visible = False
        self.visual_occ_unknown.visible = False
        self.visual_bboxes.visible = False
        self.visual_selected_bbox.visible = False
        self.visual_clearance_line.visible = False
        self.visual_clearance_markers.visible = False
        self.visual_envelope.visible = False
        self.visual_grid.visible = True
        self.visual_axes.visible = True

