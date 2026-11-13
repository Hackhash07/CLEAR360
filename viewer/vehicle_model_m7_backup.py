"""
viewer/vehicle_model.py — 3D Vehicle Model & Clearance Geometry for 2022 Hyundai Creta

Builds accurate 3D visual representations:
  - 3D Stylized SUV mesh (body panels, cabin glass, roof, headlights, taillights, side mirrors, wheels)
  - 3D Clearance Envelope wireframe (body box + side mirrors extending to +/- 1.05 m)
  - Vehicle coordinate axes (+X Right Red, +Y Forward Green, +Z Up Blue)
  - Exact Euclidean clearance point-to-box calculations for dynamic measurement lines
"""

from typing import Tuple, List, Dict, Any
import numpy as np


class VehicleModel3D:
    """
    3D geometric model of the 2022 Hyundai Creta positioned at vehicle frame origin (0, 0, 0).
    Dimensions (metres):
      - Length: 4.30 m  (Y: -2.15 to +2.15)
      - Width:  1.79 m  (X: -0.895 to +0.895)
      - Total Width with Mirrors: 2.10 m (X: -1.05 to +1.05)
      - Height: 1.635 m (Z: 0.0 to 1.635)
      - Ground Clearance: 0.19 m
      - Wheelbase: 2.61 m (Front axle Y = +1.305, Rear axle Y = -1.305)
      - Track Width: 1.57 m (X = +/- 0.785)
      - Wheel Radius: 0.34 m
    """

    def __init__(self):
        self.half_width = 0.895
        self.half_length = 2.15
        self.height = 1.635
        self.ground_clearance = 0.19
        self.mirror_tip_x = 1.05
        self.mirror_y_bounds = (0.35, 0.45)
        self.mirror_z_bounds = (0.90, 1.10)

        # Composite clearance bounding boxes (body, left mirror, right mirror)
        self.body_box = (
            np.array([-self.half_width, -self.half_length, 0.0], dtype=np.float32),
            np.array([+self.half_width, +self.half_length, self.height], dtype=np.float32),
        )
        self.left_mirror_box = (
            np.array([-self.mirror_tip_x, self.mirror_y_bounds[0], self.mirror_z_bounds[0]], dtype=np.float32),
            np.array([-self.half_width, self.mirror_y_bounds[1], self.mirror_z_bounds[1]], dtype=np.float32),
        )
        self.right_mirror_box = (
            np.array([+self.half_width, self.mirror_y_bounds[0], self.mirror_z_bounds[0]], dtype=np.float32),
            np.array([+self.mirror_tip_x, self.mirror_y_bounds[1], self.mirror_z_bounds[1]], dtype=np.float32),
        )

    def get_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates combined 3D polygonal mesh vertices, faces, and vertex colors
        representing the Hyundai Creta SUV.
        Returns:
            vertices: (V, 3) float32
            faces: (F, 3) uint32
            colors: (V, 4) float32 RGBA
        """
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        v_offset = 0

        def add_submesh(v_arr: np.ndarray, f_arr: np.ndarray, color_rgba: Tuple[float, float, float, float]):
            nonlocal v_offset
            all_verts.append(v_arr)
            all_faces.append(f_arr + v_offset)
            cols = np.tile(np.array(color_rgba, dtype=np.float32), (len(v_arr), 1))
            all_colors.append(cols)
            v_offset += len(v_arr)

        def make_box_mesh(x0, x1, y0, y1, z0, z1):
            v = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top
            ], dtype=np.float32)
            f = np.array([
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 6, 5], [4, 7, 6],  # top
                [0, 5, 1], [0, 4, 5],  # rear (-Y)
                [2, 7, 3], [2, 6, 7],  # front (+Y)
                [0, 3, 7], [0, 7, 4],  # left (-X)
                [1, 5, 6], [1, 6, 2],  # right (+X)
            ], dtype=np.uint32)
            return v, f

        # 1. Lower Body / Chassis (Polar White / Metallic Slate Blue)
        # Main body chassis: X in [-0.88, 0.88], Y in [-2.10, 2.10], Z in [0.20, 0.92]
        v_chassis, f_chassis = make_box_mesh(-0.88, 0.88, -2.10, 2.10, 0.20, 0.92)
        add_submesh(v_chassis, f_chassis, (0.18, 0.38, 0.68, 0.88))

        # 2. Hood / Front Nose (slightly lower slope towards front bumper)
        v_hood, f_hood = make_box_mesh(-0.85, 0.85, 0.70, 2.12, 0.85, 0.95)
        add_submesh(v_hood, f_hood, (0.16, 0.34, 0.62, 0.90))

        # 3. Cabin / Greenhouse (Dark Tinted Glass & Pillars)
        # Starts at Y in [-1.90, 0.70], width tapered to [-0.78, 0.78], Z in [0.92, 1.62]
        v_cabin, f_cabin = make_box_mesh(-0.78, 0.78, -1.90, 0.70, 0.92, 1.62)
        add_submesh(v_cabin, f_cabin, (0.10, 0.15, 0.22, 0.85))

        # 4. Roof Panel (Polar White / Body Color)
        v_roof, f_roof = make_box_mesh(-0.76, 0.76, -1.85, 0.65, 1.58, 1.635)
        add_submesh(v_roof, f_roof, (0.22, 0.44, 0.78, 0.95))

        # 5. Headlights (Bright Xenon White/Cyan on front bumper)
        v_hl_l, f_hl_l = make_box_mesh(-0.82, -0.45, 2.10, 2.13, 0.70, 0.85)
        v_hl_r, f_hl_r = make_box_mesh(0.45, 0.82, 2.10, 2.13, 0.70, 0.85)
        add_submesh(v_hl_l, f_hl_l, (0.80, 0.95, 1.00, 1.0))
        add_submesh(v_hl_r, f_hl_r, (0.80, 0.95, 1.00, 1.0))

        # 6. Taillights (Bright Ruby Red on rear bumper)
        v_tl_l, f_tl_l = make_box_mesh(-0.84, -0.45, -2.13, -2.10, 0.75, 0.88)
        v_tl_r, f_tl_r = make_box_mesh(0.45, 0.84, -2.13, -2.10, 0.75, 0.88)
        add_submesh(v_tl_l, f_tl_l, (1.00, 0.12, 0.15, 1.0))
        add_submesh(v_tl_r, f_tl_r, (1.00, 0.12, 0.15, 1.0))

        # 7. Side Mirrors (Left & Right extending to +/- 1.05 m)
        v_sm_l, f_sm_l = make_box_mesh(-1.05, -0.88, 0.35, 0.45, 0.90, 1.10)
        v_sm_r, f_sm_r = make_box_mesh(0.88, 1.05, 0.35, 0.45, 0.90, 1.10)
        add_submesh(v_sm_l, f_sm_l, (0.12, 0.25, 0.45, 0.95))
        add_submesh(v_sm_r, f_sm_r, (0.12, 0.25, 0.45, 0.95))

        # 8. Wheels (4 wheels: front +/- 0.785, Y=+1.305; rear +/- 0.785, Y=-1.305)
        wheel_positions = [
            (-0.84, 1.305),  # Front-Left
            (0.84, 1.305),   # Front-Right
            (-0.84, -1.305), # Rear-Left
            (0.84, -1.305),  # Rear-Right
        ]
        for wx, wy in wheel_positions:
            w_x0 = wx - 0.12 if wx > 0 else wx - 0.08
            w_x1 = wx + 0.08 if wx > 0 else wx + 0.12
            v_wh, f_wh = make_box_mesh(w_x0, w_x1, wy - 0.32, wy + 0.32, 0.0, 0.68)
            add_submesh(v_wh, f_wh, (0.15, 0.15, 0.18, 0.95))

        all_v = np.vstack(all_verts)
        all_f = np.vstack(all_faces)
        all_c = np.vstack(all_colors)
        return all_v, all_f, all_c

    def get_clearance_envelope_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D wireframe line segments for the composite vehicle clearance envelope:
        Body box + Left mirror box + Right mirror box.
        Returns:
            pos: (N, 3) float32 endpoints for connect='segments'
            color: (N, 4) float32 RGBA
        """
        line_pts: List[np.ndarray] = []

        def add_box_wireframe(b_min: np.ndarray, b_max: np.ndarray):
            x0, y0, z0 = b_min
            x1, y1, z1 = b_max
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 0,1,2,3 bottom
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 4,5,6,7 top
            ], dtype=np.float32)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
                (4, 5), (5, 6), (6, 7), (7, 4),  # top square
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
            ]
            for i1, i2 in edges:
                line_pts.append(corners[i1])
                line_pts.append(corners[i2])

        add_box_wireframe(self.body_box[0], self.body_box[1])
        add_box_wireframe(self.left_mirror_box[0], self.left_mirror_box[1])
        add_box_wireframe(self.right_mirror_box[0], self.right_mirror_box[1])

        pos = np.vstack(line_pts).astype(np.float32)
        # Cyan neon highlight wireframe (0.0, 0.9, 1.0, 0.8)
        colors = np.tile(np.array([0.0, 0.85, 1.0, 0.8], dtype=np.float32), (len(pos), 1))
        return pos, colors

    def get_axes_lines(self, length: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Coordinate axes at vehicle origin:
          +X (Right): Red
          +Y (Forward): Green
          +Z (Up): Blue
        """
        pts = np.array([
            [0, 0, 0.05], [length, 0, 0.05],       # +X Right
            [0, 0, 0.05], [0, length * 1.3, 0.05], # +Y Forward (slightly longer)
            [0, 0, 0.05], [0, 0, length + 0.05],   # +Z Up
        ], dtype=np.float32)

        cols = np.array([
            [1.0, 0.15, 0.15, 1.0], [1.0, 0.15, 0.15, 1.0],  # Red X
            [0.15, 1.0, 0.25, 1.0], [0.15, 1.0, 0.25, 1.0],  # Green Y
            [0.25, 0.45, 1.0, 1.0], [0.25, 0.45, 1.0, 1.0],  # Blue Z
        ], dtype=np.float32)

        return pts, cols

    def get_ground_grid_lines(self, size: float = 14.0, step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtle ground reference grid on Z=0 surface.
        """
        line_pts: List[np.ndarray] = []
        ticks = np.arange(-size, size + 1e-4, step, dtype=np.float32)

        for t in ticks:
            # Parallel to X
            line_pts.append(np.array([-size, t, 0.0], dtype=np.float32))
            line_pts.append(np.array([+size, t, 0.0], dtype=np.float32))
            # Parallel to Y
            line_pts.append(np.array([t, -size, 0.0], dtype=np.float32))
            line_pts.append(np.array([t, +size, 0.0], dtype=np.float32))

        pos = np.vstack(line_pts).astype(np.float32)
        cols = np.tile(np.array([0.25, 0.30, 0.38, 0.35], dtype=np.float32), (len(pos), 1))
        return pos, cols

    def compute_clearance_endpoints(
        self,
        obs_min: np.ndarray,
        obs_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes the closest 3D points between the composite Hyundai Creta envelope
        (body + mirrors) and an axis-aligned obstacle bounding box [obs_min, obs_max].
        Returns:
            p_veh: (3,) closest point on vehicle envelope
            p_obs: (3,) closest point on obstacle
            min_dist: Euclidean distance in metres
        """
        boxes = [self.body_box, self.left_mirror_box, self.right_mirror_box]
        best_dist = float("inf")
        best_p_veh = np.zeros(3, dtype=np.float32)
        best_p_obs = np.zeros(3, dtype=np.float32)

        for b_min, b_max in boxes:
            p1 = np.zeros(3, dtype=np.float32)
            p2 = np.zeros(3, dtype=np.float32)
            for i in range(3):
                if b_max[i] < obs_min[i]:
                    p1[i] = b_max[i]
                    p2[i] = obs_min[i]
                elif obs_max[i] < b_min[i]:
                    p1[i] = b_min[i]
                    p2[i] = obs_max[i]
                else:
                    mid = 0.5 * (max(b_min[i], obs_min[i]) + min(b_max[i], obs_max[i]))
                    p1[i] = mid
                    p2[i] = mid
            dist = float(np.linalg.norm(p1 - p2))
            if dist < best_dist:
                best_dist = dist
                best_p_veh = p1
                best_p_obs = p2

        return best_p_veh, best_p_obs, best_dist
