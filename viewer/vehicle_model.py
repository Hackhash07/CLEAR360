"""
viewer/vehicle_model.py — High-Fidelity 3D Vehicle Model for 2022 Hyundai Creta

Builds accurate, photorealistic 3D visual representations:
  - High-fidelity 3D SUV mesh with authentic Hyundai Creta proportions:
      * Sculpted multi-segment body panels in Gunmetal Metallic clearcoat
      * Cascading parametric front grille in dark chrome
      * Trio-beam LED projector headlights & slim upper DRL brows
      * Split C-shaped rear LED taillights with connecting red reflector bar
      * Sloping front hood / bonnet and rear tailgate with integrated spoiler
      * Raked front windshield, rear glass, and tinted side windows with pillars
      * Silver lightning arch C-pillar accent and aerodynamic silver roof rails
      * Four 3D cylindrical wheels with matte rubber tyres and alloy spoke rims
      * Aerodynamic side mirror assemblies with rearward-facing mirror glass
      * Front and rear metallic silver skid plates and lower cladding
  - 3D Clearance Envelope wireframe (body box + side mirrors extending to +/- 1.05 m)
  - Vehicle coordinate axes (+X Right Red, +Y Forward Green, +Z Up Blue)
  - Exact Euclidean clearance point-to-box calculations for dynamic measurement lines
"""

from typing import Tuple, List, Dict, Any, Optional
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
        representing the photorealistic Hyundai Creta SUV.
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
            all_verts.append(v_arr.astype(np.float32))
            all_faces.append((f_arr + v_offset).astype(np.uint32))
            cols = np.tile(np.array(color_rgba, dtype=np.float32), (len(v_arr), 1))
            all_colors.append(cols)
            v_offset += len(v_arr)

        def make_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
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

        def make_cylinder(cx: float, cy: float, cz: float, radius: float, width: float, num_segments: int = 16):
            angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
            y_ring = radius * np.cos(angles)
            z_ring = radius * np.sin(angles)

            v_left = np.column_stack([np.full(num_segments, cx - width / 2), cy + y_ring, cz + z_ring])
            v_right = np.column_stack([np.full(num_segments, cx + width / 2), cy + y_ring, cz + z_ring])
            idx_cl = 2 * num_segments
            idx_cr = 2 * num_segments + 1
            v_center_l = np.array([[cx - width / 2, cy, cz]])
            v_center_r = np.array([[cx + width / 2, cy, cz]])

            verts = np.vstack([v_left, v_right, v_center_l, v_center_r]).astype(np.float32)

            faces = []
            for i in range(num_segments):
                next_i = (i + 1) % num_segments
                faces.append([i, num_segments + i, num_segments + next_i])
                faces.append([i, num_segments + next_i, next_i])
                faces.append([idx_cl, next_i, i])
                faces.append([idx_cr, num_segments + i, num_segments + next_i])

            return verts, np.array(faces, dtype=np.uint32)

        def make_sloped_frustum(
            x_b: float, x_t: float,
            y0_b: float, y1_b: float,
            y0_t: float, y1_t: float,
            z0: float, z1: float,
        ):
            """Create sloped frustum (e.g. cabin greenhouse tapering inward)."""
            v = np.array([
                [-x_b, y0_b, z0], [+x_b, y0_b, z0], [+x_b, y1_b, z0], [-x_b, y1_b, z0],  # bottom
                [-x_t, y0_t, z1], [+x_t, y0_t, z1], [+x_t, y1_t, z1], [-x_t, y1_t, z1],  # top
            ], dtype=np.float32)
            f = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 5, 1], [0, 4, 5],
                [2, 7, 3], [2, 6, 7],
                [0, 3, 7], [0, 7, 4],
                [1, 5, 6], [1, 6, 2],
            ], dtype=np.uint32)
            return v, f

        # --- Automotive Material Palettes ---
        c_body = (0.20, 0.23, 0.28, 1.0)        # Metallic Gunmetal / Titanium Grey
        c_body_dark = (0.16, 0.18, 0.22, 1.0)   # Body creasing & shadow
        c_skid = (0.82, 0.85, 0.89, 1.0)        # Brushed Silver Skid Guard
        c_cladding = (0.08, 0.09, 0.11, 1.0)    # Matte Black SUV Protective Cladding
        c_glass = (0.07, 0.11, 0.15, 0.92)       # Dark Tinted Automotive Glass
        c_roof = (0.12, 0.14, 0.17, 1.0)        # Phantom Black Floating Roof
        c_silver_trim = (0.86, 0.88, 0.92, 1.0) # Lightning Arch & Roof Rails
        c_grille = (0.24, 0.24, 0.27, 1.0)      # Dark Chrome Parametric Grille
        c_led_hl = (0.90, 0.96, 1.00, 1.0)      # Trio-Beam Xenon / Ice Blue LED
        c_led_tl = (0.95, 0.08, 0.10, 1.0)      # Split C-shaped Ruby Red LED
        c_led_amb = (1.00, 0.65, 0.05, 1.0)     # Turn Signal Indicator Amber
        c_tyre = (0.05, 0.05, 0.06, 1.0)        # Carbon Black Matte Rubber
        c_alloy = (0.86, 0.88, 0.92, 1.0)       # Diamond-Cut Silver Alloy Rim
        c_mirror = (0.88, 0.92, 0.98, 1.0)      # Rearview Mirror Glass

        # ---------------------------------------------------------------------
        # 1. Lower Body & Protective Cladding
        # ---------------------------------------------------------------------
        # Underbody chassis cladding
        v_clad, f_clad = make_box(-0.88, 0.88, -2.10, 2.10, 0.19, 0.45)
        add_submesh(v_clad, f_clad, c_cladding)

        # Front silver skid guard
        v_fskid, f_fskid = make_box(-0.55, 0.55, 2.08, 2.14, 0.20, 0.38)
        add_submesh(v_fskid, f_fskid, c_skid)

        # Rear silver skid guard
        v_rskid, f_rskid = make_box(-0.55, 0.55, -2.14, -2.08, 0.22, 0.40)
        add_submesh(v_rskid, f_rskid, c_skid)

        # ---------------------------------------------------------------------
        # 2. Main Sculpted Body & Doors
        # ---------------------------------------------------------------------
        # Mid-body / Beltline core
        v_body, f_body = make_box(-0.895, 0.895, -2.15, 2.15, 0.45, 0.96)
        add_submesh(v_body, f_body, c_body)

        # Sloping Bonnet (Hood)
        # Slopes gently forward from cowl (Y=0.65, Z=0.96) down to nose (Y=2.14, Z=0.88)
        v_hood, f_hood = make_sloped_frustum(
            x_b=0.86, x_t=0.82,
            y0_b=0.65, y1_b=2.14,
            y0_t=0.65, y1_t=2.12,
            z0=0.85, z1=0.96,
        )
        add_submesh(v_hood, f_hood, c_body)

        # Rear Tailgate / Trunk section with subtle spoiler overhang
        v_tail, f_tail = make_box(-0.87, 0.87, -2.15, -1.85, 0.85, 1.04)
        add_submesh(v_tail, f_tail, c_body_dark)

        # ---------------------------------------------------------------------
        # 3. Greenhouse (Cabin, Windshields & Tinted Windows)
        # ---------------------------------------------------------------------
        # Tapered Cabin Glass Block (Greenhouse)
        # From Y=-1.85 to +0.65, Z=0.96 to 1.58, tapering from width 1.62m to 1.48m
        v_greenhouse, f_greenhouse = make_sloped_frustum(
            x_b=0.81, x_t=0.74,
            y0_b=-1.85, y1_b=0.65,
            y0_t=-1.65, y1_t=0.15,
            z0=0.96, z1=1.58,
        )
        add_submesh(v_greenhouse, f_greenhouse, c_glass)

        # A-Pillars, B-Pillars, C-Pillars dividers
        # B-Pillar divider (Left and Right)
        v_bp_l, f_bp_l = make_box(-0.79, -0.76, -0.45, -0.35, 0.96, 1.58)
        v_bp_r, f_bp_r = make_box(0.76, 0.79, -0.45, -0.35, 0.96, 1.58)
        add_submesh(v_bp_l, f_bp_l, c_cladding)
        add_submesh(v_bp_r, f_bp_r, c_cladding)

        # Floating Phantom Black Roof
        v_roof, f_roof = make_box(-0.74, 0.74, -1.70, 0.20, 1.57, 1.635)
        add_submesh(v_roof, f_roof, c_roof)

        # Silver Lightning Arch Trim (C-Pillar accent)
        v_arch_l, f_arch_l = make_box(-0.76, -0.73, -1.75, -1.30, 1.50, 1.63)
        v_arch_r, f_arch_r = make_box(0.73, 0.76, -1.75, -1.30, 1.50, 1.63)
        add_submesh(v_arch_l, f_arch_l, c_silver_trim)
        add_submesh(v_arch_r, f_arch_r, c_silver_trim)

        # Aerodynamic Silver Roof Rails (Left & Right)
        v_rail_l, f_rail_l = make_box(-0.64, -0.60, -1.55, 0.10, 1.635, 1.67)
        v_rail_r, f_rail_r = make_box(0.60, 0.64, -1.55, 0.10, 1.635, 1.67)
        add_submesh(v_rail_l, f_rail_l, c_silver_trim)
        add_submesh(v_rail_r, f_rail_r, c_silver_trim)

        # ---------------------------------------------------------------------
        # 4. Front Cascading Grille & Headlights
        # ---------------------------------------------------------------------
        # Creta Signature Cascading Parametric Front Grille
        v_grille, f_grille = make_box(-0.52, 0.52, 2.13, 2.155, 0.46, 0.84)
        add_submesh(v_grille, f_grille, c_grille)

        # Upper Slim DRL Eyebrows (Left & Right)
        v_drl_l, f_drl_l = make_box(-0.84, -0.54, 2.12, 2.145, 0.86, 0.90)
        v_drl_r, f_drl_r = make_box(0.54, 0.84, 2.12, 2.145, 0.86, 0.90)
        add_submesh(v_drl_l, f_drl_l, c_led_hl)
        add_submesh(v_drl_r, f_drl_r, c_led_hl)

        # Trio-Beam Projector LED Headlight Clusters (Lower Front Bumper)
        v_hl_l, f_hl_l = make_box(-0.84, -0.58, 2.12, 2.145, 0.58, 0.82)
        v_hl_r, f_hl_r = make_box(0.58, 0.84, 2.12, 2.145, 0.58, 0.82)
        add_submesh(v_hl_l, f_hl_l, c_led_hl)
        add_submesh(v_hl_r, f_hl_r, c_led_hl)

        # ---------------------------------------------------------------------
        # 5. Rear C-Shaped LED Taillights & Reflector Bar
        # ---------------------------------------------------------------------
        # Split C-shaped Ruby Red LED Taillights
        v_tl_l, f_tl_l = make_box(-0.86, -0.48, -2.155, -2.12, 0.72, 0.96)
        v_tl_r, f_tl_r = make_box(0.48, 0.86, -2.155, -2.12, 0.72, 0.96)
        add_submesh(v_tl_l, f_tl_l, c_led_tl)
        add_submesh(v_tl_r, f_tl_r, c_led_tl)

        # Center Red Connecting Reflector Bar
        v_bar, f_bar = make_box(-0.48, 0.48, -2.152, -2.13, 0.79, 0.83)
        add_submesh(v_bar, f_bar, c_led_tl)

        # ---------------------------------------------------------------------
        # 6. Sculpted Side Mirrors (Wing Mirrors extending to +/- 1.05 m)
        # ---------------------------------------------------------------------
        # Left Mirror Body
        v_sm_l, f_sm_l = make_box(-1.05, -0.895, 0.35, 0.45, 0.90, 1.10)
        add_submesh(v_sm_l, f_sm_l, c_body)
        # Left Mirror Glass (Rearward-facing at Y=0.35)
        v_gl_l, f_gl_l = make_box(-1.04, -0.91, 0.345, 0.352, 0.92, 1.08)
        add_submesh(v_gl_l, f_gl_l, c_mirror)
        # Left Turn Indicator Strip
        v_ind_l, f_ind_l = make_box(-1.052, -0.93, 0.42, 0.45, 0.98, 1.02)
        add_submesh(v_ind_l, f_ind_l, c_led_amb)

        # Right Mirror Body
        v_sm_r, f_sm_r = make_box(0.895, 1.05, 0.35, 0.45, 0.90, 1.10)
        add_submesh(v_sm_r, f_sm_r, c_body)
        # Right Mirror Glass
        v_gl_r, f_gl_r = make_box(0.91, 1.04, 0.345, 0.352, 0.92, 1.08)
        add_submesh(v_gl_r, f_gl_r, c_mirror)
        # Right Turn Indicator Strip
        v_ind_r, f_ind_r = make_box(0.93, 1.052, 0.42, 0.45, 0.98, 1.02)
        add_submesh(v_ind_r, f_ind_r, c_led_amb)

        # ---------------------------------------------------------------------
        # 7. Four 3D Cylindrical Wheels & Alloy Rims
        # ---------------------------------------------------------------------
        wheel_coords = [
            (-0.785, 1.305),   # Front-Left
            (+0.785, 1.305),   # Front-Right
            (-0.785, -1.305),  # Rear-Left
            (+0.785, -1.305),  # Rear-Right
        ]
        for wx, wy in wheel_coords:
            # Outer Rubber Tyre (radius 0.34 m, width 0.22 m)
            v_tyre, f_tyre = make_cylinder(wx, wy, cz=0.34, radius=0.34, width=0.22, num_segments=20)
            add_submesh(v_tyre, f_tyre, c_tyre)

            # Inner Diamond-Cut Alloy Rim (radius 0.24 m, slightly wider to show on face)
            rim_offset_x = wx + (0.005 if wx > 0 else -0.005)
            v_rim, f_rim = make_cylinder(rim_offset_x, wy, cz=0.34, radius=0.24, width=0.224, num_segments=16)
            add_submesh(v_rim, f_rim, c_alloy)

        # ---------------------------------------------------------------------
        # Consolidate and Return
        # ---------------------------------------------------------------------
        merged_verts = np.vstack(all_verts)
        merged_faces = np.vstack(all_faces)
        merged_colors = np.vstack(all_colors)

        return merged_verts, merged_faces, merged_colors

    def get_clearance_envelope_lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D line segments for the vehicle clearance safety envelope wireframe.
        Encloses the full vehicle body box plus the side wing mirror boxes.
        Returns:
            points: (P, 3) float32 coordinates for vispy visuals.Line (connect='segments')
            colors: (P, 4) float32 RGBA colors (electric cyan / neon aqua)
        """
        line_pts: List[np.ndarray] = []

        def add_box_wireframe(b_min: np.ndarray, b_max: np.ndarray):
            x0, y0, z0 = b_min
            x1, y1, z1 = b_max
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 0, 1, 2, 3 (bottom)
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 4, 5, 6, 7 (top)
            ], dtype=np.float32)

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
                (4, 5), (5, 6), (6, 7), (7, 4),  # top square
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
            ]
            for i1, i2 in edges:
                line_pts.append(corners[i1])
                line_pts.append(corners[i2])

        # 1. Main body envelope box
        add_box_wireframe(self.body_box[0], self.body_box[1])

        # 2. Left side mirror box
        add_box_wireframe(self.left_mirror_box[0], self.left_mirror_box[1])

        # 3. Right side mirror box
        add_box_wireframe(self.right_mirror_box[0], self.right_mirror_box[1])

        points = np.vstack(line_pts)
        # Electric cyan color with 0.85 alpha
        colors = np.tile(np.array([0.0, 0.90, 1.00, 0.85], dtype=np.float32), (len(points), 1))
        return points, colors

    def get_axes_lines(self, length: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coordinate axes lines at vehicle frame origin (0, 0, 0):
          +X (Right): Red
          +Y (Forward): Green
          +Z (Up): Blue
        """
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pts = np.array([
            origin, [length, 0.0, 0.0],     # +X Red
            origin, [0.0, length, 0.0],     # +Y Green
            origin, [0.0, 0.0, length],     # +Z Blue
        ], dtype=np.float32)

        cols = np.array([
            [1.0, 0.2, 0.2, 1.0], [1.0, 0.2, 0.2, 1.0],
            [0.2, 0.9, 0.3, 1.0], [0.2, 0.9, 0.3, 1.0],
            [0.2, 0.5, 1.0, 1.0], [0.2, 0.5, 1.0, 1.0],
        ], dtype=np.float32)

        return pts, cols

    def get_ground_grid_lines(self, size: float = 15.0, step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a planar ground reference grid at Z = 0.
        """
        line_pts: List[np.ndarray] = []
        ticks = np.arange(-size, size + 1e-4, step, dtype=np.float32)

        for t in ticks:
            # Parallel to Y axis
            line_pts.append(np.array([t, -size, 0.0], dtype=np.float32))
            line_pts.append(np.array([t, +size, 0.0], dtype=np.float32))
            # Parallel to X axis
            line_pts.append(np.array([-size, t, 0.0], dtype=np.float32))
            line_pts.append(np.array([+size, t, 0.0], dtype=np.float32))

        pts = np.vstack(line_pts)
        cols = np.tile(np.array([0.22, 0.25, 0.32, 0.45], dtype=np.float32), (len(pts), 1))
        return pts, cols

    def compute_clearance_endpoints(
        self,
        obstacle_min: np.ndarray,
        obstacle_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute minimum Euclidean distance and exact contact points between the
        composite vehicle envelope (body + mirrors) and an axis-aligned obstacle box.
        """
        min_dist = float("inf")
        best_p_veh = np.zeros(3, dtype=np.float32)
        best_p_obs = np.zeros(3, dtype=np.float32)

        vehicle_boxes = [self.body_box, self.left_mirror_box, self.right_mirror_box]

        for b_min, b_max in vehicle_boxes:
            p_v, p_o, dist = self._box_to_box_closest_points(b_min, b_max, obstacle_min, obstacle_max)
            if dist < min_dist:
                min_dist = dist
                best_p_veh = p_v
                best_p_obs = p_o

        return best_p_veh, best_p_obs, float(min_dist)

    @staticmethod
    def _box_to_box_closest_points(
        box1_min: np.ndarray, box1_max: np.ndarray,
        box2_min: np.ndarray, box2_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute closest points between two 3D AABBs."""
        p1 = np.zeros(3, dtype=np.float32)
        p2 = np.zeros(3, dtype=np.float32)

        for i in range(3):
            if box1_max[i] < box2_min[i]:
                p1[i] = box1_max[i]
                p2[i] = box2_min[i]
            elif box2_max[i] < box1_min[i]:
                p1[i] = box1_min[i]
                p2[i] = box2_max[i]
            else:
                overlap_mid = 0.5 * (max(box1_min[i], box2_min[i]) + min(box1_max[i], box2_max[i]))
                p1[i] = overlap_mid
                p2[i] = overlap_mid

        dist = float(np.linalg.norm(p1 - p2))
        return p1, p2, dist
