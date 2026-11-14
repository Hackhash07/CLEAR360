"""
viewer/data_loader.py — Multi-Pipeline Data Loader for CLEAR360 3D Viewer

Provides high-performance, cached loading of 3D point clouds (RGB and Semantic),
occupancy voxel grids, detected 3D obstacles, and scene metadata across all 3
reconstruction pipelines:
  1. Ground Truth (Blender Cycles photoreal simulation)
  2. Calibrated AI (Depth Anything V2 with per-camera polynomial/linear calibration)
  3. Raw AI (Uncalibrated Depth Anything V2)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import open3d as o3d
import cv2


@dataclass
class SceneData:
    """Container for all 3D assets and metadata associated with a single frame and pipeline source."""
    frame_id: int
    source: str
    points_rgb: np.ndarray  # (N, 3) float32
    colors_rgb: np.ndarray  # (N, 3) float32 in [0, 1]
    points_semantic: np.ndarray  # (N, 3) float32
    colors_semantic: np.ndarray  # (N, 3) float32 in [0, 1]
    occupancy_occupied_points: np.ndarray  # (M, 3) float32
    occupancy_free_points: np.ndarray  # (K, 3) float32 (subsampled)
    obstacles: List[Dict[str, Any]]
    scene_metadata: Dict[str, Any]
    occupancy_unknown_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    points_confidence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    colors_confidence: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    mesh_vertices: Optional[np.ndarray] = None  # (V, 3) float32
    mesh_faces: Optional[np.ndarray] = None     # (F, 3) uint32
    mesh_colors: Optional[np.ndarray] = None    # (V, 4) float32 in [0, 1]
    mesh_normals: Optional[np.ndarray] = None   # (V, 3) float32
    textured_mesh_vertices: Optional[np.ndarray] = None  # (V, 3) float32
    textured_mesh_faces: Optional[np.ndarray] = None     # (F, 3) uint32
    textured_mesh_uvs: Optional[np.ndarray] = None       # (V, 2) float32
    textured_mesh_texture: Optional[np.ndarray] = None   # (H, W, 3) uint8 RGB
    scenario_id: Optional[str] = None
    point_count: int = 0
    mesh_vertex_count: int = 0
    mesh_face_count: int = 0

    def __post_init__(self):
        self.point_count = len(self.points_rgb)
        if self.mesh_vertices is not None:
            self.mesh_vertex_count = len(self.mesh_vertices)
        if self.mesh_faces is not None:
            self.mesh_face_count = len(self.mesh_faces)


class ViewerDataLoader:
    """
    High-level data loader with in-memory caching for real-time viewer switching.
    """

    SOURCE_PREFIXES = {
        "ground_truth": "ground_truth",
        "ai_calibrated": "ai_per_camera",
        "ai_raw": "ai_raw",
    }

    SOURCE_LABELS = {
        "ground_truth": "Ground Truth (Cycles)",
        "ai_calibrated": "Calibrated AI (Depth Anything V2)",
        "ai_raw": "Raw AI (Uncalibrated)",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.pointcloud_dir = self.data_dir / "pointclouds"
        self.occupancy_dir = self.data_dir / "occupancy"
        self.metadata_dir = self.data_dir / "metadata"
        self.mesh_dir = self.data_dir / "meshes"
        self.textured_mesh_dir = self.data_dir / "textured_meshes"

        self.available_frames = [0, 5, 10]
        self.available_sources = ["ground_truth", "ai_calibrated", "ai_raw"]
        self.available_scenarios = [
            "scenario_01_narrow_street",
            "scenario_02_double_parked",
            "scenario_03_close_pillars_walls",
            "scenario_04_tight_parking_gap",
            "scenario_05_speed_breakers_kerbs",
            "scenario_06_approaching_pedestrians",
            "scenario_07_harsh_lighting_shadows",
            "scenario_08_partially_occluded",
            "scenario_09_t_junction_turn",
            "scenario_10_construction_zone",
        ]

        # In-memory cache: (frame_or_scenario_id, source) -> SceneData
        self._cache: Dict[Tuple[Any, str], SceneData] = {}

    def get_source_prefix(self, source: str) -> str:
        """Resolve pipeline source name to filename prefix."""
        if source not in self.SOURCE_PREFIXES:
            raise ValueError(f"Unknown source: '{source}'. Available: {list(self.SOURCE_PREFIXES.keys())}")
        return self.SOURCE_PREFIXES[source]

    def load_scene(self, frame_id: int, source: str, force_reload: bool = False) -> SceneData:
        """
        Load complete 3D scene data for specified frame and pipeline source.
        Uses in-memory cache to ensure near-zero latency (<50 ms) during viewer interaction.
        """
        cache_key = (frame_id, source)
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]

        prefix = self.get_source_prefix(source)
        frame_str = f"frame_{frame_id:04d}"

        # 1. Load RGB Point Cloud
        rgb_ply_path = self.pointcloud_dir / f"{frame_str}_{prefix}_clean.ply"
        if not rgb_ply_path.exists():
            raise FileNotFoundError(f"RGB pointcloud not found: {rgb_ply_path}")

        pcd_rgb = o3d.io.read_point_cloud(str(rgb_ply_path))
        pts_rgb = np.asarray(pcd_rgb.points, dtype=np.float32)
        cols_rgb = np.asarray(pcd_rgb.colors, dtype=np.float32)
        if len(cols_rgb) == 0:
            cols_rgb = np.ones((len(pts_rgb), 3), dtype=np.float32) * 0.7

        # 2. Load Semantic Point Cloud
        sem_ply_path = self.pointcloud_dir / f"{frame_str}_{prefix}_semantic.ply"
        if sem_ply_path.exists():
            pcd_sem = o3d.io.read_point_cloud(str(sem_ply_path))
            pts_sem = np.asarray(pcd_sem.points, dtype=np.float32)
            cols_sem = np.asarray(pcd_sem.colors, dtype=np.float32)
        else:
            pts_sem = pts_rgb
            cols_sem = cols_rgb

        # Confidence map based on distance to ego-vehicle
        if len(pts_rgb) > 0:
            dists = np.linalg.norm(pts_rgb, axis=1)
            conf = np.clip(1.0 - (dists / 16.0), 0.05, 1.0).astype(np.float32)
            r = np.clip(2.0 * (1.0 - conf), 0.0, 1.0)
            g = np.clip(2.0 * conf, 0.0, 1.0)
            b = np.full_like(conf, 0.1)
            cols_conf = np.column_stack([r, g, b]).astype(np.float32)
        else:
            conf = np.zeros(0, dtype=np.float32)
            cols_conf = np.zeros((0, 3), dtype=np.float32)

        # 3. Load Occupancy Grid and extract 3D occupied/free/unknown voxel centroids
        occ_path = self.occupancy_dir / f"{frame_str}_{prefix}_occupancy.npz"
        occ_occupied_pts, occ_free_pts, occ_unknown_pts = self._load_occupancy_points(occ_path)

        # 4. Load Obstacles Metadata
        obs_path = self.metadata_dir / f"{frame_str}_{prefix}_obstacles.json"
        obstacles = []
        if obs_path.exists():
            with open(obs_path, "r", encoding="utf-8") as f:
                obstacles = json.load(f)

        # 5. Load Scene Metadata
        scene_path = self.metadata_dir / f"{frame_str}_{prefix}_scene.json"
        scene_meta = {}
        if scene_path.exists():
            with open(scene_path, "r", encoding="utf-8") as f:
                scene_meta = json.load(f)

        # 6. Load 3D Photorealistic Surface Mesh (M7.1)
        mesh_tag = "ground_truth" if source == "ground_truth" else "ai_calibrated"
        mesh_path = self.mesh_dir / f"{frame_str}_{mesh_tag}_mesh.ply"
        mesh_verts, mesh_faces, mesh_cols, mesh_norms = self._load_surface_mesh(mesh_path)

        # 7. Load 3D Photorealistic Textured Mesh (M7.2)
        tex_verts, tex_faces, tex_uvs, tex_img = self._load_textured_mesh(frame_str, source)

        scene_data = SceneData(
            frame_id=frame_id,
            source=source,
            points_rgb=pts_rgb,
            colors_rgb=cols_rgb,
            points_semantic=pts_sem,
            colors_semantic=cols_sem,
            occupancy_occupied_points=occ_occupied_pts,
            occupancy_free_points=occ_free_pts,
            occupancy_unknown_points=occ_unknown_pts,
            points_confidence=conf,
            colors_confidence=cols_conf,
            obstacles=obstacles,
            scene_metadata=scene_meta,
            mesh_vertices=mesh_verts,
            mesh_faces=mesh_faces,
            mesh_colors=mesh_cols,
            mesh_normals=mesh_norms,
            textured_mesh_vertices=tex_verts,
            textured_mesh_faces=tex_faces,
            textured_mesh_uvs=tex_uvs,
            textured_mesh_texture=tex_img,
        )

        self._cache[cache_key] = scene_data
        return scene_data

    def load_scenario(self, scenario_id: str, source: str = "ai_calibrated", force_reload: bool = False) -> SceneData:
        """
        Load complete 3D scene data for one of the 10 M8 test scenarios.
        Uses in-memory cache to ensure responsive switching.
        """
        cache_key = (scenario_id, source)
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]

        scenario_dir = self.data_dir / "m8_scenarios" / scenario_id
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        # Choose PLY and occupancy based on source
        if source == "ground_truth":
            ply_path = scenario_dir / "pointcloud_gt.ply"
            occ_path = scenario_dir / "occupancy_gt.npz"
        else:
            ply_path = scenario_dir / "pointcloud_ai.ply"
            occ_path = scenario_dir / "occupancy_ai.npz"

        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud not found for {scenario_id}: {ply_path}")

        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts_rgb = np.asarray(pcd.points, dtype=np.float32)
        cols_rgb = np.asarray(pcd.colors, dtype=np.float32)
        if len(cols_rgb) == 0:
            cols_rgb = np.ones((len(pts_rgb), 3), dtype=np.float32) * 0.7

        # Confidence map
        if len(pts_rgb) > 0:
            dists = np.linalg.norm(pts_rgb, axis=1)
            conf = np.clip(1.0 - (dists / 16.0), 0.05, 1.0).astype(np.float32)
            r = np.clip(2.0 * (1.0 - conf), 0.0, 1.0)
            g = np.clip(2.0 * conf, 0.0, 1.0)
            b = np.full_like(conf, 0.1)
            cols_conf = np.column_stack([r, g, b]).astype(np.float32)
        else:
            conf = np.zeros(0, dtype=np.float32)
            cols_conf = np.zeros((0, 3), dtype=np.float32)

        # Occupancy points
        occ_occupied, occ_free, occ_unknown = self._load_occupancy_points(occ_path)

        # Obstacles
        obs_path = scenario_dir / "obstacles_ai.json"
        obstacles = []
        if obs_path.exists():
            with open(obs_path, "r", encoding="utf-8") as f:
                obstacles = json.load(f)

        # Ensure each obstacle has standard 'bbox' dictionary
        for o in obstacles:
            if "bbox" not in o:
                c = np.array(o.get("centroid", [0, 0, 0]), dtype=np.float32)
                d = np.array(o.get("dimensions", [1, 1, 1]), dtype=np.float32)
                b_min = c - d / 2.0
                b_max = c + d / 2.0
                o["bbox"] = {
                    "min_point": b_min.tolist(),
                    "max_point": b_max.tolist(),
                    "dimensions": d.tolist(),
                    "volume_m3": float(np.prod(d)),
                }

        # Metadata
        meta_path = scenario_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Surface Mesh (M7.1)
        mesh_tag = "ground_truth" if source == "ground_truth" else "ai_calibrated"
        mesh_path = self.mesh_dir / f"{scenario_id}_{mesh_tag}_mesh.ply"
        mesh_verts, mesh_faces, mesh_cols, mesh_norms = self._load_surface_mesh(mesh_path)

        scene_data = SceneData(
            frame_id=0,
            source=source,
            points_rgb=pts_rgb,
            colors_rgb=cols_rgb,
            points_semantic=pts_rgb,
            colors_semantic=cols_rgb,
            occupancy_occupied_points=occ_occupied,
            occupancy_free_points=occ_free,
            occupancy_unknown_points=occ_unknown,
            points_confidence=conf,
            colors_confidence=cols_conf,
            obstacles=obstacles,
            scene_metadata=metadata,
            mesh_vertices=mesh_verts,
            mesh_faces=mesh_faces,
            mesh_colors=mesh_cols,
            mesh_normals=mesh_norms,
            scenario_id=scenario_id,
        )

        self._cache[cache_key] = scene_data
        return scene_data

    def _load_surface_mesh(
        self, mesh_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load continuous 3D surface mesh from PLY file.
        Returns: (vertices (V, 3), faces (F, 3), colors (V, 4), normals (V, 3))
        """
        if not mesh_path.exists():
            return None, None, None, None
        try:
            from mesh.rgbd_mesher import RGBDSurfaceMesher
            return RGBDSurfaceMesher.load_mesh_ply(mesh_path)
        except Exception as e:
            print(f"Warning: Failed to load surface mesh from {mesh_path}: {e}")
            return None, None, None, None

    def _load_textured_mesh(
        self, frame_str: str, source: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load continuous 3D textured mesh and texture atlas (M7.2).
        Returns: (vertices, faces, uvs, texture_image)
        """
        mesh_tag = "ground_truth" if source == "ground_truth" else "ai_calibrated"
        ply_path = self.textured_mesh_dir / f"{frame_str}_{mesh_tag}.ply"
        uv_path = self.textured_mesh_dir / f"{frame_str}_{mesh_tag}_uvs.npy"
        tex_path = self.textured_mesh_dir / f"{frame_str}_{mesh_tag}_texture.png"

        if not (ply_path.exists() and uv_path.exists() and tex_path.exists()):
            return None, None, None, None

        try:
            from mesh.rgbd_mesher import RGBDSurfaceMesher
            verts, faces, cols, norms = RGBDSurfaceMesher.load_mesh_ply(ply_path)
            uvs = np.load(str(uv_path)).astype(np.float32)
            tex_bgr = cv2.imread(str(tex_path))
            tex_rgb = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB)
            return verts, faces, uvs, tex_rgb
        except Exception as e:
            print(f"Warning: Failed to load textured mesh ({frame_str}, {source}): {e}")
            return None, None, None, None

    def _load_occupancy_points(self, occ_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract 3D world coordinates of occupied, free, and unknown voxels from .npz.
        Subsamples free and unknown space for smooth rendering performance.
        """
        if not occ_path.exists():
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        data = np.load(occ_path)
        grid = data["grid"]  # 0=UNKNOWN, 1=FREE, 2=OCCUPIED
        bounds_x = data["bounds_x"]
        bounds_y = data["bounds_y"]
        bounds_z = data["bounds_z"]
        res = float(data["resolution"])

        # Occupied points: state == 2
        occ_idx = np.argwhere(grid == 2)
        if len(occ_idx) > 0:
            occ_x = occ_idx[:, 0] * res + bounds_x[0] + 0.5 * res
            occ_y = occ_idx[:, 1] * res + bounds_y[0] + 0.5 * res
            occ_z = occ_idx[:, 2] * res + bounds_z[0] + 0.5 * res
            occ_pts = np.column_stack([occ_x, occ_y, occ_z]).astype(np.float32)
        else:
            occ_pts = np.zeros((0, 3), dtype=np.float32)

        # Free space: state == 1 (subsampled)
        free_idx = np.argwhere(grid == 1)
        if len(free_idx) > 0:
            step = max(1, len(free_idx) // 15000)
            sampled_free = free_idx[::step]
            free_x = sampled_free[:, 0] * res + bounds_x[0] + 0.5 * res
            free_y = sampled_free[:, 1] * res + bounds_y[0] + 0.5 * res
            free_z = sampled_free[:, 2] * res + bounds_z[0] + 0.5 * res
            free_pts = np.column_stack([free_x, free_y, free_z]).astype(np.float32)
        else:
            free_pts = np.zeros((0, 3), dtype=np.float32)

        # Unknown space: state == 0 (subsampled)
        unk_idx = np.argwhere(grid == 0)
        if len(unk_idx) > 0:
            step = max(1, len(unk_idx) // 10000)
            sampled_unk = unk_idx[::step]
            unk_x = sampled_unk[:, 0] * res + bounds_x[0] + 0.5 * res
            unk_y = sampled_unk[:, 1] * res + bounds_y[0] + 0.5 * res
            unk_z = sampled_unk[:, 2] * res + bounds_z[0] + 0.5 * res
            unk_pts = np.column_stack([unk_x, unk_y, unk_z]).astype(np.float32)
        else:
            unk_pts = np.zeros((0, 3), dtype=np.float32)

        return occ_pts, free_pts, unk_pts
