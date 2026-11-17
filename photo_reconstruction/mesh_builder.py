"""
photo_reconstruction/mesh_builder.py — Multi-Radius Ball-Pivoting and Poisson Surface Reconstruction
====================================================================================================

Reconstructs high-fidelity 3D surface meshes from dense oriented point clouds using:
  1. Multi-Radius Ball-Pivoting Algorithm (BPA):
     - Directly triangulates authentic point samples without artificial volume inflation.
     - Preserves sharp architectural steps, carved scrollwork, and projecting basin rims.
     - Adapts ball radii based on measured nearest-neighbor distance distribution.
  2. Screened Poisson Surface Reconstruction with Taubin Anti-Ripple Smoothing:
     - Watertight continuous surface with non-shrinking Taubin smoothing.
  3. Hybrid Region-Aware Surface Reconstruction:
     - Plane-aware planar meshing for background brick wall and pavement.
     - Fine-radius BPA for carved sculptural relief and projecting basin.
  4. Fragment Suppression:
     - Prunes small disconnected floating triangle clusters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import open3d as o3d


class MeshBuilder:
    """3D surface mesh reconstructor with multi-radius BPA and region-aware meshing."""

    def __init__(
        self,
        method: str = "bpa",
        poisson_depth: int = 8,
        density_quantile: float = 0.05,
        taubin_iterations: int = 8,
        min_cluster_faces: int = 40,
        bpa_radii_multipliers: Tuple[float, float, float] = (1.2, 2.5, 5.0),
    ):
        """
        Parameters
        ----------
        method : str ('bpa', 'poisson', 'hybrid')
        poisson_depth : int (octree depth for Poisson: 7-9)
        density_quantile : float (removes low-density bounding box artifacts)
        taubin_iterations : int (anti-ripple non-shrinking surface smoothing)
        min_cluster_faces : int (minimum connected triangle cluster size)
        bpa_radii_multipliers : Tuple[float, float, float]
        """
        self.method = method.lower()
        self.poisson_depth = int(poisson_depth)
        self.density_quantile = float(density_quantile)
        self.taubin_iterations = int(taubin_iterations)
        self.min_cluster_faces = int(min_cluster_faces)
        self.bpa_multipliers = bpa_radii_multipliers

    @staticmethod
    def _query_nearest_neighbors(
        tree: Any,
        query_points: np.ndarray,
        *,
        chunk_size: int = 200_000,
        workers: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query a cKDTree in bounded batches instead of looping over vertices.

        Chunking bounds temporary memory for high-detail Poisson meshes while
        keeping the nearest-neighbour search inside SciPy's compiled backend.
        """
        from scipy.spatial import cKDTree

        if not isinstance(tree, cKDTree):
            raise TypeError("tree must be a scipy.spatial.cKDTree instance.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least one.")
        if workers < 1:
            raise ValueError("workers must be at least one.")

        queries = np.asarray(query_points, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("query_points must have shape (N, 3).")

        distances = np.empty(len(queries), dtype=np.float64)
        indices = np.empty(len(queries), dtype=np.intp)
        for start in range(0, len(queries), chunk_size):
            stop = min(start + chunk_size, len(queries))
            batch_distances, batch_indices = tree.query(
                queries[start:stop], k=1, workers=workers
            )
            distances[start:stop] = batch_distances
            indices[start:stop] = batch_indices

        return distances, indices

    def build_mesh(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct a 3D surface mesh from points, colors, and normals.

        Returns
        -------
        vertices : (V, 3) float32
        faces : (F, 3) uint32
        vertex_colors : (V, 4) float32 RGBA
        normals : (V, 3) float32
        """
        if len(points) < 15:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        pcd.remove_non_finite_points()
        pcd.normalize_normals()

        # Compute point spacing statistics
        nn_dists = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.04

        mesh = None

        # ---------------------------------------------------------------------
        # 1. Multi-Radius Ball-Pivoting Algorithm (BPA)
        # ---------------------------------------------------------------------
        if self.method in ("bpa", "ball_pivoting"):
            radii = [
                float(avg_dist * self.bpa_multipliers[0]),
                float(avg_dist * self.bpa_multipliers[1]),
                float(avg_dist * self.bpa_multipliers[2]),
            ]
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
            except Exception:
                mesh = None

        # ---------------------------------------------------------------------
        # 2. Hybrid Region-Aware Meshing
        # ---------------------------------------------------------------------
        elif self.method == "hybrid":
            # Segment vertical background wall plane
            try:
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=1000)
                if len(inliers) >= max(50, int(len(points) * 0.15)):
                    # Wall points vs relief points
                    wall_pcd = pcd.select_by_index(inliers)
                    relief_pcd = pcd.select_by_index(inliers, invert=True)

                    # Wall meshing with slightly larger radii for planar continuity
                    wall_radii = [avg_dist * 1.5, avg_dist * 3.5, avg_dist * 7.0]
                    wall_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        wall_pcd, o3d.utility.DoubleVector(wall_radii)
                    )

                    # Fine BPA for carved relief, basin, and sculpture
                    relief_radii = [avg_dist * 1.0, avg_dist * 2.2, avg_dist * 4.5]
                    relief_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        relief_pcd, o3d.utility.DoubleVector(relief_radii)
                    )

                    # Combine meshes
                    mesh = wall_mesh + relief_mesh
            except Exception:
                mesh = None

        # ---------------------------------------------------------------------
        # 3. Screened Poisson Surface Reconstruction (Fallback or Preset)
        # ---------------------------------------------------------------------
        if mesh is None or len(mesh.vertices) == 0:
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth, n_threads=1
                )
                dens = np.asarray(densities)
                if len(dens) > 0:
                    thresh = np.quantile(dens, max(0.10, self.density_quantile))
                    mesh.remove_vertices_by_mask(dens < thresh)

                if len(mesh.vertices) > 0 and len(pcd.points) > 10:
                    from scipy.spatial import cKDTree

                    point_xyz = np.asarray(pcd.points, dtype=np.float64)
                    point_tree = cKDTree(point_xyz)
                    mesh_v = np.asarray(mesh.vertices, dtype=np.float64)
                    max_allowed_dist = max(0.08, avg_dist * 3.5)
                    nearest_distances, _ = self._query_nearest_neighbors(
                        point_tree, mesh_v
                    )
                    mesh.remove_vertices_by_mask(nearest_distances > max_allowed_dist)

                if self.taubin_iterations > 0 and len(mesh.vertices) > 100:
                    mesh = mesh.filter_smooth_taubin(number_of_iterations=self.taubin_iterations)
            except Exception:
                mesh = None

        # Final Fallback to standard BPA
        if mesh is None or len(mesh.vertices) == 0:
            radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        # ---------------------------------------------------------------------
        # 4. Mesh Cleaning & Fragment Pruning
        # ---------------------------------------------------------------------
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Connected cluster filtering: prune disconnected tiny floating triangles
        if self.min_cluster_faces > 0 and len(mesh.triangles) > self.min_cluster_faces * 3:
            try:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                    mesh.cluster_connected_triangles()
                )
                tri_clusters = np.asarray(triangle_clusters)
                n_tris = np.asarray(cluster_n_triangles)

                # Keep clusters with at least min_cluster_faces
                keep_clusters = np.where(n_tris >= self.min_cluster_faces)[0]
                tri_mask = np.isin(tri_clusters, keep_clusters)
                mesh.remove_triangles_by_mask(~tri_mask)
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass

        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        mesh_cols = np.asarray(mesh.vertex_colors, dtype=np.float32)
        mesh_norms = np.asarray(mesh.vertex_normals, dtype=np.float32)

        # Transfer authentic photographic point colors
        if len(verts) > 0 and (len(mesh_cols) == 0 or len(mesh_cols) != len(verts) or np.all(mesh_cols == 0.7)):
            if len(pcd.points) > 0 and len(pcd.colors) == len(pcd.points):
                from scipy.spatial import cKDTree

                point_xyz = np.asarray(pcd.points, dtype=np.float64)
                if point_tree is None:
                    point_tree = cKDTree(point_xyz)
                pcd_colors = np.asarray(pcd.colors, dtype=np.float32)
                _, nearest_indices = self._query_nearest_neighbors(
                    point_tree, verts
                )
                mesh_cols = pcd_colors[nearest_indices]
            else:
                mesh_cols = np.ones((len(verts), 3), dtype=np.float32) * 0.7

        # Ensure outward face winding and filter sliver artifacts (CLEAR360 M7.3.4)
        from photo_reconstruction.normal_orientation import NormalOrientationEngine
        norm_engine = NormalOrientationEngine()
        verts, faces, _ = norm_engine.filter_triangular_artifacts(verts, faces)
        if len(faces) > 0:
            cam_c = np.array([0.0, 0.0, 2.1])
            faces, mesh_norms, _ = norm_engine.orient_faces_towards_cameras(verts, faces, cam_c)

        alpha = np.ones((len(mesh_cols), 1), dtype=np.float32)
        if len(alpha) != len(verts):
            alpha = np.ones((len(verts), 1), dtype=np.float32)
            mesh_cols = np.ones((len(verts), 3), dtype=np.float32) * 0.7
        rgba_cols = np.hstack([mesh_cols, alpha])

        return verts, faces, rgba_cols, mesh_norms

    @staticmethod
    def save_mesh_obj(
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        out_path: Path,
        texture_img: Optional[np.ndarray] = None,
        uvs: Optional[np.ndarray] = None,
    ) -> Path:
        """Export reconstructed surface mesh to Wavefront OBJ format."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        obj_file = out if out.suffix.lower() == ".obj" else out.with_suffix(".obj")
        mtl_file = obj_file.with_suffix(".mtl")
        png_file = obj_file.with_suffix(".png")

        has_texture = texture_img is not None and uvs is not None and len(uvs) == len(vertices)

        if has_texture:
            cv2.imwrite(str(png_file), cv2.cvtColor(texture_img, cv2.COLOR_RGB2BGR))
            with open(mtl_file, "w") as f:
                f.write(f"# CLEAR360 Photogrammetric Material\n")
                f.write(f"newmtl material_0\n")
                f.write("Ka 1.000 1.000 1.000\n")
                f.write("Kd 1.000 1.000 1.000\n")
                f.write("Ks 0.000 0.000 0.000\n")
                f.write("d 1.0\n")
                f.write("illum 1\n")
                f.write(f"map_Kd {png_file.name}\n")

        with open(obj_file, "w") as f:
            f.write("# CLEAR360 3D Mesh Output\n")
            if has_texture:
                f.write(f"mtllib {mtl_file.name}\n")
                f.write("usemtl material_0\n")

            for i, v in enumerate(vertices):
                if not has_texture and colors is not None and len(colors) == len(vertices):
                    c = colors[i][:3]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if has_texture:
                for uv in uvs:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            for tri in faces:
                if has_texture:
                    f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
                else:
                    f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        return obj_file

    @staticmethod
    def save_mesh_ply(
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        out_path: Path,
        normals: Optional[np.ndarray] = None,
    ) -> Path:
        """Export reconstructed surface mesh to PLY format with vertex colors."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        if colors.shape[1] == 4:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64))
        else:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if normals is not None and len(normals) == len(vertices):
            mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        else:
            mesh_o3d.compute_vertex_normals()

        o3d.io.write_triangle_mesh(str(out), mesh_o3d, write_vertex_colors=True, write_vertex_normals=True)
        return out
