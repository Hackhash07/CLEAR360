"""
photo_reconstruction/mesh_builder.py — Poisson and Ball-Pivoting Surface Mesh Generation
========================================================================================

Converts dense oriented point clouds into continuous 3D surface meshes using Open3D's
Screened Poisson Surface Reconstruction and Ball-Pivoting algorithm.
Generates UV coordinates and exports universal Wavefront OBJ + MTL and PLY files.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import open3d as o3d


class MeshBuilder:
    """3D surface mesh reconstructor from dense oriented point clouds."""

    def __init__(self, method: str = "poisson", poisson_depth: int = 8, density_quantile: float = 0.05):
        """
        Parameters
        ----------
        method : str ('poisson', 'bpa')
        poisson_depth : int (octree depth for Poisson reconstruction: 7-9)
        density_quantile : float (removes low-density bounding box artifacts)
        """
        self.method = method.lower()
        self.poisson_depth = int(poisson_depth)
        self.density_quantile = float(density_quantile)

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
        if len(points) < 10:
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

        if self.method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth
            )
            # 1. Filter low density vertices to remove outer boundary envelope
            dens = np.asarray(densities)
            if len(dens) > 0:
                thresh = np.quantile(dens, max(0.12, self.density_quantile))
                mesh.remove_vertices_by_mask(dens < thresh)

            # 2. Prune phantom Poisson bubble extrapolation far from actual point cloud
            if len(mesh.vertices) > 0 and len(pcd.points) > 10:
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                mesh_v = np.asarray(mesh.vertices)
                nn_dist = float(np.mean(pcd.compute_nearest_neighbor_distance()))
                max_allowed_dist = max(0.08, nn_dist * 3.5)

                remove_phantom = np.zeros(len(mesh_v), dtype=bool)
                for i, v in enumerate(mesh_v):
                    [k, idx, dist_sq] = pcd_tree.search_knn_vector_3d(v, 1)
                    if k == 0 or np.sqrt(dist_sq[0]) > max_allowed_dist:
                        remove_phantom[i] = True
                mesh.remove_vertices_by_mask(remove_phantom)
        else:
            # Ball-Pivoting Algorithm
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(distances))
            radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )

        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        mesh_cols = np.asarray(mesh.vertex_colors, dtype=np.float32)
        mesh_norms = np.asarray(mesh.vertex_normals, dtype=np.float32)

        if len(mesh_cols) == 0:
            mesh_cols = np.ones((len(verts), 3), dtype=np.float32) * 0.7

        alpha = np.ones((len(mesh_cols), 1), dtype=np.float32)
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

        has_tex = texture_img is not None and uvs is not None and len(uvs) == len(vertices)

        if has_tex:
            tex_file = obj_file.parent / f"{obj_file.stem}_texture.png"
            cv2.imwrite(str(tex_file), cv2.cvtColor(texture_img, cv2.COLOR_RGB2BGR))

            with open(mtl_file, "w") as fm:
                fm.write(f"newmtl Material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nmap_Kd {tex_file.name}\n")

        with open(obj_file, "w") as fo:
            fo.write("# CLEAR360 Phone Photo Reconstruction\n")
            if has_tex:
                fo.write(f"mtllib {mtl_file.name}\n")

            for v in vertices:
                fo.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            if has_tex:
                for uv in uvs:
                    fo.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                fo.write("usemtl Material_0\n")

            for f in faces:
                i1, i2, i3 = f[0] + 1, f[1] + 1, f[2] + 1
                if has_tex:
                    fo.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")
                else:
                    fo.write(f"f {i1} {i2} {i3}\n")

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

