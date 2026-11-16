"""Backend-independent quality assurance for CLEAR360 reconstruction outputs.

This module audits generated geometry and export dependencies. It deliberately does
not require meshes to be watertight: photogrammetry commonly reconstructs open
surfaces such as walls, terrain, furniture and incomplete object backsides.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components



def _load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh file without silently repairing or dropping disconnected parts."""
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not geometries:
            raise ValueError(f"No triangle geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh data returned for {path}: {type(loaded).__name__}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Mesh contains non-finite vertex coordinates: {path}")
    if int(mesh.faces.min()) < 0 or int(mesh.faces.max()) >= len(mesh.vertices):
        raise ValueError(f"Mesh contains invalid face indices: {path}")
    return mesh


def _audit_geometry(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Calculate topology and triangle metrics without requiring Open3D."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count, face_count = len(vertices), len(faces)

    sorted_edges = np.sort(mesh.edges, axis=1)
    _, edge_counts = np.unique(sorted_edges, axis=0, return_counts=True)
    non_manifold_edges = int(np.sum(edge_counts > 2))
    boundary_edges = int(np.sum(edge_counts == 1))
    interior_edges = int(np.sum(edge_counts == 2))

    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    a = np.linalg.norm(v1 - v0, axis=1)
    b = np.linalg.norm(v2 - v1, axis=1)
    c = np.linalg.norm(v0 - v2, axis=1)
    semiperimeter = (a + b + c) / 2.0
    area_argument = np.maximum(
        0.0,
        semiperimeter
        * (semiperimeter - a)
        * (semiperimeter - b)
        * (semiperimeter - c),
    )
    areas = np.sqrt(area_argument)
    degenerate_faces = int(np.sum(areas <= 1e-8))

    denominator = (
        8.0
        * (semiperimeter - a)
        * (semiperimeter - b)
        * (semiperimeter - c)
    )
    aspect_ratios = np.where(
        denominator > 1e-12,
        (a * b * c) / (denominator + 1e-12),
        100.0,
    )
    sliver_faces = int(np.sum(aspect_ratios > 20.0))

    unique_edge_lengths = mesh.edges_unique_length
    face_adjacency = mesh.face_adjacency
    if face_count == 0:
        component_count = 0
        component_face_counts = np.zeros(0, dtype=np.int64)
    elif len(face_adjacency) == 0:
        # A single isolated triangle is one component; multiple isolated faces
        # are separate components even though the adjacency edge list is empty.
        component_count = face_count
        component_face_counts = np.ones(face_count, dtype=np.int64)
    else:
        rows = np.concatenate((face_adjacency[:, 0], face_adjacency[:, 1]))
        cols = np.concatenate((face_adjacency[:, 1], face_adjacency[:, 0]))
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(face_count, face_count),
        ).tocsr()
        component_count, component_labels = connected_components(
            adjacency, directed=False, return_labels=True
        )
        component_face_counts = np.bincount(
            component_labels, minlength=component_count
        )

    largest_component_ratio = (
        float(component_face_counts.max() / face_count) if face_count else 0.0
    )
    small_component_threshold = max(1, int(np.ceil(face_count * 0.001)))
    small_component_mask = component_face_counts < small_component_threshold
    small_component_count = int(np.sum(small_component_mask))
    small_component_face_ratio = (
        float(component_face_counts[small_component_mask].sum() / face_count)
        if face_count else 0.0
    )

    is_watertight = bool(mesh.is_watertight)
    boundary_edge_count = boundary_edges
    surface_type = (
        "closed_volume"
        if boundary_edge_count == 0 and is_watertight
        else "open_surface"
    )

    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "is_watertight": is_watertight,
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "surface_type": surface_type,
        "connected_components": int(component_count),
        "largest_component_face_ratio": largest_component_ratio,
        "small_component_count_below_0_1pct": small_component_count,
        "small_component_face_ratio": small_component_face_ratio,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "manifold_interior_edges": interior_edges,
        "degenerate_faces": degenerate_faces,
        "sliver_faces": sliver_faces,
        "sliver_ratio": float(sliver_faces / max(1, face_count)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "median_aspect_ratio": float(np.median(aspect_ratios)),
        "min_edge_length_coordinate_units": (
            float(unique_edge_lengths.min()) if len(unique_edge_lengths) else 0.0
        ),
        "max_edge_length_coordinate_units": (
            float(unique_edge_lengths.max()) if len(unique_edge_lengths) else 0.0
        ),
        "median_edge_length_coordinate_units": (
            float(np.median(unique_edge_lengths)) if len(unique_edge_lengths) else 0.0
        ),
        "surface_area_coordinate_units_squared": float(np.sum(areas)),
        "bounding_box_extent": (
            vertices.max(axis=0) - vertices.min(axis=0)
        ).round(4).tolist(),
        "coordinate_scale": "un-calibrated SfM units unless scale calibration is supplied",
    }


def audit_obj_materials(obj_path: str | Path) -> dict[str, Any]:
    """Check OBJ -> MTL -> diffuse-map references using paths relative to each MTL."""
    obj_file = Path(obj_path).expanduser().resolve()
    result: dict[str, Any] = {
        "obj_exists": obj_file.is_file(),
        "material_libraries": [],
        "diffuse_maps": [],
        "missing_material_libraries": [],
        "missing_diffuse_maps": [],
        "all_references_resolve": False,
    }
    if not obj_file.is_file():
        result["missing_material_libraries"].append(str(obj_file))
        return result

    material_names: list[str] = []
    for raw_line in obj_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped, comments=True)
        except ValueError:
            continue
        if tokens and tokens[0].lower() == "mtllib":
            material_names.extend(tokens[1:])

    for material_name in material_names:
        mtl_file = (obj_file.parent / material_name).resolve()
        result["material_libraries"].append(str(mtl_file))
        if not mtl_file.is_file():
            result["missing_material_libraries"].append(str(mtl_file))
            continue

        for raw_line in mtl_file.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                tokens = shlex.split(stripped, comments=True)
            except ValueError:
                continue
            if not tokens or tokens[0].lower() != "map_kd":
                continue
            if len(tokens) < 2:
                result["missing_diffuse_maps"].append(
                    f"{mtl_file.name}: map_Kd without a filename"
                )
                continue

            # Wavefront MTL map options (such as -s and -o) precede the filename.
            texture_path = Path(tokens[-1]).expanduser()
            if not texture_path.is_absolute():
                texture_path = mtl_file.parent / texture_path
            texture_path = texture_path.resolve()
            result["diffuse_maps"].append(str(texture_path))
            if not texture_path.is_file():
                result["missing_diffuse_maps"].append(str(texture_path))

    result["all_references_resolve"] = (
        bool(material_names)
        and not result["missing_material_libraries"]
        and not result["missing_diffuse_maps"]
        and bool(result["diffuse_maps"])
    )
    return result


def _texture_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "error": str(exc)}
    total = int(data.get("total_faces", 0) or 0)
    textured = int(data.get("textured_faces", 0) or 0)
    coverage = data.get("coverage_percentage")
    if coverage is None and total > 0:
        coverage = 100.0 * textured / total
    return {
        "available": True,
        "total_faces": total,
        "textured_faces": textured,
        "untextured_faces": int(data.get("untextured_faces", max(0, total - textured)) or 0),
        "coverage_percentage": round(float(coverage), 2) if coverage is not None else None,
        "atlas_width": data.get("atlas_width"),
        "atlas_height": data.get("atlas_height"),
    }


def audit_reconstruction_outputs(
    mesh_path: str | Path,
    *,
    obj_path: str | Path | None = None,
    glb_path: str | Path | None = None,
    texturing_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a fail-soft, backend-independent geometry and export QA report.

    The function reports issues but never deletes components or rejects an open
    surface automatically. Separate scene objects can legitimately be disconnected.
    """
    mesh_file = Path(mesh_path).expanduser().resolve()
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    textures: dict[str, Any] = {}
    geometry: dict[str, Any] | None = None

    try:
        mesh = _load_triangle_mesh(mesh_file)
        geometry = _audit_geometry(mesh)
        checks["mesh_file_readable"] = True
        checks["vertices_finite"] = True
        checks["faces_valid"] = True
    except Exception as exc:
        checks["mesh_file_readable"] = False
        warnings.append(f"Geometry audit could not read the mesh: {exc}")

    if obj_path is not None:
        obj_materials = audit_obj_materials(obj_path)
        textures["obj_materials"] = obj_materials
        checks["obj_material_references_resolve"] = obj_materials["all_references_resolve"]
        if not obj_materials["all_references_resolve"]:
            warnings.append("OBJ material or diffuse-texture references are missing or unresolved.")

    if glb_path is not None:
        glb_file = Path(glb_path).expanduser().resolve()
        glb_result: dict[str, Any] = {
            "exists": glb_file.is_file(),
            "size_bytes": glb_file.stat().st_size if glb_file.is_file() else 0,
            "loadable": False,
        }
        if glb_file.is_file():
            try:
                glb_scene = trimesh.load(glb_file, force="scene", process=False)
                glb_result["loadable"] = (
                    isinstance(glb_scene, trimesh.Scene) and bool(glb_scene.geometry)
                ) or (
                    isinstance(glb_scene, trimesh.Trimesh) and len(glb_scene.faces) > 0
                )
            except Exception as exc:
                glb_result["error"] = str(exc)
        textures["glb"] = glb_result
        checks["glb_loadable"] = glb_result["loadable"]
        if not glb_result["loadable"]:
            warnings.append("GLB export is missing or could not be loaded as a 3D scene.")

    texture_summary = _texture_report(
        Path(texturing_report_path).expanduser().resolve()
        if texturing_report_path is not None else None
    )
    if texture_summary is not None:
        textures["coverage_report"] = texture_summary
        if texture_summary.get("coverage_percentage") is not None:
            coverage = texture_summary["coverage_percentage"]
            checks["texture_coverage_percentage"] = coverage
            if coverage < 70.0:
                warnings.append(
                    f"Photographic texture coverage is {coverage:.1f}%; inspect untextured surfaces."
                )

    score = 0.0
    if geometry is not None:
        face_count = max(1, int(geometry["face_count"]))
        edge_count = max(
            1,
            int(geometry["non_manifold_edges"])
            + int(geometry["boundary_edges"])
            + int(geometry["manifold_interior_edges"]),
        )
        degenerate_ratio = int(geometry["degenerate_faces"]) / face_count
        non_manifold_ratio = int(geometry["non_manifold_edges"]) / edge_count
        sliver_ratio = float(geometry["sliver_ratio"])

        penalties = {
            "degenerate_faces": min(30.0, degenerate_ratio * 1000.0),
            "sliver_faces": min(25.0, sliver_ratio * 100.0),
            "non_manifold_edges": min(25.0, non_manifold_ratio * 250.0),
            "inconsistent_winding": 10.0 if not geometry["is_winding_consistent"] else 0.0,
            # Several disconnected components may be valid; flag for review rather
            # than applying a hard rejection or deleting them automatically.
            "disconnected_components": (
                min(10.0, 2.0 * np.log2(max(1, int(geometry["connected_components"]))))
                if int(geometry["connected_components"]) > 1 else 0.0
            ),
            "small_component_faces": min(
                15.0, float(geometry["small_component_face_ratio"]) * 100.0
            ),
        }
        score = max(0.0, 100.0 - sum(penalties.values()))
        geometry["derived_ratios"] = {
            "degenerate_face_ratio": round(degenerate_ratio, 6),
            "non_manifold_edge_ratio": round(non_manifold_ratio, 6),
        }
        geometry["quality_penalties"] = {
            key: round(float(value), 2) for key, value in penalties.items()
        }

        if degenerate_ratio > 0:
            warnings.append(f"{geometry['degenerate_faces']} degenerate faces were detected.")
        if sliver_ratio > 0.05:
            warnings.append(f"{sliver_ratio * 100:.1f}% of faces are classified as sliver triangles.")
        if int(geometry["non_manifold_edges"]) > 0:
            warnings.append(
                f"{geometry['non_manifold_edges']} non-manifold edges were detected."
            )
        if int(geometry["connected_components"]) > 1:
            warnings.append(
                f"The mesh contains {geometry['connected_components']} disconnected components "
                f"(largest component covers {geometry['largest_component_face_ratio'] * 100:.1f}% of faces). "
                "This may be valid for multi-part scenes; inspect small floating fragments before cleanup."
            )
        if geometry["small_component_face_ratio"] > 0.01:
            warnings.append(
                f"{geometry['small_component_face_ratio'] * 100:.1f}% of faces belong to components "
                "smaller than 0.1% of the mesh; inspect for floating fragments."
            )
        if not geometry["is_winding_consistent"]:
            warnings.append("Face winding is inconsistent.")
        if geometry["surface_type"] == "open_surface":
            checks["open_surface_allowed"] = True
    else:
        checks["geometry_quality_available"] = False

    if geometry is None:
        geometry_status = "unavailable"
        status = "unavailable"
    else:
        geometry_status = (
            "healthy" if score >= 90.0 else "review" if score >= 70.0 else "poor"
        )
        export_review_required = (
            checks.get("obj_material_references_resolve") is False
            or checks.get("glb_loadable") is False
            or (
                checks.get("texture_coverage_percentage") is not None
                and checks["texture_coverage_percentage"] < 70.0
            )
        )
        status = (
            "poor" if geometry_status == "poor"
            else "review" if geometry_status == "review" or export_review_required
            else "healthy"
        )

    return {
        "schema_version": 1,
        "status": status,
        "geometry_status": geometry_status,
        "score": round(float(score), 1),
        "mesh_path": str(mesh_file),
        "geometry": geometry,
        "textures": textures,
        "checks": checks,
        "warnings": warnings,
        "policy": {
            "watertight_mesh_required": False,
            "automatic_component_deletion": False,
            "quality_score_is_heuristic": True,
            "coordinate_units_require_scale_calibration": True,
        },
    }
