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

from photo_reconstruction.mesh_diagnostics import MeshQualityAuditor


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
        audit = MeshQualityAuditor.audit_mesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            expected_closed=False,
        ).to_dict()

        # The model may not have a known metric scale. Rename dimensional values
        # so consumers cannot mistake arbitrary SfM units for metres.
        audit["min_edge_length_coordinate_units"] = audit.pop("min_edge_length_m")
        audit["max_edge_length_coordinate_units"] = audit.pop("max_edge_length_m")
        audit["median_edge_length_coordinate_units"] = audit.pop("median_edge_length_m")
        audit["surface_area_coordinate_units_squared"] = audit.pop("total_surface_area_m2")
        audit["coordinate_scale"] = "un-calibrated SfM units unless scale calibration is supplied"

        geometry = audit
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
                f"The mesh contains {geometry['connected_components']} disconnected components. "
                "This may be valid for multi-part scenes; inspect small floating fragments before cleanup."
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
