"""
photo_reconstruction/quality_report.py — Structured Reconstruction Quality Reporting
=====================================================================================

Generates comprehensive photogrammetry reports documenting camera registration rates,
reprojection errors, point-cloud and mesh densities, scale calibration provenance, and
failure diagnostics for any unaligned photographs.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time


def classify_reconstruction_grade(
    uploaded_count: int,
    registered_count: int,
    reprojection_error_px: Optional[float],
    *,
    mesh_quality: Optional[Dict[str, Any]] = None,
    camera_overlap: Optional[Dict[str, Any]] = None,
    photo_set: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify overall reconstruction evidence without confusing export with quality."""
    completeness_pct = 100.0 * registered_count / max(1, uploaded_count)
    mesh_status = (mesh_quality or {}).get("status")
    overlap_available = bool((camera_overlap or {}).get("available", False))
    overlap_status = (camera_overlap or {}).get("status")
    photo_set_status = (photo_set or {}).get("status")

    mesh_allows_optimal = mesh_quality is None or mesh_status == "healthy"
    overlap_allows_optimal = camera_overlap is None or (
        overlap_available and overlap_status == "connected"
    )
    photo_set_allows_optimal = photo_set is None or photo_set_status == "ready"

    has_optimal_registration = (
        uploaded_count > 0
        and registered_count >= uploaded_count
        and reprojection_error_px is not None
        and reprojection_error_px < 1.5
    )
    if registered_count < 2 or mesh_status == "poor":
        return "FAILED"
    if (
        has_optimal_registration
        and mesh_allows_optimal
        and overlap_allows_optimal
        and photo_set_allows_optimal
    ):
        return "OPTIMAL"
    if registered_count >= 2 and completeness_pct >= 60.0:
        return "ACCEPTABLE"
    if registered_count >= 2:
        return "PARTIAL"
    return "FAILED"


class QualityReportGenerator:
    """Compiles and exports standard photogrammetry quality reports."""

    @classmethod
    def generate_report(
        cls,
        session_id: str,
        uploaded_count: int,
        registered_count: int,
        failed_images: List[str],
        sparse_point_count: int,
        dense_point_count: int,
        mesh_triangle_count: int,
        reprojection_error_px: Optional[float],
        processing_time_s: float,
        scale_info: Dict[str, Any],
        image_quality_summary: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        quality_preset: Optional[str] = None,
        mesh_quality: Optional[Dict[str, Any]] = None,
        camera_overlap: Optional[Dict[str, Any]] = None,
        photo_set: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble standardized JSON photogrammetry report.
        """
        completeness_pct = round((registered_count / max(1, uploaded_count)) * 100.0, 1)
        recon_grade = classify_reconstruction_grade(
            uploaded_count,
            registered_count,
            reprojection_error_px,
            mesh_quality=mesh_quality,
            camera_overlap=camera_overlap,
            photo_set=photo_set,
        )

        report = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quality_preset": quality_preset or "standard",
            "reconstruction_grade": recon_grade,
            "completeness_percentage": completeness_pct,
            "image_counts": {
                "uploaded": uploaded_count,
                "registered": registered_count,
                "failed": len(failed_images),
            },
            "failed_images": failed_images,
            "geometric_metrics": {
                "sparse_points": sparse_point_count,
                "dense_points": dense_point_count,
                "mesh_triangles": mesh_triangle_count,
                "mean_reprojection_error_px": (
                    round(reprojection_error_px, 3)
                    if reprojection_error_px is not None else None
                ),
            },
            "scale_calibration": scale_info,
            "performance": {
                "total_processing_time_s": round(processing_time_s, 2),
            },
            "image_quality_summary": image_quality_summary or {},
            "mesh_quality": mesh_quality or {},
            "camera_overlap": camera_overlap or {},
            "photo_set": photo_set or {},
            "diagnostics": {
                "photo_set_warnings": (
                    photo_set.get("warnings", []) if photo_set else []
                ),
                "camera_overlap_warnings": (
                    camera_overlap.get("warnings", []) if camera_overlap else []
                ),
                "has_failures": len(failed_images) > 0,
                "failure_reasons": (
                    [f"Image '{name}' had insufficient overlapping feature matches with registered views." for name in failed_images]
                    if failed_images else []
                ),
                "guidance": (
                    "Capture images with 60-80% visual overlap and ensure adequate lighting and sharp focus."
                    if len(failed_images) > 0 else "Nominal registration achieved across all views."
                ),
            },
            "notes": notes or "Automated CLEAR360 Smartphone Photogrammetry Reconstruction.",
        }

        return report

    @classmethod
    def save_report_json(cls, report: Dict[str, Any], out_path: Path) -> Path:
        """Write JSON report to disk."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return p

