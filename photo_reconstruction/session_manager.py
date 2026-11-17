"""
photo_reconstruction/session_manager.py — Photogrammetry Session Storage and Versioning
========================================================================================

Manages persistent session folders under data/phone_reconstruction/session_XXX/ ensuring
photographs, poses, point clouds, meshes, and quality reports are preserved across runs.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import json


class SessionManager:
    """Manages versioned photogrammetry reconstruction sessions."""

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent / "data" / "phone_reconstruction"
        else:
            p = Path(root_dir)
            if p.name != "phone_reconstruction":
                self.root_dir = p / "phone_reconstruction"
            else:
                self.root_dir = p
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: Optional[str] = None) -> Path:
        """
        Create a new sequential session directory with subfolders.

        Subfolders:
          - original_images/
          - exif_metadata/
          - camera_poses/
          - sparse_cloud/
          - dense_cloud/
          - mesh/
        """
        if session_id is None:
            existing = self.list_sessions()
            next_idx = len(existing) + 1
            session_id = f"session_{next_idx:03d}"

        session_dir = self.root_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        (session_dir / "original_images").mkdir(exist_ok=True)
        (session_dir / "exif_metadata").mkdir(exist_ok=True)
        (session_dir / "camera_poses").mkdir(exist_ok=True)
        (session_dir / "sparse_cloud").mkdir(exist_ok=True)
        (session_dir / "dense_cloud").mkdir(exist_ok=True)
        (session_dir / "mesh").mkdir(exist_ok=True)
        (session_dir / "openmvs").mkdir(exist_ok=True)

        return session_dir

    def list_sessions(self) -> List[str]:
        """Return list of existing session directory names."""
        if not self.root_dir.exists():
            return []
        sessions = [
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (
                d.name.startswith("session_") or (d / "mesh" / "scene_mesh.ply").exists()
            )
        ]
        return sorted(sessions)

    def get_session_dir(self, session_id: str) -> Optional[Path]:
        """Get path to a specific session directory."""
        p = self.root_dir / session_id
        return p if p.is_dir() else None

    def get_session_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load report.json from a session."""
        s_dir = self.get_session_dir(session_id)
        if not s_dir:
            return None
        report_file = s_dir / "report.json"
        if not report_file.exists():
            return None
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def cleanup_session_intermediates(self, session_id: str) -> Dict[str, Any]:
        """Remove only known bulky intermediates after an OpenMVS session is finalized.

        Final mesh/texture assets, root-level reports, manifests and logs are
        deliberately retained so the session remains viewable and auditable.
        The worker calls this only after persisting a success/failure report.
        Existing sessions are never swept automatically.
        """
        if (
            not isinstance(session_id, str)
            or not session_id.startswith("session_")
            or Path(session_id).name != session_id
        ):
            raise ValueError("Cleanup requires a direct session_XXX folder name.")

        root = self.root_dir.resolve()
        session_path = self.root_dir / session_id
        if not session_path.is_dir() or session_path.is_symlink():
            raise FileNotFoundError(f"Session directory not found: {session_id}")
        resolved_session = session_path.resolve()
        if resolved_session.parent != root or resolved_session.name != session_id:
            raise ValueError("Refusing to clean a session outside the configured root.")

        # These are generated working folders only. In particular, mesh/ is
        # excluded because it contains the final OBJ, GLB, PLY and texture maps.
        intermediate_dirs = (
            "openmvs",
            "source_uploads",
            "original_images",
            "exif_metadata",
            "camera_poses",
            "sparse_cloud",
            "dense_cloud",
        )
        removed: List[str] = []
        bytes_removed = 0

        for directory_name in intermediate_dirs:
            candidate = session_path / directory_name
            if candidate.is_symlink() or not candidate.is_dir():
                continue

            # Count regular files without following symlinks before removal.
            stack = [candidate]
            directory_bytes = 0
            while stack:
                current = stack.pop()
                try:
                    for child in current.iterdir():
                        if child.is_symlink():
                            continue
                        if child.is_dir():
                            stack.append(child)
                        elif child.is_file():
                            try:
                                directory_bytes += child.stat().st_size
                            except OSError:
                                pass
                except OSError:
                    continue

            shutil.rmtree(candidate)
            removed.append(directory_name)
            bytes_removed += directory_bytes

        return {
            "enabled": True,
            "removed_directories": removed,
            "bytes_removed": bytes_removed,
            "megabytes_removed": round(bytes_removed / (1024 * 1024), 2),
            "final_mesh_and_reports_retained": True,
            "older_sessions_touched": False,
        }
