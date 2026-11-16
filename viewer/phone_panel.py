"""
viewer/phone_panel.py — Smartphone Photograph to 3D World GUI Panel (M7.3)
========================================================================

Provides an upload-first desktop workflow for:
  1. Multi-photo upload and file management (JPG, PNG, HEIC).
  2. Thumbnail preview and EXIF metadata inspector.
  3. Image quality diagnostics (blur, exposure, texture score).
  4. Automatic reconstruction-engine detection and balanced quality selection.
  5. Asynchronous photogrammetry execution with real-time progress tracking.
  6. Session management and instant 3D viewport scene loading.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import threading

from PySide6 import QtWidgets, QtCore, QtGui

from photo_reconstruction.image_loader import ImageLoader
from photo_reconstruction.exif_reader import ExifReader
from photo_reconstruction.image_quality import ImageQualityAssessor
from photo_reconstruction.session_manager import SessionManager
from photo_reconstruction.pipeline import PhoneReconstructionPipeline
from photo_reconstruction.backend_selector import resolve_reconstruction_backend


class ReconstructionWorker(QtCore.QThread):
    """
    Dedicated background worker thread for smartphone 3D photogrammetry.
    Communicates lifecycle, progress, and errors cleanly via Qt signals.
    """

    progress = QtCore.Signal(int, str)
    finished_success = QtCore.Signal(dict)
    failed = QtCore.Signal(str, str)  # (short_error_message, detailed_traceback)

    def __init__(
        self,
        pipeline: PhoneReconstructionPipeline,
        image_paths: List[Path],
        quality_preset: str,
        reference_distance_m: Optional[float],
        backend: str = "native",
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.image_paths = list(image_paths)
        self.quality_preset = quality_preset
        self.reference_distance_m = reference_distance_m
        self.backend = backend

    def run(self):
        try:
            def progress_callback(pct: int, msg: str):
                self.progress.emit(pct, msg)

            if self.backend == "openmvs":
                from photo_reconstruction.openmvs_backend import OpenMVSBackend
                import os

                session_dir = self.pipeline.session_manager.create_session()
                copied = []
                source_dir = session_dir / "original_images"
                import shutil
                from PIL import Image

                for index, source in enumerate(self.image_paths):
                    if not source.is_file():
                        continue
                    suffix = source.suffix.lower()
                    out_suffix = suffix if suffix in {".jpg", ".jpeg", ".png"} else ".jpg"
                    destination = source_dir / f"{source.stem}{out_suffix}"
                    if destination.exists():
                        destination = source_dir / f"{source.stem}_{index:03d}{out_suffix}"

                    if out_suffix == suffix:
                        shutil.copy2(source, destination)
                    else:
                        rgb, _ = ImageLoader.load_image(source)
                        Image.fromarray(rgb).save(
                            destination, format="JPEG", quality=95, optimize=True
                        )
                    copied.append(destination)

                openmvs_dir = os.environ.get("CLEAR360_OPENMVS_DIR")
                colmap_bin = os.environ.get("CLEAR360_COLMAP_BIN")
                backend_runner = OpenMVSBackend(
                    openmvs_dir=Path(openmvs_dir) if openmvs_dir else None,
                    colmap_bin=colmap_bin,
                )
                result = backend_runner.run(
                    copied,
                    session_dir=session_dir,
                    quality_preset=self.quality_preset,
                    progress_cb=progress_callback,
                )
                import json
                from datetime import datetime

                registration = result.get("registration", {})
                mesh_quality = result.get("mesh_quality", {})
                geometry_quality = mesh_quality.get("geometry") or {}
                mesh_warnings = mesh_quality.get("warnings", [])
                registration_warnings = registration.get("warnings", [])
                report = {
                    "session_id": session_dir.name,
                    "backend": "COLMAP + OpenMVS",
                    "uploaded_count": len(copied),
                    "registered_count": result.get("registered_count", 0),
                    "failed_images": result.get("failed_images", []),
                    "image_counts": {
                        "uploaded": len(copied),
                        "registered": result.get("registered_count", 0),
                        "failed": len(result.get("failed_images", [])),
                    },
                    "registration": registration,
                    "camera_overlap": registration.get("camera_overlap", {}),
                    "geometric_metrics": {
                        "sparse_points": result.get("sparse_point_count", 0),
                        "mesh_vertices": geometry_quality.get("vertex_count", 0),
                        "mesh_triangles": geometry_quality.get("face_count", 0),
                        "mean_reprojection_error_px": result.get("reprojection_error_px"),
                    },
                    "mesh_quality": mesh_quality,
                    "diagnostics": {
                        "has_failures": bool(result.get("failed_images")),
                        "failure_reasons": [
                            f"Image '{name}' was not registered by COLMAP."
                            for name in result.get("failed_images", [])
                        ],
                        "registration_warnings": registration_warnings,
                        "mesh_warnings": mesh_warnings,
                        "guidance": (
                            "Some uploaded photographs were not registered. "
                            "Review failed_images and the reconstruction diagnostics."
                            if result.get("failed_images") else
                            "Review camera overlap and mesh quality warnings before using this model."
                            if registration_warnings or mesh_warnings else
                            "No camera-overlap or mesh QA warnings were reported."
                        ),
                    },
                    "notes": "COLMAP + OpenMVS reconstruction completed.",
                    "mesh_ply": str(result["mesh_ply"]),
                    "mesh_obj": str(result.get("mesh_obj", "")),
                    "mesh_glb": str(result["mesh_glb"]),
                    "dense_ply": str(result["dense_ply"]),
                    "log_file": str(result["log_file"]),
                    "mesh_quality_report": str(result.get("mesh_quality_report", "")),
                    "orientation": result.get("orientation"),
                    "created_at": datetime.now().astimezone().isoformat(),
                }
                (session_dir / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                res = {
                    "success": True,
                    "session_id": session_dir.name,
                    "session_dir": session_dir,
                    "mesh_ply": result["mesh_ply"],
                    "mesh_viewer": result.get("mesh_obj", result["mesh_ply"]),
                    "mesh_glb": result["mesh_glb"],
                    "dense_ply": result["dense_ply"],
                    "report": report,
                }
            else:
                res = self.pipeline.run_pipeline(
                    image_paths=self.image_paths,
                    quality_preset=self.quality_preset,
                    reference_distance_m=self.reference_distance_m,
                    progress_cb=progress_callback,
                )

            if res.get("success", False):
                self.finished_success.emit(res)
            else:
                report = res.get("report", {})
                notes = report.get("notes") or "Camera registration failed for the uploaded photographs."
                self.failed.emit(notes, "")
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            self.failed.emit(str(exc), tb)


class PhoneReconstructionPanel(QtWidgets.QWidget):
    """
    Dedicated GUI sidebar panel for Smartphone Photo to 3D World Reconstruction.
    """

    reconstruction_completed = QtCore.Signal(str, str)  # (session_id, mesh_ply_path)
    load_session_requested = QtCore.Signal(str)          # session_id
    display_mode_changed = QtCore.Signal(str)            # 'mesh' or 'cloud'

    def __init__(self, data_root: Optional[Path] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.data_root = data_root
        self.session_manager = SessionManager(data_root)
        self.pipeline = PhoneReconstructionPipeline(data_root)
        self.worker: Optional[ReconstructionWorker] = None

        self.selected_image_paths: List[Path] = []
        self.image_metadata: Dict[str, Dict[str, Any]] = {}
        self.current_session_id: Optional[str] = None

        self._init_ui()
        self._refresh_session_list()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Header Title
        title_box = QtWidgets.QWidget()
        title_layout = QtWidgets.QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        lbl_title = QtWidgets.QLabel("PHOTO → 3D WORLD")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #a855f7; letter-spacing: 1px;")
        lbl_subtitle = QtWidgets.QLabel("Smartphone Photogrammetry & SfM")
        lbl_subtitle.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_layout.addWidget(lbl_title)
        title_layout.addWidget(lbl_subtitle)
        main_layout.addWidget(title_box)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # 1. Photograph Upload & File List
        # ---------------------------------------------------------------------
        grp_upload = QtWidgets.QGroupBox("1. Smartphone Photographs")
        grp_up_layout = QtWidgets.QVBoxLayout(grp_upload)
        grp_up_layout.setSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_photos = QtWidgets.QPushButton("📁 Add Photos...")
        self.btn_add_photos.setStyleSheet("background-color: #7e22ce; color: #ffffff; font-weight: bold;")
        self.btn_add_photos.clicked.connect(self._on_add_photos_clicked)

        self.btn_clear_photos = QtWidgets.QPushButton("🗑️ Clear")
        self.btn_clear_photos.clicked.connect(self._on_clear_photos_clicked)

        btn_row.addWidget(self.btn_add_photos)
        btn_row.addWidget(self.btn_clear_photos)
        grp_up_layout.addLayout(btn_row)

        self.list_photos = QtWidgets.QListWidget()
        self.list_photos.setFixedHeight(120)
        self.list_photos.setStyleSheet("background-color: #161821; border: 1px solid #282c3c; border-radius: 4px;")
        self.list_photos.currentRowChanged.connect(self._on_photo_selected)
        grp_up_layout.addWidget(self.list_photos)

        self.lbl_photo_count = QtWidgets.QLabel("0 photographs selected (min 2, rec 4–40+)")
        self.lbl_photo_count.setStyleSheet("font-size: 10px; color: #94a3b8;")
        grp_up_layout.addWidget(self.lbl_photo_count)

        self.btn_capture_guide = QtWidgets.QPushButton("📖 Capture Guidelines")
        self.btn_capture_guide.setStyleSheet("background-color: #1e2230; font-size: 11px;")
        self.btn_capture_guide.clicked.connect(self._on_show_capture_guide)
        grp_up_layout.addWidget(self.btn_capture_guide)

        layout.addWidget(grp_upload)

        # ---------------------------------------------------------------------
        # 2. Selected Photo EXIF & Quality Inspector
        # ---------------------------------------------------------------------
        grp_meta = QtWidgets.QGroupBox("2. Photo Diagnostics & EXIF")
        grp_meta_layout = QtWidgets.QVBoxLayout(grp_meta)
        grp_meta_layout.setSpacing(4)

        self.lbl_meta_dims = QtWidgets.QLabel("Dimensions: --")
        self.lbl_meta_camera = QtWidgets.QLabel("Camera: --")
        self.lbl_meta_focal = QtWidgets.QLabel("Focal Length: --")
        self.lbl_meta_quality = QtWidgets.QLabel("Quality Rating: --")
        self.lbl_meta_quality.setStyleSheet("font-weight: bold; color: #38bdf8;")

        for lbl in [self.lbl_meta_dims, self.lbl_meta_camera, self.lbl_meta_focal, self.lbl_meta_quality]:
            lbl.setStyleSheet(lbl.styleSheet() + " font-size: 11px;")
            grp_meta_layout.addWidget(lbl)

        layout.addWidget(grp_meta)

        # ---------------------------------------------------------------------
        # 3. Automatic Reconstruction
        # ---------------------------------------------------------------------
        grp_settings = QtWidgets.QGroupBox("3. Automatic 3D Reconstruction")
        grp_set_layout = QtWidgets.QVBoxLayout(grp_settings)
        grp_set_layout.setSpacing(6)

        self.lbl_auto_engine = QtWidgets.QLabel(
            "No settings required. CLEAR360 detects the available 3D engine "
            "and selects a laptop-safe reconstruction quality automatically."
        )
        self.lbl_auto_engine.setWordWrap(True)
        self.lbl_auto_engine.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        grp_set_layout.addWidget(self.lbl_auto_engine)

        self.lbl_auto_quality = QtWidgets.QLabel(
            "Quality: Automatic (balanced detail and memory use)"
        )
        self.lbl_auto_quality.setStyleSheet("font-size: 10px; color: #38bdf8;")
        grp_set_layout.addWidget(self.lbl_auto_quality)

        layout.addWidget(grp_settings)

        # ---------------------------------------------------------------------
        # 4. Reconstruction Execution & Progress
        # ---------------------------------------------------------------------
        grp_exec = QtWidgets.QGroupBox("4. Reconstruction Engine")
        grp_exec_layout = QtWidgets.QVBoxLayout(grp_exec)
        grp_exec_layout.setSpacing(6)

        self.btn_start = QtWidgets.QPushButton("🚀 Start 3D Reconstruction")
        self.btn_start.setStyleSheet(
            "background-color: #059669; color: #ffffff; font-weight: bold; padding: 8px; font-size: 13px;"
        )
        self.btn_start.clicked.connect(self._on_start_clicked)
        grp_exec_layout.addWidget(self.btn_start)

        self.prog_bar = QtWidgets.QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setTextVisible(True)
        grp_exec_layout.addWidget(self.prog_bar)

        self.lbl_status = QtWidgets.QLabel("Ready for photograph upload.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 10px; color: #38bdf8;")
        grp_exec_layout.addWidget(self.lbl_status)

        self.lbl_elapsed = QtWidgets.QLabel("")
        self.lbl_elapsed.setStyleSheet("font-size: 10px; color: #a5f3fc; font-weight: bold;")
        grp_exec_layout.addWidget(self.lbl_elapsed)

        layout.addWidget(grp_exec)

        # ---------------------------------------------------------------------
        # 5. Session History & Result Viewer
        # ---------------------------------------------------------------------
        grp_sessions = QtWidgets.QGroupBox("5. Saved Sessions")
        grp_sess_layout = QtWidgets.QVBoxLayout(grp_sessions)
        grp_sess_layout.setSpacing(6)

        self.combo_sessions = QtWidgets.QComboBox()
        self.combo_sessions.currentIndexChanged.connect(self._on_session_changed)
        grp_sess_layout.addWidget(self.combo_sessions)

        btn_sess_row = QtWidgets.QHBoxLayout()
        self.btn_open_scene = QtWidgets.QPushButton("👁️ Open in 3D View")
        self.btn_open_scene.setStyleSheet("background-color: #0284c7; font-weight: bold;")
        self.btn_open_scene.clicked.connect(self._on_open_scene_clicked)

        self.btn_view_report = QtWidgets.QPushButton("📋 Report")
        self.btn_view_report.clicked.connect(self._on_view_report_clicked)

        btn_sess_row.addWidget(self.btn_open_scene)
        btn_sess_row.addWidget(self.btn_view_report)
        grp_sess_layout.addLayout(btn_sess_row)

        # 3D View Display Mode: Surface Mesh vs Point Cloud
        lbl_display = QtWidgets.QLabel("3D View Display Mode:")
        lbl_display.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: bold; margin-top: 4px;")
        grp_sess_layout.addWidget(lbl_display)

        display_row = QtWidgets.QHBoxLayout()
        self.btn_view_tex = QtWidgets.QPushButton("🎨 Textured")
        self.btn_view_tex.setCheckable(True)
        self.btn_view_tex.setChecked(True)
        self.btn_view_tex.setStyleSheet(
            "background-color: #10b981; color: #ffffff; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_view_tex.clicked.connect(lambda: self._on_display_mode_clicked("textured"))

        self.btn_view_mesh = QtWidgets.QPushButton("🧊 Surface")
        self.btn_view_mesh.setCheckable(True)
        self.btn_view_mesh.setChecked(False)
        self.btn_view_mesh.setStyleSheet(
            "background-color: #334155; color: #cbd5e1; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_view_mesh.clicked.connect(lambda: self._on_display_mode_clicked("mesh"))

        self.btn_view_cloud = QtWidgets.QPushButton("🔍 Cloud")
        self.btn_view_cloud.setCheckable(True)
        self.btn_view_cloud.setChecked(False)
        self.btn_view_cloud.setStyleSheet(
            "background-color: #334155; color: #cbd5e1; font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        self.btn_view_cloud.clicked.connect(lambda: self._on_display_mode_clicked("cloud"))

        display_row.addWidget(self.btn_view_tex)
        display_row.addWidget(self.btn_view_mesh)
        display_row.addWidget(self.btn_view_cloud)
        grp_sess_layout.addLayout(display_row)

        layout.addWidget(grp_sessions)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Event Handlers & User Actions
    # -------------------------------------------------------------------------
    def _on_add_photos_clicked(self):
        """Open file dialog to pick smartphone images."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Smartphone Photographs",
            "",
            "Images (*.jpg *.jpeg *.png *.heic *.bmp);;All Files (*)",
        )
        if not files:
            return

        for f in files:
            p = Path(f)
            if p not in self.selected_image_paths:
                self.selected_image_paths.append(p)
                self.list_photos.addItem(p.name)

                # Pre-analyze metadata
                try:
                    rgb, meta = ImageLoader.load_image(p)
                    c_info = ExifReader.get_camera_info(p, meta["width"], meta["height"])
                    q_res = ImageQualityAssessor.evaluate(rgb)
                    self.image_metadata[p.name] = {"meta": meta, "exif": c_info, "quality": q_res}
                except Exception as e:
                    self.image_metadata[p.name] = {"error": str(e)}

        count = len(self.selected_image_paths)
        self.lbl_photo_count.setText(f"{count} photographs selected (min 2, rec 4–40+)")
        if count > 0 and self.list_photos.currentRow() == -1:
            self.list_photos.setCurrentRow(0)
        if count >= 2:
            self.lbl_status.setText(
                f"{count} photographs ready. CLEAR360 will select the reconstruction engine automatically."
            )

    def _on_clear_photos_clicked(self):
        """Clear selected images."""
        self.selected_image_paths.clear()
        self.image_metadata.clear()
        self.list_photos.clear()
        self.lbl_photo_count.setText("0 photographs selected (min 2, rec 4–40+)")
        self._reset_meta_labels()

    def _on_show_capture_guide(self):
        """Display smartphone capture guidelines dialog."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Smartphone Photogrammetry Capture Guidelines")
        dlg.resize(550, 420)
        layout = QtWidgets.QVBoxLayout(dlg)

        text = QtWidgets.QTextBrowser()
        text.setHtml("""
        <h3 style='color: #a855f7;'>CLEAR360 — Smartphone Photogrammetry Capture Guidelines</h3>
        <p>Follow these best practices to ensure successful 3D Structure-from-Motion reconstruction:</p>
        <ul>
            <li><b>60–80% Visual Overlap:</b> Adjacent photographs must share significant overlapping scene features.</li>
            <li><b>Stationary Environment:</b> Photograph static scenes (parked cars, walls, alleys, pillars). Avoid moving objects or pedestrians.</li>
            <li><b>Move Physically:</b> Walk around the subject in an arc or grid. Do not stand in one place and simply pan/rotate the camera.</li>
            <li><b>Consistent Main Lens:</b> Use your phone's primary 1x wide lens for all photos. Do not switch between ultra-wide (0.5x) and telephoto (3x/5x).</li>
            <li><b>Avoid Digital Zoom:</b> Digital zoom crops pixels and destroys optical geometry calibration.</li>
            <li><b>Ensure Sharp Focus:</b> Tap to focus. Avoid motion blur, low light, and dirty camera lenses.</li>
            <li><b>Feature-Rich Surfaces:</b> Avoid blank featureless white walls or uniform asphalt without visual landmarks.</li>
            <li><b>Recommended Photo Counts:</b>
                <ul>
                    <li>Single obstacle or small vehicle: <b>4–12 photographs</b></li>
                    <li>Parking bay or small street corner: <b>12–25 photographs</b></li>
                    <li>Full urban scene or large alleyway: <b>25–40+ photographs</b></li>
                </ul>
            </li>
        </ul>
        """)
        layout.addWidget(text)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        layout.addWidget(btn_box)
        dlg.exec()

    def _on_photo_selected(self, row: int):
        """Display metadata and quality diagnostics for selected photo."""
        if row < 0 or row >= len(self.selected_image_paths):
            self._reset_meta_labels()
            return

        p = self.selected_image_paths[row]
        data = self.image_metadata.get(p.name, {})

        if "error" in data:
            self.lbl_meta_dims.setText(f"Error: {data['error']}")
            return

        meta = data.get("meta", {})
        exif = data.get("exif", {})
        q = data.get("quality", {})

        w = meta.get("width", "--")
        h = meta.get("height", "--")
        self.lbl_meta_dims.setText(f"Dimensions: {w} × {h} px")

        make = exif.get("make", "")
        model = exif.get("model", "Smartphone")
        cam_str = f"{make} {model}".strip() or "Standard Smartphone"
        self.lbl_meta_camera.setText(f"Camera: {cam_str}")

        fl = exif.get("focal_length_mm")
        fl_str = f"{fl:.1f} mm" if fl else "Estimated"
        self.lbl_meta_focal.setText(f"Focal Length: {fl_str}")

        q_score = q.get("quality_score", 0.0)
        q_stat = q.get("status", "--")
        color = "#22c55e" if q_score >= 70.0 else ("#eab308" if q_score >= 50.0 else "#ef4444")
        self.lbl_meta_quality.setText(f"Quality Rating: {q_score}/100 ({q_stat})")
        self.lbl_meta_quality.setStyleSheet(f"font-weight: bold; color: {color};")

    def _reset_meta_labels(self):
        self.lbl_meta_dims.setText("Dimensions: --")
        self.lbl_meta_camera.setText("Camera: --")
        self.lbl_meta_focal.setText("Focal Length: --")
        self.lbl_meta_quality.setText("Quality Rating: --")
        self.lbl_meta_quality.setStyleSheet("font-weight: bold; color: #94a3b8;")

    def _on_start_clicked(self):
        """Run the best installed reconstruction engine with automatic settings."""
        if len(self.selected_image_paths) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Add more photographs",
                "Please upload at least two photographs with overlapping views of the same scene.",
            )
            return

        backend, backend_reason = resolve_reconstruction_backend()
        if backend == "openmvs" and len(self.selected_image_paths) < 6:
            backend = "native"
            backend_reason = (
                "built-in CLEAR360 engine selected automatically because this upload "
                "has fewer than six photos (COLMAP + OpenMVS needs at least six)"
            )

        self.btn_start.setEnabled(False)
        self.btn_add_photos.setEnabled(False)
        self.prog_bar.setValue(0)
        self.lbl_auto_engine.setText(f"Automatic engine: {backend_reason}")
        self.lbl_status.setText("Preparing automatic 3D reconstruction...")
        self.lbl_status.setStyleSheet("font-size: 10px; color: #38bdf8;")

        # A balanced preset is selected automatically to avoid the excessive
        # runtime and memory use of Maximum on an 8 GB laptop.
        preset = "standard"
        reference_distance = None

        import time
        self._recon_start_time = time.time()
        if not hasattr(self, "elapsed_timer"):
            self.elapsed_timer = QtCore.QTimer(self)
            self.elapsed_timer.setInterval(250)
            self.elapsed_timer.timeout.connect(self._update_elapsed_ui)
        self.elapsed_timer.start()

        self.worker = ReconstructionWorker(
            pipeline=self.pipeline,
            image_paths=self.selected_image_paths,
            quality_preset=preset,
            reference_distance_m=reference_distance,
            backend=backend,
            parent=self,
        )
        self.worker.progress.connect(self._update_progress_ui)
        self.worker.finished_success.connect(self._on_reconstruction_success)
        self.worker.failed.connect(self._on_reconstruction_failed)
        self.worker.start()

    def _update_elapsed_ui(self):
        if hasattr(self, "_recon_start_time") and self._recon_start_time > 0:
            import time
            elapsed = time.time() - self._recon_start_time
            dots = "." * (int(elapsed * 2) % 4)
            self.lbl_elapsed.setText(f"Elapsed: {elapsed:.1f} s {dots}")

    @QtCore.Slot(int, str)
    def _update_progress_ui(self, pct: int, msg: str):
        self.prog_bar.setValue(pct)
        self.lbl_status.setText(msg)

    @QtCore.Slot(dict)
    def _on_reconstruction_success(self, res: dict):
        if hasattr(self, "elapsed_timer") and self.elapsed_timer.isActive():
            self.elapsed_timer.stop()
        self.lbl_elapsed.setText("")
        self.btn_start.setEnabled(True)
        self.btn_add_photos.setEnabled(True)
        self.prog_bar.setValue(100)

        session_id = str(res.get("session_id", ""))
        # The desktop viewport can display the OpenMVS textured OBJ/atlas
        # directly; keep the GLB in the session for export and external viewers.
        mesh_path = str(res.get("mesh_viewer") or res.get("mesh_ply", ""))
        self.current_session_id = session_id

        backend_label = "COLMAP + OpenMVS" if res.get("mesh_glb") else "CLEAR360 Python"
        result_report = res.get("report") or {}
        quality = result_report.get("mesh_quality", {})
        quality_status = quality.get("status", "not available")
        quality_score = quality.get("score")
        quality_label = (
            f"Mesh QA: {quality_status.upper()} ({quality_score:.1f}/100)"
            if isinstance(quality_score, (int, float))
            else f"Mesh QA: {quality_status.upper()}"
        )
        registration = result_report.get("registration") or {}
        overlap = (
            registration.get("camera_overlap")
            or result_report.get("camera_overlap")
            or {}
        )
        if overlap:
            if not overlap.get("available", False):
                overlap_status = "NOT AVAILABLE"
            elif overlap.get("component_count", 0) > 1:
                overlap_status = "REVIEW"
            else:
                overlap_status = "CONNECTED"
            quality_label += f" · Camera overlap: {overlap_status}"
        self.lbl_status.setText(
            f"Reconstruction successful! {backend_label} backend: {session_id}. {quality_label}."
        )
        status_color = (
            "#22c55e" if quality_status == "healthy"
            else "#eab308" if quality_status == "review"
            else "#ef4444" if quality_status == "poor"
            else "#38bdf8"
        )
        self.lbl_status.setStyleSheet(f"font-size: 10px; color: {status_color};")
        self._refresh_session_list()

        # Emit completion signal to open in 3D viewport
        self.reconstruction_completed.emit(session_id, mesh_path)

        QtWidgets.QMessageBox.information(
            self,
            "Photogrammetry Complete",
            f"3D reconstruction completed successfully!\nSession: {session_id}\n\n"
            "The reconstructed scene has been loaded into the 3D viewer.",
        )

    @QtCore.Slot(str, str)
    def _on_reconstruction_failed(self, error_msg: str, tb_str: str):
        if hasattr(self, "elapsed_timer") and self.elapsed_timer.isActive():
            self.elapsed_timer.stop()
        self.lbl_elapsed.setText("")
        self.btn_start.setEnabled(True)
        self.btn_add_photos.setEnabled(True)
        self.prog_bar.setValue(0)

        # Uploaded images are preserved!
        self.lbl_status.setText(f"Reconstruction failed: {error_msg}")
        self.lbl_status.setStyleSheet("font-size: 10px; color: #ef4444;")
        self._refresh_session_list()

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setWindowTitle("Reconstruction Incomplete / Error")
        msg_box.setText(f"Photogrammetry reconstruction could not be completed:\n\n{error_msg}")
        msg_box.setInformativeText("Your uploaded photographs remain in the list. You can adjust the quality preset, add more overlapping photos, or retry.")
        if tb_str:
            msg_box.setDetailedText(tb_str)
        msg_box.exec()

    def _refresh_session_list(self):
        """Update sessions dropdown with all saved directories."""
        self.combo_sessions.blockSignals(True)
        self.combo_sessions.clear()
        sessions = self.session_manager.list_sessions()
        if not sessions:
            self.combo_sessions.addItem("No saved sessions", None)
            self.btn_open_scene.setEnabled(False)
            self.btn_view_report.setEnabled(False)
        else:
            for s in reversed(sessions):  # Most recent first
                self.combo_sessions.addItem(s, s)
            self.btn_open_scene.setEnabled(True)
            self.btn_view_report.setEnabled(True)
        self.combo_sessions.blockSignals(False)

    def _on_session_changed(self, index: int):
        sess_id = self.combo_sessions.currentData()
        self.current_session_id = sess_id

    def _on_open_scene_clicked(self):
        sess_id = self.combo_sessions.currentData()
        if sess_id:
            self.load_session_requested.emit(sess_id)

    def _on_view_report_clicked(self):
        sess_id = self.combo_sessions.currentData()
        if not sess_id:
            return
        report = self.session_manager.get_session_report(sess_id)
        if not report:
            QtWidgets.QMessageBox.information(self, "Report", f"No report.json found for {sess_id}.")
            return

        # Display formatted report
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Reconstruction Quality Report — {sess_id}")
        dlg.resize(600, 500)
        dlg_layout = QtWidgets.QVBoxLayout(dlg)

        txt = QtWidgets.QTextEdit()
        txt.setReadOnly(True)
        import json
        txt.setPlainText(json.dumps(report, indent=2))
        dlg_layout.addWidget(txt)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        dlg_layout.addWidget(btn_box)
        dlg.exec()

    def _on_display_mode_clicked(self, mode: str):
        self.set_display_mode(mode)
        self.display_mode_changed.emit(mode)

    def set_display_mode(self, mode: str):
        active_style = "color: #ffffff; font-weight: bold; padding: 6px; border-radius: 4px;"
        inactive_style = "background-color: #334155; color: #cbd5e1; font-weight: bold; padding: 6px; border-radius: 4px;"

        is_tex = mode in ("textured", "texture")
        is_mesh = mode == "mesh"
        is_cloud = mode == "cloud"

        self.btn_view_tex.setChecked(is_tex)
        self.btn_view_mesh.setChecked(is_mesh)
        self.btn_view_cloud.setChecked(is_cloud)

        self.btn_view_tex.setStyleSheet(f"background-color: #10b981; {active_style}" if is_tex else inactive_style)
        self.btn_view_mesh.setStyleSheet(f"background-color: #8b5cf6; {active_style}" if is_mesh else inactive_style)
        self.btn_view_cloud.setStyleSheet(f"background-color: #0284c7; {active_style}" if is_cloud else inactive_style)
