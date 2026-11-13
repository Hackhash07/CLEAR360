"""
viewer/controls.py — Control Panel for CLEAR360 3D Surround Viewer

Provides responsive, automotive-styled controls:
  - Pipeline source selection (Ground Truth, Calibrated AI, Raw AI)
  - Frame / timestamp timeline navigation (Frame 0000, 0005, 0010)
  - Multi-layer visibility toggles (RGB/Semantic clouds, Occupancy grid, Vehicle, BBoxes, Clearance line)
  - Camera viewpoint presets (Chase 3D, Front, Rear, Left, Right, Top perspective, Reset)
  - Visual semantic legend
"""

from typing import Optional
from PySide6 import QtWidgets, QtCore, QtGui


class ViewerControlPanel(QtWidgets.QWidget):
    """
    Left-hand sidebar control panel for user interaction, layer toggles, and camera presets.
    """

    vis_mode_changed = QtCore.Signal(str)
    source_changed = QtCore.Signal(str)
    frame_changed = QtCore.Signal(int)
    scenario_changed = QtCore.Signal(str)
    cloud_mode_changed = QtCore.Signal(str)
    point_size_changed = QtCore.Signal(float)
    occupancy_toggled = QtCore.Signal(bool, bool, bool)  # (show_occupied, show_free, show_unknown)
    vehicle_toggled = QtCore.Signal(bool)
    envelope_toggled = QtCore.Signal(bool)
    bboxes_toggled = QtCore.Signal(bool)
    clearance_line_toggled = QtCore.Signal(bool)
    grid_toggled = QtCore.Signal(bool)
    preset_selected = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Header Title
        title_box = QtWidgets.QWidget()
        title_layout = QtWidgets.QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        app_title = QtWidgets.QLabel("CLEAR360")
        app_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00e5ff; letter-spacing: 1px;")
        sub_title = QtWidgets.QLabel("Automotive 3D Surround Viewer")
        sub_title.setStyleSheet("font-size: 11px; color: #8892b0;")
        title_layout.addWidget(app_title)
        title_layout.addWidget(sub_title)
        main_layout.addWidget(title_box)

        # Scroll area for compact panels
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # 0. Primary Visualization Mode Group (M7.1)
        # ---------------------------------------------------------------------
        grp_vis_mode = QtWidgets.QGroupBox("Primary Visualization Mode")
        grp_vis_mode_layout = QtWidgets.QVBoxLayout(grp_vis_mode)
        grp_vis_mode_layout.setSpacing(6)

        self.combo_vis_mode = QtWidgets.QComboBox()
        self.combo_vis_mode.addItem("✨ Photorealistic 3D Surface", "surface")
        self.combo_vis_mode.addItem("🔍 Point Cloud Debug Mode", "pointcloud")
        self.combo_vis_mode.addItem("🧊 Semantic / Occupancy Grid", "semantic_occupancy")
        self.combo_vis_mode.addItem("🚗 Driver View (Cockpit Surround)", "driver")
        self.combo_vis_mode.currentIndexChanged.connect(self._on_vis_mode_changed)
        grp_vis_mode_layout.addWidget(self.combo_vis_mode)

        self.vis_hint = QtWidgets.QLabel("M7.2 Driver View: Clean photorealistic cockpit experience.")
        self.vis_hint.setWordWrap(True)
        self.vis_hint.setStyleSheet("font-size: 10px; color: #38bdf8;")
        grp_vis_mode_layout.addWidget(self.vis_hint)
        layout.addWidget(grp_vis_mode)

        # ---------------------------------------------------------------------
        # 1. Pipeline Source Group
        # ---------------------------------------------------------------------
        grp_source = QtWidgets.QGroupBox("Reconstruction Pipeline")
        grp_source_layout = QtWidgets.QVBoxLayout(grp_source)
        grp_source_layout.setSpacing(6)

        self.lbl_pipeline_badge = QtWidgets.QLabel("[SIMULATION GROUND TRUTH]")
        self.lbl_pipeline_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_pipeline_badge.setStyleSheet(
            "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
            "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 11px;"
        )
        grp_source_layout.addWidget(self.lbl_pipeline_badge)

        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItem("Ground Truth (Cycles)", "ground_truth")
        self.combo_source.addItem("Calibrated AI (Depth Anything V2)", "ai_calibrated")
        self.combo_source.addItem("Raw AI (Uncalibrated)", "ai_raw")
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        grp_source_layout.addWidget(self.combo_source)

        source_hint = QtWidgets.QLabel("Switch between GT and AI pipelines independently.")
        source_hint.setWordWrap(True)
        source_hint.setStyleSheet("font-size: 10px; color: #8892b0;")
        grp_source_layout.addWidget(source_hint)
        layout.addWidget(grp_source)

        # ---------------------------------------------------------------------
        # 2. Timeline / Frame Selection Group
        # ---------------------------------------------------------------------
        grp_frame = QtWidgets.QGroupBox("Timeline & Frames")
        grp_frame_layout = QtWidgets.QVBoxLayout(grp_frame)
        grp_frame_layout.setSpacing(6)

        frame_ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_prev_frame = QtWidgets.QPushButton("◀")
        self.btn_prev_frame.setFixedWidth(36)
        self.btn_prev_frame.clicked.connect(self._prev_frame)

        self.combo_frame = QtWidgets.QComboBox()
        self.combo_frame.addItem("Frame 0000 (t = 0.00 s)", 0)
        self.combo_frame.addItem("Frame 0005 (t = 0.25 s)", 5)
        self.combo_frame.addItem("Frame 0010 (t = 0.50 s)", 10)
        self.combo_frame.currentIndexChanged.connect(self._on_frame_changed)

        self.btn_next_frame = QtWidgets.QPushButton("▶")
        self.btn_next_frame.setFixedWidth(36)
        self.btn_next_frame.clicked.connect(self._next_frame)

        frame_ctrl_layout.addWidget(self.btn_prev_frame)
        frame_ctrl_layout.addWidget(self.combo_frame)
        frame_ctrl_layout.addWidget(self.btn_next_frame)
        grp_frame_layout.addLayout(frame_ctrl_layout)
        layout.addWidget(grp_frame)

        # ---------------------------------------------------------------------
        # 2b. M8 Benchmark Scenarios Group
        # ---------------------------------------------------------------------
        grp_scenario = QtWidgets.QGroupBox("M8 Benchmark Scenarios")
        grp_scen_layout = QtWidgets.QVBoxLayout(grp_scenario)
        grp_scen_layout.setSpacing(6)

        self.combo_scenario = QtWidgets.QComboBox()
        self.combo_scenario.addItem("Baseline Simulation (M7)", "default")
        self.combo_scenario.addItem("01: Narrow Urban Street", "scenario_01_narrow_street")
        self.combo_scenario.addItem("02: Double Parked Cars", "scenario_02_double_parked")
        self.combo_scenario.addItem("03: Close Pillars & Walls", "scenario_03_close_pillars_walls")
        self.combo_scenario.addItem("04: Tight Parking Gap", "scenario_04_tight_parking_gap")
        self.combo_scenario.addItem("05: Speed Breakers & Kerbs", "scenario_05_speed_breakers_kerbs")
        self.combo_scenario.addItem("06: Approaching Pedestrians", "scenario_06_approaching_pedestrians")
        self.combo_scenario.addItem("07: Harsh Lighting & Shadows", "scenario_07_harsh_lighting_shadows")
        self.combo_scenario.addItem("08: Partially Occluded", "scenario_08_partially_occluded")
        self.combo_scenario.addItem("09: T-Junction 90° Turn", "scenario_09_t_junction_turn")
        self.combo_scenario.addItem("10: Construction Zone", "scenario_10_construction_zone")
        self.combo_scenario.currentIndexChanged.connect(self._on_scenario_changed)
        grp_scen_layout.addWidget(self.combo_scenario)
        layout.addWidget(grp_scenario)

        # ---------------------------------------------------------------------
        # 3. Layer Visibility Toggles Group
        # ---------------------------------------------------------------------
        self.grp_layers = QtWidgets.QGroupBox("Display Layers")
        grp_layers_layout = QtWidgets.QVBoxLayout(self.grp_layers)
        grp_layers_layout.setSpacing(6)

        # Point cloud mode
        lbl_cloud = QtWidgets.QLabel("Point Cloud Mode:")
        lbl_cloud.setStyleSheet("font-weight: bold; font-size: 11px;")
        grp_layers_layout.addWidget(lbl_cloud)

        self.combo_cloud = QtWidgets.QComboBox()
        self.combo_cloud.addItem("Photorealistic RGB", "rgb")
        self.combo_cloud.addItem("Semantic Classification", "semantic")
        self.combo_cloud.addItem("Reconstruction Confidence", "confidence")
        self.combo_cloud.addItem("Hidden", "none")
        self.combo_cloud.currentIndexChanged.connect(self._on_cloud_mode_changed)
        grp_layers_layout.addWidget(self.combo_cloud)

        # Point size slider
        size_layout = QtWidgets.QHBoxLayout()
        lbl_size = QtWidgets.QLabel("Size:")
        lbl_size.setStyleSheet("font-size: 11px;")
        self.slider_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_size.setRange(10, 60)  # 1.0 to 6.0 px
        self.slider_size.setValue(25)
        self.slider_size.valueChanged.connect(self._on_size_changed)
        self.lbl_size_val = QtWidgets.QLabel("2.5 px")
        self.lbl_size_val.setStyleSheet("font-size: 11px; font-weight: bold; min-width: 45px;")

        size_layout.addWidget(lbl_size)
        size_layout.addWidget(self.slider_size)
        size_layout.addWidget(self.lbl_size_val)
        grp_layers_layout.addLayout(size_layout)

        # Checkboxes for geometry and overlays
        self.chk_vehicle = QtWidgets.QCheckBox("Hyundai Creta 3D Model")
        self.chk_vehicle.setChecked(True)
        self.chk_vehicle.toggled.connect(self.vehicle_toggled.emit)
        grp_layers_layout.addWidget(self.chk_vehicle)

        self.chk_envelope = QtWidgets.QCheckBox("Clearance Envelope (Body + Mirrors)")
        self.chk_envelope.setChecked(True)
        self.chk_envelope.toggled.connect(self.envelope_toggled.emit)
        grp_layers_layout.addWidget(self.chk_envelope)

        self.chk_bboxes = QtWidgets.QCheckBox("Obstacle 3D Bounding Boxes")
        self.chk_bboxes.setChecked(True)
        self.chk_bboxes.toggled.connect(self.bboxes_toggled.emit)
        grp_layers_layout.addWidget(self.chk_bboxes)

        self.chk_clearance = QtWidgets.QCheckBox("Clearance Measurement Line")
        self.chk_clearance.setChecked(True)
        self.chk_clearance.toggled.connect(self.clearance_line_toggled.emit)
        grp_layers_layout.addWidget(self.chk_clearance)

        # Occupancy checkboxes
        self.chk_occ_occupied = QtWidgets.QCheckBox("Occupancy: Occupied (Red)")
        self.chk_occ_occupied.setChecked(False)
        self.chk_occ_occupied.toggled.connect(self._on_occ_toggled)
        grp_layers_layout.addWidget(self.chk_occ_occupied)

        self.chk_occ_free = QtWidgets.QCheckBox("Occupancy: Free Space (Cyan)")
        self.chk_occ_free.setChecked(False)
        self.chk_occ_free.toggled.connect(self._on_occ_toggled)
        grp_layers_layout.addWidget(self.chk_occ_free)

        self.chk_occ_unknown = QtWidgets.QCheckBox("Occupancy: Unknown Space (Gray)")
        self.chk_occ_unknown.setChecked(False)
        self.chk_occ_unknown.toggled.connect(self._on_occ_toggled)
        grp_layers_layout.addWidget(self.chk_occ_unknown)

        self.chk_grid = QtWidgets.QCheckBox("Ground Reference Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self.grid_toggled.emit)
        grp_layers_layout.addWidget(self.chk_grid)

        layout.addWidget(self.grp_layers)

        # ---------------------------------------------------------------------
        # 4. Camera Viewpoint Presets Group
        # ---------------------------------------------------------------------
        grp_camera = QtWidgets.QGroupBox("Camera Viewpoints")
        grp_camera_layout = QtWidgets.QGridLayout(grp_camera)
        grp_camera_layout.setSpacing(6)

        btn_chase = QtWidgets.QPushButton("Chase 3D")
        btn_chase.clicked.connect(lambda: self.preset_selected.emit("chase"))
        btn_top = QtWidgets.QPushButton("Top-Down")
        btn_top.clicked.connect(lambda: self.preset_selected.emit("top"))

        btn_front = QtWidgets.QPushButton("Front View")
        btn_front.clicked.connect(lambda: self.preset_selected.emit("front"))
        btn_rear = QtWidgets.QPushButton("Rear View")
        btn_rear.clicked.connect(lambda: self.preset_selected.emit("rear"))

        btn_left = QtWidgets.QPushButton("Left View")
        btn_left.clicked.connect(lambda: self.preset_selected.emit("left"))
        btn_right = QtWidgets.QPushButton("Right View")
        btn_right.clicked.connect(lambda: self.preset_selected.emit("right"))

        btn_reset = QtWidgets.QPushButton("Reset Camera")
        btn_reset.setStyleSheet("background-color: #2c3e50; font-weight: bold;")
        btn_reset.clicked.connect(lambda: self.preset_selected.emit("reset"))

        grp_camera_layout.addWidget(btn_chase, 0, 0)
        grp_camera_layout.addWidget(btn_top, 0, 1)
        grp_camera_layout.addWidget(btn_front, 1, 0)
        grp_camera_layout.addWidget(btn_rear, 1, 1)
        grp_camera_layout.addWidget(btn_left, 2, 0)
        grp_camera_layout.addWidget(btn_right, 2, 1)
        grp_camera_layout.addWidget(btn_reset, 3, 0, 1, 2)
        layout.addWidget(grp_camera)

        # ---------------------------------------------------------------------
        # 5. Semantic Legend
        # ---------------------------------------------------------------------
        grp_legend = QtWidgets.QGroupBox("Semantic Legend")
        grp_legend_layout = QtWidgets.QVBoxLayout(grp_legend)
        grp_legend_layout.setSpacing(4)

        legend_items = [
            ("#888888", "Road Surface"),
            ("#f39c12", "Walls / Barriers"),
            ("#3498db", "Parked Vehicles"),
            ("#e67e22", "Pillars / Columns"),
            ("#2ecc71", "Kerbs / Speed Breakers"),
            ("#e74c3c", "Occupied Voxels"),
            ("#708090", "Unknown Space Voxels"),
            ("#00e676", "High AI Depth Confidence"),
            ("#00e5ff", "Vehicle Clearance Envelope"),
            ("#ff33aa", "Minimum Clearance Vector"),
        ]
        for color_hex, name in legend_items:
            row = QtWidgets.QHBoxLayout()
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: {color_hex}; border-radius: 3px;")
            label = QtWidgets.QLabel(name)
            label.setStyleSheet("font-size: 11px;")
            row.addWidget(swatch)
            row.addWidget(label)
            row.addStretch()
            grp_legend_layout.addLayout(row)

        layout.addWidget(grp_legend)
        layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # Internal Event Handlers
    # -------------------------------------------------------------------------
    def _on_vis_mode_changed(self, index: int):
        mode = self.combo_vis_mode.currentData()
        if hasattr(self, "grp_layers"):
            self.grp_layers.setEnabled(mode != "driver")
        if hasattr(self, "vis_hint"):
            if mode == "driver":
                self.vis_hint.setText("M7.2 Driver View: Clean photorealistic cockpit experience.")
            elif mode == "surface":
                self.vis_hint.setText("Continuous 3D surface mesh reconstruction.")
            elif mode == "pointcloud":
                self.vis_hint.setText("Point cloud debugging mode.")
            else:
                self.vis_hint.setText("Engineering View: Volumetric occupancy & telemetry.")
        self.vis_mode_changed.emit(mode)

    def _on_source_changed(self, index: int):
        source = self.combo_source.currentData()
        if source == "ground_truth":
            self.lbl_pipeline_badge.setText("[SIMULATION GROUND TRUTH]")
            self.lbl_pipeline_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 11px;"
            )
        elif source == "ai_calibrated":
            self.lbl_pipeline_badge.setText("[CALIBRATED AI (DEPTH ANYTHING V2)]")
            self.lbl_pipeline_badge.setStyleSheet(
                "background-color: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 11px;"
            )
        else:
            self.lbl_pipeline_badge.setText("[RAW AI (UNCALIBRATED)]")
            self.lbl_pipeline_badge.setStyleSheet(
                "background-color: #451a03; color: #fbbf24; border: 1px solid #d97706; "
                "border-radius: 4px; padding: 4px; font-weight: bold; font-size: 11px;"
            )
        self.source_changed.emit(source)

    def _on_frame_changed(self, index: int):
        frame_id = self.combo_frame.currentData()
        self.frame_changed.emit(frame_id)

    def _prev_frame(self):
        curr = self.combo_frame.currentIndex()
        if curr > 0:
            self.combo_frame.setCurrentIndex(curr - 1)

    def _next_frame(self):
        curr = self.combo_frame.currentIndex()
        if curr < self.combo_frame.count() - 1:
            self.combo_frame.setCurrentIndex(curr + 1)

    def _on_cloud_mode_changed(self, index: int):
        mode = self.combo_cloud.currentData()
        self.cloud_mode_changed.emit(mode)

    def _on_size_changed(self, val: int):
        size = val / 10.0
        self.lbl_size_val.setText(f"{size:.1f} px")
        self.point_size_changed.emit(size)

    def _on_occ_toggled(self):
        self.occupancy_toggled.emit(
            self.chk_occ_occupied.isChecked(),
            self.chk_occ_free.isChecked(),
            self.chk_occ_unknown.isChecked(),
        )

    def _on_scenario_changed(self, index: int):
        scen_id = self.combo_scenario.currentData()
        self.scenario_changed.emit(scen_id)

