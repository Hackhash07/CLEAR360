"""PySide6 editor for drawing depth regions over a reference photograph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets


class PhotoPolygonView(QtWidgets.QGraphicsView):
    point_clicked = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QtGui.QColor("#10131b"))
        self.setMinimumSize(720, 420)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._photo_rect = QtCore.QRectF()
        self._drawing_enabled = False
        self._current_points: list[QtCore.QPointF] = []
        self._point_items: list[QtWidgets.QGraphicsEllipseItem] = []
        self._line_items: list[QtWidgets.QGraphicsLineItem] = []
        self._region_items: list[QtWidgets.QGraphicsPolygonItem] = []

    def set_photo(self, path: str | Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"Could not open reference image: {path}")
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-100)
        self._photo_rect = QtCore.QRectF(pixmap.rect())
        self._scene.setSceneRect(self._photo_rect)
        self._current_points.clear()
        self._point_items.clear()
        self._line_items.clear()
        self._region_items.clear()
        self.fitInView(self._photo_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_drawing_enabled(self, enabled: bool) -> None:
        self._drawing_enabled = bool(enabled)
        self.setCursor(
            QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
        )

    def add_region_point(self, x: float, y: float) -> None:
        point = QtCore.QPointF(x, y)
        self._current_points.append(point)
        radius = max(3.0, min(self._photo_rect.width(), self._photo_rect.height()) * 0.004)
        dot = self._scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2,
            QtGui.QPen(QtGui.QColor("#ffffff"), 1.5),
            QtGui.QBrush(QtGui.QColor("#a855f7")),
        )
        dot.setZValue(20)
        self._point_items.append(dot)
        if len(self._current_points) > 1:
            previous = self._current_points[-2]
            line = self._scene.addLine(
                previous.x(), previous.y(), x, y,
                QtGui.QPen(QtGui.QColor("#a855f7"), max(2.0, radius * 0.7)),
            )
            line.setZValue(19)
            self._line_items.append(line)

    def finish_region(self, color: QtGui.QColor) -> list[list[float]]:
        points = [[float(p.x()), float(p.y())] for p in self._current_points]
        if len(points) >= 3:
            polygon = QtGui.QPolygonF(self._current_points)
            item = self._scene.addPolygon(
                polygon,
                QtGui.QPen(color, 2.5),
                QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 48)),
            )
            item.setZValue(10)
            self._region_items.append(item)
        self.clear_current_points()
        return points

    def undo_last_point(self) -> bool:
        if not self._current_points:
            return False
        self._current_points.pop()
        if self._point_items:
            self._scene.removeItem(self._point_items.pop())
        if self._line_items:
            self._scene.removeItem(self._line_items.pop())
        return True

    def clear_current_points(self) -> None:
        for item in self._point_items + self._line_items:
            self._scene.removeItem(item)
        self._current_points.clear()
        self._point_items.clear()
        self._line_items.clear()

    def clear_regions(self) -> None:
        self.clear_current_points()
        for item in self._region_items:
            self._scene.removeItem(item)
        self._region_items.clear()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing_enabled and event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self._photo_rect.contains(point):
                self.point_clicked.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if not self._photo_rect.isNull():
            self.fitInView(self._photo_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


class GuidedPhotoEditorDialog(QtWidgets.QDialog):
    """Let the user draw photo regions and assign forward depth offsets."""

    COLORS = [
        QtGui.QColor("#22d3ee"),
        QtGui.QColor("#f59e0b"),
        QtGui.QColor("#a78bfa"),
        QtGui.QColor("#fb7185"),
        QtGui.QColor("#4ade80"),
        QtGui.QColor("#f472b6"),
    ]

    def __init__(
        self,
        image_paths: list[Path],
        preferred_path: Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        if not image_paths:
            raise ValueError("Add at least one photograph before opening guided modelling.")
        self.setWindowTitle("CLEAR360 — Guided Photo Projection")
        self.resize(1080, 820)
        self.image_paths = [Path(p) for p in image_paths]
        self.regions: list[dict[str, Any]] = []
        self._drawing = False
        self._init_ui(preferred_path)

    def _init_ui(self, preferred_path: Path | None) -> None:
        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Choose a clear front-facing photograph. Click around a feature to outline it, "
            "then assign how far it should sit in front of the photo plane. "
            "The background photo remains the full-resolution texture."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        root.addWidget(intro)

        photo_row = QtWidgets.QHBoxLayout()
        photo_row.addWidget(QtWidgets.QLabel("Reference photo:"))
        self.photo_combo = QtWidgets.QComboBox()
        for path in self.image_paths:
            self.photo_combo.addItem(path.name, str(path))
        if preferred_path:
            found = self.photo_combo.findData(str(preferred_path))
            if found >= 0:
                self.photo_combo.setCurrentIndex(found)
        photo_row.addWidget(self.photo_combo, 1)
        root.addLayout(photo_row)

        self.view = PhotoPolygonView(self)
        root.addWidget(self.view, 1)

        edit_row = QtWidgets.QHBoxLayout()
        self.region_name = QtWidgets.QLineEdit()
        self.region_name.setPlaceholderText("Region name (e.g. fountain frame)")
        self.depth_spin = QtWidgets.QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 200.0)
        self.depth_spin.setDecimals(1)
        self.depth_spin.setSingleStep(1.0)
        self.depth_spin.setValue(8.0)
        self.depth_spin.setSuffix(" cm forward")
        self.btn_add_region = QtWidgets.QPushButton("＋ Draw region")
        self.btn_finish_region = QtWidgets.QPushButton("✓ Finish region")
        self.btn_undo = QtWidgets.QPushButton("↶ Undo point")
        edit_row.addWidget(self.region_name, 2)
        edit_row.addWidget(self.depth_spin)
        edit_row.addWidget(self.btn_add_region)
        edit_row.addWidget(self.btn_finish_region)
        edit_row.addWidget(self.btn_undo)
        root.addLayout(edit_row)

        self.region_list = QtWidgets.QListWidget()
        self.region_list.setMaximumHeight(90)
        root.addWidget(self.region_list)

        footer = QtWidgets.QHBoxLayout()
        self.btn_clear_regions = QtWidgets.QPushButton("Clear regions")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_build = QtWidgets.QPushButton("Build guided 2.5D model")
        self.btn_build.setDefault(True)
        self.btn_build.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 8px;"
        )
        footer.addWidget(self.btn_clear_regions)
        footer.addStretch(1)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_build)
        root.addLayout(footer)

        self.photo_combo.currentIndexChanged.connect(self._on_photo_changed)
        self.view.point_clicked.connect(self.view.add_region_point)
        self.btn_add_region.clicked.connect(self._begin_region)
        self.btn_finish_region.clicked.connect(self._finish_region)
        self.btn_undo.clicked.connect(self._undo_point)
        self.btn_clear_regions.clicked.connect(self._clear_regions)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_build.clicked.connect(self._accept_model)

        self.btn_finish_region.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self._load_photo()

    def _load_photo(self) -> None:
        path = Path(self.photo_combo.currentData())
        self.view.set_drawing_enabled(False)
        self.view.set_photo(path)
        self.regions.clear()
        self.region_list.clear()
        self._drawing = False

    def _on_photo_changed(self, _index: int) -> None:
        self._load_photo()

    def _begin_region(self) -> None:
        self.view.clear_current_points()
        self._drawing = True
        self.view.set_drawing_enabled(True)
        if not self.region_name.text().strip():
            self.region_name.setText(f"Region {len(self.regions) + 1}")
        self.btn_finish_region.setEnabled(True)
        self.btn_undo.setEnabled(True)

    def _finish_region(self) -> None:
        points = self.view.finish_region(self.COLORS[len(self.regions) % len(self.COLORS)])
        if len(points) < 3:
            QtWidgets.QMessageBox.information(
                self, "More points needed", "A region needs at least three points."
            )
            return
        name = self.region_name.text().strip() or f"Region {len(self.regions) + 1}"
        depth_cm = float(self.depth_spin.value())
        self.regions.append({"name": name, "points": points, "depth_cm": depth_cm})
        self.region_list.addItem(f"{name} — {depth_cm:.1f} cm forward")
        self.region_name.clear()
        self._drawing = False
        self.view.set_drawing_enabled(False)
        self.btn_finish_region.setEnabled(False)
        self.btn_undo.setEnabled(False)

    def _undo_point(self) -> None:
        if not self._drawing:
            return
        has_points = self.view.undo_last_point()
        self.btn_undo.setEnabled(has_points and bool(self.view._current_points))

    def _clear_regions(self) -> None:
        self.view.clear_regions()
        self.regions.clear()
        self.region_list.clear()
        self._drawing = False
        self.view.set_drawing_enabled(False)
        self.btn_finish_region.setEnabled(False)
        self.btn_undo.setEnabled(False)

    def _accept_model(self) -> None:
        if self._drawing:
            QtWidgets.QMessageBox.information(
                self, "Finish the current region", "Click 'Finish region' before building the model."
            )
            return
        if not self.regions:
            QtWidgets.QMessageBox.information(
                self, "Add a depth region", "Draw at least one region, such as the fountain or basin."
            )
            return
        self.accept()

    def selected_image_path(self) -> Path:
        return Path(self.photo_combo.currentData())

    def get_regions(self) -> list[dict[str, Any]]:
        return list(self.regions)
