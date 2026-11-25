"""Interactive synchronized image canvases, DualViewport, interactive Crop Overlay, and cancellable LoadingOverlay."""

from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...assets import asset_path
from ...assets.icons import get_icon
from ...core.io_utils import SUPPORTED_OPEN_EXTS, ensure_uint8, to_bgr
from ..theme import DARK_PALETTE, ThemePalette


_ASPECT_RATIOS = {
    "1:1 (Square)": 1.0,
    "4:3": 4.0 / 3.0,
    "16:9": 16.0 / 9.0,
    "3:2": 3.0 / 2.0,
    "4:5 (Portrait)": 4.0 / 5.0,
    "9:16 (Vertical)": 9.0 / 16.0,
    "21:9 (Cinema)": 21.0 / 9.0,
}


def draw_viewport_badge(
    painter: QPainter,
    text: str,
    viewport_width: int,
    palette: ThemePalette,
    *,
    align_right: bool = False,
) -> None:
    """Render a consistent viewport label badge across all comparison modes."""
    if not text:
        return
    font = QFont(painter.font())
    font.setPointSize(9)
    font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(font)
    fm = QFontMetrics(font)
    badge_w = fm.horizontalAdvance(text) + 16
    badge_h = 22
    margin = 10
    x = (viewport_width - badge_w - margin) if align_right else margin
    badge = QRectF(x, margin, badge_w, badge_h)

    bg = QColor(palette.bg_panel)
    bg.setAlpha(220)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(palette.border_strong), 1))
    painter.setBrush(bg)
    painter.drawRoundedRect(badge, 4, 4)
    painter.setPen(QColor(palette.text_secondary))
    painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)


def ndarray_to_qimage(img: np.ndarray) -> QImage:
    """Convert a uint8 Grayscale, BGR, or BGRA NumPy array into an owned QImage."""
    arr = ensure_uint8(img)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()
    if arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return qimg.copy()
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return qimg.copy()


class _SpinnerWidget(QWidget):
    """Lightweight 60fps rotating arc indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._angle = 0
        self._palette: ThemePalette = DARK_PALETTE

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self.update()

    def advance(self) -> None:
        self._angle = (self._angle + 10) % 360
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(2.5, 2.5, 17, 17)
        p.setPen(QPen(QColor(self._palette.border_strong), 2.3))
        p.drawEllipse(rect)
        pen = QPen(QColor(self._palette.accent), 2.3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, -self._angle * 16, 115 * 16)


class LoadingOverlay(QFrame):
    """Floating cancellable status pill shown over the viewport during heavy operations."""

    cancelClicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SurfaceCard")
        self._base_text = "Processing..."
        self._started_at = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(18, 12, 18, 12)
        root_lay.setSpacing(8)

        row_top = QHBoxLayout()
        row_top.setContentsMargins(0, 0, 0, 0)
        row_top.setSpacing(10)
        row_top.addStretch(1)

        self._spinner = _SpinnerWidget()
        row_top.addWidget(self._spinner)

        self._label = QLabel("Processing...")
        self._label.setStyleSheet("font-size: 11.5px; font-weight: 600;")
        row_top.addWidget(self._label)
        row_top.addStretch(1)
        root_lay.addLayout(row_top)

        row_btn = QHBoxLayout()
        row_btn.setContentsMargins(0, 0, 0, 0)
        row_btn.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMinimumWidth(92)
        self._cancel_btn.setFixedHeight(24)
        self._cancel_btn.setStyleSheet("padding: 2px 14px; font-size: 11px;")
        self._cancel_btn.clicked.connect(self.cancelClicked.emit)
        row_btn.addWidget(self._cancel_btn)
        row_btn.addStretch(1)
        root_lay.addLayout(row_btn)
        self.setVisible(False)

    def set_palette(self, palette: ThemePalette) -> None:
        self._spinner.set_palette(palette)

    def start(self, text: str = "Processing...") -> None:
        self._base_text = text
        self._started_at = time.monotonic()
        self._label.setText(text)
        self.adjustSize()
        self.setVisible(True)
        self.raise_()
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.setVisible(False)

    def _tick(self) -> None:
        self._spinner.advance()
        elapsed = time.monotonic() - self._started_at
        if elapsed >= 0.6:
            self._label.setText(f"{self._base_text} ({elapsed:.1f}s)")
            self.adjustSize()
            parent = self.parentWidget()
            if parent is not None and hasattr(parent, "_position_loading_overlay"):
                parent._position_loading_overlay()


class ImageCanvas(QWidget):
    """High-DPI zoomable, pannable image viewport with live pixel hover and interactive crop overlay."""

    viewChanged = pyqtSignal(float, QPointF)
    pixelHovered = pyqtSignal(int, int)
    cropChanged = pyqtSignal(float, float, float, float)  # x_pct, y_pct, w_pct, h_pct

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self._qimage: QImage | None = None
        self._zoom: float = 1.0
        self._fit_mode: bool = True
        self._pan_offset = QPointF(0.0, 0.0)
        self._dragging = False
        self._last_pos = QPointF()
        self._syncing = False
        self._palette: ThemePalette = DARK_PALETTE
        self._checker: QPixmap | None = None

        # Interactive crop state
        self._crop_active = False
        self._crop_box = [10.0, 10.0, 80.0, 80.0]  # x_pct, y_pct, w_pct, h_pct
        self._crop_aspect = "Free"
        self._crop_drag_mode: str = ""  # "", "move", "tl", "tr", "bl", "br"
        self._crop_drag_start = QPointF()
        self._crop_box_start = [10.0, 10.0, 80.0, 80.0]

        self.setMouseTracking(True)
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self._checker = None
        self.update()

    def set_image(self, img: np.ndarray | None, *, reset_view: bool = False) -> None:
        self._qimage = ndarray_to_qimage(img) if img is not None else None
        if reset_view:
            self._fit_mode = True
            self._pan_offset = QPointF(0.0, 0.0)
        self.update()

    def set_crop_overlay(
        self,
        active: bool,
        x_pct: float = 10.0,
        y_pct: float = 10.0,
        w_pct: float = 80.0,
        h_pct: float = 80.0,
        aspect: str = "Free",
    ) -> None:
        self._crop_active = active
        self._crop_aspect = aspect
        self._crop_box = [
            float(np.clip(x_pct, 0.0, 95.0)),
            float(np.clip(y_pct, 0.0, 95.0)),
            float(np.clip(w_pct, 5.0, 100.0)),
            float(np.clip(h_pct, 5.0, 100.0)),
        ]
        self.update()

    def current_zoom(self) -> float:
        if self._qimage is None:
            return 1.0
        return self._compute_fit_zoom() if self._fit_mode else self._zoom

    def fit_to_view(self, emit_sync: bool = True) -> None:
        self._fit_mode = True
        self._pan_offset = QPointF(0.0, 0.0)
        self._zoom = self._compute_fit_zoom()
        self.update()
        if emit_sync and not self._syncing:
            self.viewChanged.emit(self._zoom, self._pan_offset)

    def set_zoom_100(self, emit_sync: bool = True) -> None:
        self._fit_mode = False
        self._zoom = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self.update()
        if emit_sync and not self._syncing:
            self.viewChanged.emit(self._zoom, self._pan_offset)

    def step_zoom(self, factor: float) -> None:
        base = self.current_zoom()
        self._fit_mode = False
        self._zoom = max(0.05, min(32.0, base * factor))
        self.update()
        if not self._syncing:
            self.viewChanged.emit(self._zoom, self._pan_offset)

    def sync_view(self, zoom: float, pan_offset: QPointF, fit_mode: bool) -> None:
        self._syncing = True
        self._fit_mode = fit_mode
        self._zoom = zoom
        self._pan_offset = QPointF(pan_offset)
        self.update()
        self._syncing = False

    def _compute_fit_zoom(self) -> float:
        if self._qimage is None or self._qimage.isNull():
            return 1.0
        pad = 24.0
        avail_w = max(20.0, self.width() - pad)
        avail_h = max(20.0, self.height() - pad)
        return min(avail_w / self._qimage.width(), avail_h / self._qimage.height())

    def _image_rect(self) -> QRectF:
        if self._qimage is None or self._qimage.isNull():
            return QRectF()
        z = self._compute_fit_zoom() if self._fit_mode else self._zoom
        w = self._qimage.width() * z
        h = self._qimage.height() * z
        cx = self.width() / 2.0 + self._pan_offset.x()
        cy = self.height() / 2.0 + self._pan_offset.y()
        return QRectF(cx - w / 2.0, cy - h / 2.0, w, h)

    def _crop_screen_rect(self, img_rect: QRectF) -> QRectF:
        x_p, y_p, w_p, h_p = self._crop_box
        if self._crop_aspect in _ASPECT_RATIOS and self._qimage is not None and self._qimage.height() > 0:
            ratio = _ASPECT_RATIOS[self._crop_aspect]
            iw, ih = self._qimage.width(), self._qimage.height()
            cw_px = iw * (w_p / 100.0)
            ch_px = cw_px / ratio
            if ch_px > ih:
                ch_px = float(ih)
                cw_px = ch_px * ratio
            w_p = cw_px / iw * 100.0
            h_p = ch_px / ih * 100.0
        x = img_rect.left() + img_rect.width() * (x_p / 100.0)
        y = img_rect.top() + img_rect.height() * (y_p / 100.0)
        w = img_rect.width() * (min(100.0 - x_p, w_p) / 100.0)
        h = img_rect.height() * (min(100.0 - y_p, h_p) / 100.0)
        return QRectF(x, y, max(8.0, w), max(8.0, h))

    def _hit_crop_handle(self, pos: QPointF, crop_rect: QRectF) -> str:
        tol = 11.0
        corners = {
            "tl": crop_rect.topLeft(),
            "tr": crop_rect.topRight(),
            "bl": crop_rect.bottomLeft(),
            "br": crop_rect.bottomRight(),
        }
        for name, pt in corners.items():
            if (pos - pt).manhattanLength() <= tol * 1.4:
                return name
        if crop_rect.contains(pos):
            return "move"
        return ""

    def _ensure_checker(self) -> QPixmap:
        if self._checker is not None:
            return self._checker
        tile = 12
        pm = QPixmap(tile * 2, tile * 2)
        c1 = QColor("#1D2129") if self._palette.name == "dark" else QColor("#FFFFFF")
        c2 = QColor("#2B313D") if self._palette.name == "dark" else QColor("#D5DCE6")
        p = QPainter(pm)
        p.fillRect(0, 0, tile, tile, c1)
        p.fillRect(tile, 0, tile, tile, c2)
        p.fillRect(0, tile, tile, tile, c2)
        p.fillRect(tile, tile, tile, tile, c1)
        p.end()
        self._checker = pm
        return pm

    def _draw_crop_overlay(self, p: QPainter, img_rect: QRectF) -> None:
        if not self._crop_active or img_rect.isEmpty():
            return
        cr = self._crop_screen_rect(img_rect)
        p.save()
        scrim = QColor(0, 0, 0, 130)
        # 4 outer scrim rectangles around crop box
        p.fillRect(QRectF(img_rect.left(), img_rect.top(), img_rect.width(), max(0.0, cr.top() - img_rect.top())), scrim)
        p.fillRect(QRectF(img_rect.left(), cr.bottom(), img_rect.width(), max(0.0, img_rect.bottom() - cr.bottom())), scrim)
        p.fillRect(QRectF(img_rect.left(), cr.top(), max(0.0, cr.left() - img_rect.left()), cr.height()), scrim)
        p.fillRect(QRectF(cr.right(), cr.top(), max(0.0, img_rect.right() - cr.right()), cr.height()), scrim)

        # Rule of thirds guides
        p.setPen(QPen(QColor(255, 255, 255, 85), 1, Qt.PenStyle.DashLine))
        for frac in (1.0 / 3.0, 2.0 / 3.0):
            gx = cr.left() + cr.width() * frac
            gy = cr.top() + cr.height() * frac
            p.drawLine(QPointF(gx, cr.top()), QPointF(gx, cr.bottom()))
            p.drawLine(QPointF(cr.left(), gy), QPointF(cr.right(), gy))

        # Crisp border + corner handles
        p.setPen(QPen(QColor(self._palette.accent), 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(cr)

        p.setBrush(QColor("#FFFFFF"))
        p.setPen(QPen(QColor(self._palette.accent), 1.5))
        for pt in (cr.topLeft(), cr.topRight(), cr.bottomLeft(), cr.bottomRight()):
            p.drawRect(QRectF(pt.x() - 4.5, pt.y() - 4.5, 9, 9))
        p.restore()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette.bg_canvas))

        if self._qimage is None or self._qimage.isNull():
            return

        rect = self._image_rect()
        p.drawTiledPixmap(rect.toAlignedRect(), self._ensure_checker())

        z = self.current_zoom()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, z < 3.0)
        p.drawImage(rect, self._qimage)

        p.setPen(QPen(QColor(self._palette.border_strong), 1))
        p.drawRect(rect)

        self._draw_crop_overlay(p, rect)
        draw_viewport_badge(p, self.title, self.width(), self._palette, align_right=False)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._qimage is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        old_z = self.current_zoom()
        new_z = max(0.05, min(32.0, old_z * factor))
        if self._fit_mode:
            self._zoom = old_z
            self._fit_mode = False
        cursor = event.position()
        center = QPointF(self.width() / 2.0, self.height() / 2.0) + self._pan_offset
        rel = cursor - center
        scale_ratio = new_z / max(old_z, 1e-6)
        new_center = cursor - rel * scale_ratio
        self._pan_offset = new_center - QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom = new_z
        self.update()
        if not self._syncing:
            self.viewChanged.emit(self._zoom, self._pan_offset)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._crop_active and self._qimage is not None:
            rect = self._image_rect()
            cr = self._crop_screen_rect(rect)
            hit = self._hit_crop_handle(event.position(), cr)
            if hit:
                self._crop_drag_mode = hit
                self._crop_drag_start = event.position()
                self._crop_box_start = list(self._crop_box)
                return

        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._crop_drag_mode and self._qimage is not None:
            rect = self._image_rect()
            if rect.width() > 1 and rect.height() > 1:
                dx_pct = (pos.x() - self._crop_drag_start.x()) / rect.width() * 100.0
                dy_pct = (pos.y() - self._crop_drag_start.y()) / rect.height() * 100.0
                x0, y0, w0, h0 = self._crop_box_start
                if self._crop_drag_mode == "move":
                    nx = float(np.clip(x0 + dx_pct, 0.0, 100.0 - w0))
                    ny = float(np.clip(y0 + dy_pct, 0.0, 100.0 - h0))
                    self._crop_box = [round(nx, 1), round(ny, 1), round(w0, 1), round(h0, 1)]
                else:
                    nx, ny, nw, nh = x0, y0, w0, h0
                    if "l" in self._crop_drag_mode:
                        nx = float(np.clip(x0 + dx_pct, 0.0, x0 + w0 - 5.0))
                        nw = (x0 + w0) - nx
                    if "r" in self._crop_drag_mode:
                        nw = float(np.clip(w0 + dx_pct, 5.0, 100.0 - x0))
                    if "t" in self._crop_drag_mode:
                        ny = float(np.clip(y0 + dy_pct, 0.0, y0 + h0 - 5.0))
                        nh = (y0 + h0) - ny
                    if "b" in self._crop_drag_mode:
                        nh = float(np.clip(h0 + dy_pct, 5.0, 100.0 - y0))
                    self._crop_box = [round(nx, 1), round(ny, 1), round(nw, 1), round(nh, 1)]
                self.update()
                self.cropChanged.emit(*self._crop_box)
            return

        if self._dragging:
            if self._fit_mode:
                self._zoom = self._compute_fit_zoom()
                self._fit_mode = False
            delta = pos - self._last_pos
            self._last_pos = pos
            self._pan_offset += delta
            self.update()
            if not self._syncing:
                self.viewChanged.emit(self._zoom, self._pan_offset)

        if self._crop_active and self._qimage is not None and not self._dragging:
            hit = self._hit_crop_handle(pos, self._crop_screen_rect(self._image_rect()))
            if hit in ("tl", "br"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif hit in ("tr", "bl"):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif hit == "move":
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)

        if self._qimage is not None and not self._qimage.isNull():
            rect = self._image_rect()
            if rect.contains(pos) and rect.width() > 0 and rect.height() > 0:
                ix = int((pos.x() - rect.left()) / rect.width() * self._qimage.width())
                iy = int((pos.y() - rect.top()) / rect.height() * self._qimage.height())
                ix = max(0, min(self._qimage.width() - 1, ix))
                iy = max(0, min(self._qimage.height() - 1, iy))
                self.pixelHovered.emit(ix, iy)
            else:
                self.pixelHovered.emit(-1, -1)

    def mouseReleaseEvent(self, event) -> None:
        if self._crop_drag_mode:
            self._crop_drag_mode = ""
            return
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._crop_active:
            self.fit_to_view()


class SplitCompareCanvas(ImageCanvas):
    """Interactive draggable vertical split curtain comparing Original (left) vs Result (right)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(title="", parent=parent)
        self._orig_qimage: QImage | None = None
        self._split_ratio: float = 0.5
        self._dragging_split = False

    def set_pair(self, orig: np.ndarray | None, res: np.ndarray | None, *, reset_view: bool = False) -> None:
        self._orig_qimage = ndarray_to_qimage(orig) if orig is not None else None
        self._qimage = ndarray_to_qimage(res) if res is not None else None
        if reset_view:
            self._fit_mode = True
            self._pan_offset = QPointF(0.0, 0.0)
        self.update()

    def _split_x(self, rect: QRectF) -> float:
        return rect.left() + rect.width() * self._split_ratio

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self._palette.bg_canvas))
        if self._qimage is None or self._qimage.isNull():
            return

        rect = self._image_rect()
        p.drawTiledPixmap(rect.toAlignedRect(), self._ensure_checker())
        z = self.current_zoom()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, z < 3.0)

        split_x = self._split_x(rect)

        p.save()
        p.setClipRect(QRectF(split_x, rect.top() - 2, rect.right() - split_x + 2, rect.height() + 4))
        p.drawImage(rect, self._qimage)
        p.restore()

        if self._orig_qimage is not None and not self._orig_qimage.isNull():
            p.save()
            p.setClipRect(QRectF(rect.left() - 2, rect.top() - 2, split_x - rect.left() + 2, rect.height() + 4))
            p.drawImage(rect, self._orig_qimage)
            p.restore()

        p.setPen(QPen(QColor(self._palette.border_strong), 1))
        p.drawRect(rect)

        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(self._palette.accent), 2))
        p.drawLine(QPointF(split_x, rect.top()), QPointF(split_x, rect.bottom()))
        cy = rect.center().y()
        p.setBrush(QColor(self._palette.accent))
        p.setPen(QPen(QColor("#FFFFFF"), 1.5))
        p.drawEllipse(QPointF(split_x, cy), 9, 9)

        draw_viewport_badge(p, "ORIGINAL", self.width(), self._palette, align_right=False)
        draw_viewport_badge(p, "RESULT", self.width(), self._palette, align_right=True)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._qimage is not None:
            rect = self._image_rect()
            if abs(event.position().x() - self._split_x(rect)) <= 14:
                self._dragging_split = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_split and self._qimage is not None:
            rect = self._image_rect()
            if rect.width() > 1:
                self._split_ratio = max(0.02, min(0.98, (event.position().x() - rect.left()) / rect.width()))
                self.update()
            return
        if self._qimage is not None and not self._dragging:
            rect = self._image_rect()
            if abs(event.position().x() - self._split_x(rect)) <= 10:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_split:
            self._dragging_split = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        super().mouseReleaseEvent(event)


class DualViewport(QWidget):
    """Central image comparison workspace with Side-by-Side, Split Curtain, Diff Map, and Result Only."""

    fileDropped = pyqtSignal(str)
    openRequested = pyqtSignal()
    closeRequested = pyqtSignal()
    sampleRequested = pyqtSignal(str)
    pixelInspected = pyqtSignal(int, int, str)
    cropRectChanged = pyqtSignal(float, float, float, float)
    cancelProcessingRequested = pyqtSignal()

    MODE_SIDE = 0
    MODE_SPLIT = 1
    MODE_DIFF = 2
    MODE_RESULT = 3

    def __init__(self, sample_paths: list[Path] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._palette: ThemePalette = DARK_PALETTE
        self._orig: np.ndarray | None = None
        self._res: np.ndarray | None = None
        self._mode: int = self.MODE_SIDE

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = QFrame()
        self._header.setObjectName("HeaderBar")
        self._header.setFixedHeight(38)
        hbar = QHBoxLayout(self._header)
        hbar.setContentsMargins(10, 4, 10, 4)
        hbar.setSpacing(6)

        track = QFrame()
        track.setObjectName("SegmentedTrack")
        tlay = QHBoxLayout(track)
        tlay.setContentsMargins(2, 2, 2, 2)
        tlay.setSpacing(2)

        self._mode_group = QButtonGroup(self)
        self._mode_buttons: list[QPushButton] = []
        for idx, label in (
            (self.MODE_SIDE, "Side-by-Side"),
            (self.MODE_SPLIT, "Split Curtain"),
            (self.MODE_DIFF, "Diff Map"),
            (self.MODE_RESULT, "Result Only"),
        ):
            btn = QPushButton(label)
            btn.setObjectName("SegmentedBtn")
            btn.setCheckable(True)
            btn.setChecked(idx == self.MODE_SIDE)
            self._mode_group.addButton(btn, idx)
            self._mode_buttons.append(btn)
            tlay.addWidget(btn)
        self._mode_group.idClicked.connect(self.set_view_mode)
        hbar.addWidget(track)

        hbar.addStretch(1)

        self._dims_label = QLabel("No image loaded")
        self._dims_label.setObjectName("MutedLabel")
        hbar.addWidget(self._dims_label)

        hbar.addSpacing(8)
        zoom_track = QFrame()
        zoom_track.setObjectName("SegmentedTrack")
        zlay = QHBoxLayout(zoom_track)
        zlay.setContentsMargins(2, 2, 2, 2)
        zlay.setSpacing(1)

        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setObjectName("SegmentedBtn")
        self._zoom_out_btn.setFixedSize(26, 22)
        self._zoom_out_btn.setStyleSheet("padding: 0px; font-size: 13px; font-weight: 600;")
        self._zoom_out_btn.setToolTip("Zoom Out")
        self._zoom_out_btn.clicked.connect(lambda: self.step_zoom(1.0 / 1.25))
        zlay.addWidget(self._zoom_out_btn)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(40)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setObjectName("MutedLabel")
        self._zoom_label.setStyleSheet("font-size: 11px; font-weight: 600; background: transparent;")
        zlay.addWidget(self._zoom_label)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setObjectName("SegmentedBtn")
        self._zoom_in_btn.setFixedSize(26, 22)
        self._zoom_in_btn.setStyleSheet("padding: 0px; font-size: 13px; font-weight: 600;")
        self._zoom_in_btn.setToolTip("Zoom In")
        self._zoom_in_btn.clicked.connect(lambda: self.step_zoom(1.25))
        zlay.addWidget(self._zoom_in_btn)

        self._zoom_sep = QFrame()
        self._zoom_sep.setFixedSize(1, 14)
        zlay.addWidget(self._zoom_sep)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setObjectName("SegmentedBtn")
        self._fit_btn.setToolTip("Fit image to viewport (Ctrl+0)")
        self._fit_btn.clicked.connect(self.fit_view)
        zlay.addWidget(self._fit_btn)

        self._one_btn = QPushButton("1:1")
        self._one_btn.setObjectName("SegmentedBtn")
        self._one_btn.setToolTip("Actual pixels 100% (Ctrl+1)")
        self._one_btn.clicked.connect(self.zoom_100)
        zlay.addWidget(self._one_btn)

        hbar.addWidget(zoom_track)

        self._close_btn = QToolButton()
        self._close_btn.setObjectName("RowBtn")
        self._close_btn.setFixedSize(26, 26)
        self._close_btn.setToolTip("Close Image (Ctrl+W)")
        self._close_btn.setVisible(False)
        self._close_btn.clicked.connect(self.closeRequested.emit)
        hbar.addWidget(self._close_btn)

        root.addWidget(self._header)

        self._stack = QStackedLayout()
        self._stack_host = QWidget()
        self._stack_host.setLayout(self._stack)
        root.addWidget(self._stack_host, 1)

        self._welcome = self._build_welcome(sample_paths or [])
        self._stack.addWidget(self._welcome)

        side_page = QWidget()
        side_lay = QHBoxLayout(side_page)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(2)
        self.orig_canvas = ImageCanvas("ORIGINAL")
        self.res_canvas = ImageCanvas("RESULT")
        side_lay.addWidget(self.orig_canvas, 1)
        side_lay.addWidget(self.res_canvas, 1)
        self._stack.addWidget(side_page)

        self.split_canvas = SplitCompareCanvas()
        self._stack.addWidget(self.split_canvas)

        self.single_canvas = ImageCanvas("RESULT")
        self._stack.addWidget(self.single_canvas)

        # Floating interactive loading overlay
        self._loading_overlay = LoadingOverlay(self._stack_host)
        self._loading_overlay.cancelClicked.connect(self.cancelProcessingRequested.emit)

        self.orig_canvas.viewChanged.connect(self._on_orig_view_changed)
        self.res_canvas.viewChanged.connect(self._on_res_view_changed)
        self.orig_canvas.cropChanged.connect(self.cropRectChanged.emit)
        self.orig_canvas.pixelHovered.connect(lambda x, y: self.pixelInspected.emit(x, y, "orig"))
        self.res_canvas.pixelHovered.connect(lambda x, y: self.pixelInspected.emit(x, y, "res"))
        self.split_canvas.pixelHovered.connect(lambda x, y: self.pixelInspected.emit(x, y, "res"))
        self.single_canvas.pixelHovered.connect(lambda x, y: self.pixelInspected.emit(x, y, "res"))

        self._stack.setCurrentIndex(0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_loading_overlay()

    def _position_loading_overlay(self) -> None:
        if not hasattr(self, "_loading_overlay"):
            return
        host = self._stack_host
        ov = self._loading_overlay
        ov.adjustSize()
        # Center over the Result pane when Side-by-Side is visible
        if self._mode == self.MODE_SIDE and self._orig is not None:
            cx = int(host.width() * 0.75 - ov.width() / 2)
        else:
            cx = int(host.width() * 0.50 - ov.width() / 2)
        cy = int(host.height() * 0.50 - ov.height() / 2)
        ov.move(max(12, cx), max(12, cy))

    def _build_welcome(self, sample_paths: list[Path]) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("SurfaceCard")
        card.setMaximumWidth(560)
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(28, 26, 28, 26)
        card_lay.setSpacing(12)
        card_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_file = asset_path("dips.png")
        if icon_file.exists():
            pm = QPixmap(str(icon_file))
            if not pm.isNull():
                icon_lbl = QLabel()
                icon_lbl.setPixmap(
                    pm.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
                title_row.addWidget(icon_lbl)
        title = QLabel("DIPS")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        title_row.addWidget(title)
        card_lay.addLayout(title_row)

        subtitle = QLabel("Digital Image Processing Studio")
        subtitle.setObjectName("MutedLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px;")
        card_lay.addWidget(subtitle)

        sub = QLabel(
            "Drag and drop an image or .dips.json preset anywhere here,\n"
            "or open a file to begin non-destructive image processing."
        )
        sub.setObjectName("MutedLabel")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_lay.addWidget(sub)

        open_btn = QPushButton("Open Image File...")
        open_btn.setObjectName("PrimaryButton")
        open_btn.setMinimumHeight(34)
        open_btn.clicked.connect(self.openRequested.emit)
        card_lay.addWidget(open_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        if sample_paths:
            sample_lbl = QLabel("QUICK-LOAD SAMPLE IMAGES")
            sample_lbl.setObjectName("SectionTitle")
            sample_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_lay.addSpacing(8)
            card_lay.addWidget(sample_lbl)

            preferred = ["lena.png", "face.jpg", "coins.png", "rice.png", "cells.png", "pollen.png", "sudoku.png", "aort.png"]
            preferred_order = {name: idx for idx, name in enumerate(preferred)}
            ordered = sorted(
                sample_paths,
                key=lambda p: (preferred_order.get(p.name, len(preferred_order)), p.name.lower()),
            )

            for row_start in range(0, len(ordered), 4):
                row_lay = QHBoxLayout()
                row_lay.setSpacing(6)
                row_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                for sp in ordered[row_start : row_start + 4]:
                    chip = QPushButton(sp.name)
                    chip.setObjectName("ChipButton")
                    chip.clicked.connect(lambda _chk=False, p=str(sp): self.sampleRequested.emit(p))
                    row_lay.addWidget(chip)
                card_lay.addLayout(row_lay)

        lay.addWidget(card)
        return page

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        for c in (self.orig_canvas, self.res_canvas, self.split_canvas, self.single_canvas):
            c.set_palette(palette)
        self._loading_overlay.set_palette(palette)
        self._zoom_sep.setStyleSheet(f"background-color: {palette.border};")
        self._close_btn.setIcon(get_icon("close", palette.text_secondary, 15))

    def set_loading(self, active: bool, text: str = "Processing...") -> None:
        if active:
            self._loading_overlay.start(text)
            self._position_loading_overlay()
        else:
            self._loading_overlay.stop()

    def set_crop_overlay(
        self,
        active: bool,
        x_pct: float = 10.0,
        y_pct: float = 10.0,
        w_pct: float = 80.0,
        h_pct: float = 80.0,
        aspect: str = "Free",
    ) -> None:
        self.orig_canvas.set_crop_overlay(active, x_pct, y_pct, w_pct, h_pct, aspect)

    def set_images(self, orig: np.ndarray | None, res: np.ndarray | None, *, reset_view: bool = False) -> None:
        self._orig = orig
        self._res = res
        self._close_btn.setVisible(orig is not None)
        if orig is None:
            self._stack.setCurrentIndex(0)
            self._dims_label.setText("No image loaded")
            return

        oh, ow = orig.shape[:2]
        oc = 1 if orig.ndim == 2 else orig.shape[2]
        if res is not None:
            rh, rw = res.shape[:2]
            rc = 1 if res.ndim == 2 else res.shape[2]
            self._dims_label.setText(f"Orig: {ow}×{oh} ({oc}ch)  |  Result: {rw}×{rh} ({rc}ch)")
        else:
            self._dims_label.setText(f"{ow}×{oh} ({oc}ch)")

        self.orig_canvas.set_image(orig, reset_view=reset_view)
        self.res_canvas.set_image(res if res is not None else orig, reset_view=reset_view)
        self.split_canvas.set_pair(orig, res if res is not None else orig, reset_view=reset_view)
        self._refresh_single_canvas(reset_view=reset_view)
        if not reset_view and not self.orig_canvas._fit_mode:
            self._sync_from_orig(self.orig_canvas.current_zoom(), self.orig_canvas._pan_offset, False)
        self._apply_stack_page()
        self._update_zoom_label()
        self._position_loading_overlay()

    def _refresh_single_canvas(self, *, reset_view: bool = False) -> None:
        if self._orig is None:
            return
        res = self._res if self._res is not None else self._orig
        if self._mode == self.MODE_DIFF:
            self.single_canvas.title = "DIFFERENCE HEATMAP"
            o_bgr = to_bgr(self._orig)
            r_bgr = to_bgr(res)
            if o_bgr.shape[:2] != r_bgr.shape[:2]:
                r_bgr = cv2.resize(r_bgr, (o_bgr.shape[1], o_bgr.shape[0]))
            diff = cv2.absdiff(o_bgr, r_bgr)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            heat = cv2.applyColorMap(diff_gray, cv2.COLORMAP_TURBO)
            self.single_canvas.set_image(heat, reset_view=reset_view)
        else:
            self.single_canvas.title = "RESULT"
            self.single_canvas.set_image(res, reset_view=reset_view)

    def _apply_stack_page(self) -> None:
        if self._orig is None:
            self._stack.setCurrentIndex(0)
        elif self._mode == self.MODE_SIDE:
            self._stack.setCurrentIndex(1)
        elif self._mode == self.MODE_SPLIT:
            self._stack.setCurrentIndex(2)
        else:
            self._stack.setCurrentIndex(3)

    def set_view_mode(self, mode: int) -> None:
        self._mode = mode
        if 0 <= mode < len(self._mode_buttons):
            self._mode_buttons[mode].setChecked(True)
        self._refresh_single_canvas()
        self._apply_stack_page()
        self._update_zoom_label()
        self._position_loading_overlay()

    def _res_to_orig_scale(self) -> float:
        """Ratio mapping Original zoom to Result zoom so differently sized images compare at equal visual scale."""
        if self._orig is None:
            return 1.0
        res = self._res if self._res is not None else self._orig
        ow = float(self._orig.shape[1])
        rw = float(res.shape[1])
        if ow > 0 and rw > 0:
            return ow / rw
        return 1.0

    def _sync_from_orig(self, orig_zoom: float, pan: QPointF, fit_mode: bool) -> None:
        scale = self._res_to_orig_scale()
        res_zoom = orig_zoom * scale
        self.res_canvas.sync_view(res_zoom, pan, fit_mode)
        self.split_canvas.sync_view(res_zoom, pan, fit_mode)
        diff_mode = self._mode == self.MODE_DIFF
        self.single_canvas.sync_view(orig_zoom if diff_mode else res_zoom, pan, fit_mode)
        self._update_zoom_label()

    def fit_view(self) -> None:
        for c in (self.orig_canvas, self.res_canvas, self.split_canvas, self.single_canvas):
            c.fit_to_view(emit_sync=False)
        self._update_zoom_label()

    def zoom_100(self) -> None:
        self.orig_canvas.set_zoom_100(emit_sync=False)
        self._sync_from_orig(1.0, QPointF(0.0, 0.0), False)

    def step_zoom(self, factor: float) -> None:
        active = self._active_canvas()
        active.step_zoom(factor)
        z = active.current_zoom()
        pan = active._pan_offset
        is_orig_space = (active is self.orig_canvas) or (active is self.single_canvas and self._mode == self.MODE_DIFF)
        scale = self._res_to_orig_scale()
        orig_zoom = z if is_orig_space else (z / max(scale, 1e-6))
        self.orig_canvas.sync_view(orig_zoom, pan, False)
        self._sync_from_orig(orig_zoom, pan, False)

    def _active_canvas(self) -> ImageCanvas:
        if self._mode == self.MODE_SPLIT:
            return self.split_canvas
        if self._mode in (self.MODE_DIFF, self.MODE_RESULT):
            return self.single_canvas
        return self.res_canvas

    def _on_orig_view_changed(self, zoom: float, pan: QPointF) -> None:
        self._sync_from_orig(zoom, pan, self.orig_canvas._fit_mode)

    def _on_res_view_changed(self, zoom: float, pan: QPointF) -> None:
        scale = self._res_to_orig_scale()
        orig_zoom = zoom / max(scale, 1e-6)
        fit_mode = self.res_canvas._fit_mode
        self.orig_canvas.sync_view(orig_zoom, pan, fit_mode)
        self.split_canvas.sync_view(zoom, pan, fit_mode)
        self.single_canvas.sync_view(orig_zoom if self._mode == self.MODE_DIFF else zoom, pan, fit_mode)
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        pct = int(round(self.orig_canvas.current_zoom() * 100))
        self._zoom_label.setText(f"{pct}%")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    p = Path(url.toLocalFile())
                    if p.suffix.lower() in SUPPORTED_OPEN_EXTS or p.name.lower().endswith(".json"):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.fileDropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
