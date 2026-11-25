"""Specialized visualizers for Transfer Curves, Custom Kernels, Structuring Elements, and Affine Matrices."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.ops_morphology import build_structuring_element
from ...core.ops_spatial import CUSTOM_KERNEL_PRESETS
from ..theme import DARK_PALETTE, ThemePalette


class TransferCurveWidget(QWidget):
    """Plots the 256-entry intensity mapping s = T(r) against the y = x identity diagonal."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette: ThemePalette = DARK_PALETTE
        self._lut: np.ndarray | None = None
        self.setFixedHeight(90)

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self.update()

    def set_lut(self, lut: np.ndarray | None) -> None:
        self._lut = lut
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.fillRect(r, QColor(self._palette.bg_canvas))
        p.setPen(QPen(QColor(self._palette.border), 1))
        p.drawRect(r)

        # Keep square aspect centered inside widget
        side = min(r.width() - 24, r.height() - 16)
        plot = QRectF(
            r.center().x() - side / 2.0,
            r.center().y() - side / 2.0,
            side,
            side,
        )
        p.setPen(QPen(QColor(self._palette.border), 1, Qt.PenStyle.DotLine))
        p.drawRect(plot)
        p.drawLine(plot.bottomLeft(), plot.topRight())

        if self._lut is not None and len(self._lut) == 256:
            path = QPainterPath()
            for i in range(256):
                x = plot.left() + (i / 255.0) * plot.width()
                y = plot.bottom() - (float(self._lut[i]) / 255.0) * plot.height()
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(QColor(self._palette.accent), 2.0))
            p.drawPath(path)

        p.setPen(QColor(self._palette.text_muted))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(QRectF(plot.right() + 6, plot.top(), 60, 16), Qt.AlignmentFlag.AlignLeft, "s = T(r)")


class KernelMatrixWidget(QWidget):
    """Interactive 3x3 or 5x5 convolution matrix editor with classic presets."""

    kernelChanged = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = 3
        self._updating = False
        self._cells: list[list[QDoubleSpinBox]] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Presets...")
        for name in CUSTOM_KERNEL_PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        bar.addWidget(self._preset_combo, 1)

        self._size_btn = QPushButton("Switch 5×5")
        self._size_btn.clicked.connect(self._toggle_size)
        bar.addWidget(self._size_btn)
        lay.addLayout(bar)

        self._grid_host = QFrame()
        self._grid_host.setObjectName("SurfaceCard")
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setSpacing(4)
        lay.addWidget(self._grid_host)

        self.set_kernel(CUSTOM_KERNEL_PRESETS["Sharpen"])

    def _rebuild_grid(self, size: int) -> None:
        self._size = size
        self._size_btn.setText("Switch 3×3" if size == 5 else "Switch 5×5")
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cells = []
        for r in range(size):
            row_cells: list[QDoubleSpinBox] = []
            for c in range(size):
                sb = QDoubleSpinBox()
                sb.setRange(-99.0, 99.0)
                sb.setDecimals(2)
                sb.setSingleStep(0.5)
                sb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                sb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sb.setMinimumWidth(42)
                sb.valueChanged.connect(self._emit_kernel)
                self._grid.addWidget(sb, r, c)
                row_cells.append(sb)
            self._cells.append(row_cells)

    def set_kernel(self, matrix: list[list[float]]) -> None:
        size = 5 if len(matrix) == 5 else 3
        self._updating = True
        if size != self._size or not self._cells:
            self._rebuild_grid(size)
        for r in range(size):
            for c in range(size):
                val = float(matrix[r][c]) if r < len(matrix) and c < len(matrix[r]) else 0.0
                self._cells[r][c].setValue(val)
        self._updating = False

    def kernel(self) -> list[list[float]]:
        return [[round(cell.value(), 4) for cell in row] for row in self._cells]

    def _emit_kernel(self) -> None:
        if not self._updating:
            self.kernelChanged.emit(self.kernel())

    def _apply_preset(self, name: str) -> None:
        if name in CUSTOM_KERNEL_PRESETS:
            self.set_kernel(CUSTOM_KERNEL_PRESETS[name])
            self.kernelChanged.emit(self.kernel())

    def _toggle_size(self) -> None:
        new_size = 5 if self._size == 3 else 3
        mat = [[0.0] * new_size for _ in range(new_size)]
        mat[new_size // 2][new_size // 2] = 1.0
        self.set_kernel(mat)
        self.kernelChanged.emit(self.kernel())


class StructuringElementPreview(QWidget):
    """Draws the binary Structuring Element grid (Rect / Ellipse / Cross)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette: ThemePalette = DARK_PALETTE
        self._se: np.ndarray = build_structuring_element("Rectangle", 5, 5)
        self._label = "Rectangle 5×5"
        self.setFixedHeight(92)

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self.update()

    def set_se(self, shape: str, w: int, h: int) -> None:
        self._se = build_structuring_element(shape, w, h)
        self._label = f"{shape} {self._se.shape[1]}×{self._se.shape[0]}"
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.fillRect(r, QColor(self._palette.bg_canvas))
        p.setPen(QPen(QColor(self._palette.border), 1))
        p.drawRect(r)

        rows, cols = self._se.shape
        avail = min(r.height() - 16, 76.0)
        cell = max(2.0, min(14.0, avail / max(rows, cols, 1)))
        grid_w = cols * cell
        grid_h = rows * cell
        x0 = r.left() + 14.0
        y0 = r.center().y() - grid_h / 2.0

        acc = QColor(self._palette.accent)
        off = QColor(self._palette.bg_elevated)
        border = QColor(self._palette.border_strong)
        for rr in range(rows):
            for cc in range(cols):
                cell_rect = QRectF(x0 + cc * cell, y0 + rr * cell, cell - 1, cell - 1)
                p.fillRect(cell_rect, acc if self._se[rr, cc] else off)
                if cell >= 5:
                    p.setPen(QPen(border, 0.5))
                    p.drawRect(cell_rect)

        p.setPen(QColor(self._palette.text_secondary))
        p.drawText(
            QRectF(x0 + grid_w + 14, r.top(), r.width() - grid_w - 22, r.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"SE: {self._label}",
        )


class AffineMatrixView(QFrame):
    """Displays the live 2x3 affine transformation matrix returned by the core engine."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SurfaceCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        hdr = QLabel("2×3 Affine Matrix")
        hdr.setObjectName("MutedLabel")
        lay.addWidget(hdr)
        self._lbl = QLabel("[ 1.0000   0.0000   0.0000 ]\n[ 0.0000   1.0000   0.0000 ]")
        self._lbl.setStyleSheet("font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 11px;")
        lay.addWidget(self._lbl)

    def set_matrix(self, matrix: list[list[float]] | None) -> None:
        if not matrix or len(matrix) != 2:
            return
        rows = [
            f"[ {r[0]:>8.4f}  {r[1]:>8.4f}  {r[2]:>8.2f} ]"
            for r in matrix
        ]
        self._lbl.setText("\n".join(rows))
