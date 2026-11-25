"""High-speed QPainter live histogram widget with Original/Result/Overlay, Log scale, CDF, and Stats."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.ops_histogram import compute_histogram, compute_quality_metrics, compute_stats
from ...core.types import ImageStats
from ..theme import DARK_PALETTE, ThemePalette, configure_combobox_popup


class _HistogramPlot(QWidget):
    """Canvas rendering 256-bin histogram bars/curves and optional CDF overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette: ThemePalette = DARK_PALETTE
        self._orig_hist: dict[str, np.ndarray] | None = None
        self._res_hist: dict[str, np.ndarray] | None = None
        self._source_mode: str = "Result"      # "Original" | "Result" | "Overlay"
        self._channel_mode: str = "Auto"       # "Auto" | "Lum" | "R" | "G" | "B"
        self._log_scale: bool = False
        self._show_cdf: bool = False
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self.update()

    def set_data(
        self,
        orig_hist: dict[str, np.ndarray] | None,
        res_hist: dict[str, np.ndarray] | None,
        source_mode: str,
        channel_mode: str,
        log_scale: bool,
        show_cdf: bool,
    ) -> None:
        self._orig_hist = orig_hist
        self._res_hist = res_hist
        self._source_mode = source_mode
        self._channel_mode = channel_mode
        self._log_scale = log_scale
        self._show_cdf = show_cdf
        self.update()

    def _transform_vals(self, arr: np.ndarray) -> np.ndarray:
        vals = arr.astype(np.float64)
        if self._log_scale:
            vals = np.log1p(vals)
        return vals

    def _draw_filled_curve(
        self,
        p: QPainter,
        rect: QRectF,
        vals: np.ndarray,
        max_val: float,
        stroke: QColor,
        fill: QColor,
    ) -> None:
        if max_val <= 0:
            return
        w, h = rect.width(), rect.height()
        path = QPainterPath()
        path.moveTo(rect.left(), rect.bottom())
        for i in range(256):
            x = rect.left() + (i / 255.0) * w
            y = rect.bottom() - (vals[i] / max_val) * (h - 4.0)
            path.lineTo(x, y)
        path.lineTo(rect.right(), rect.bottom())
        path.closeSubpath()
        p.fillPath(path, fill)

        line_path = QPainterPath()
        for i in range(256):
            x = rect.left() + (i / 255.0) * w
            y = rect.bottom() - (vals[i] / max_val) * (h - 4.0)
            if i == 0:
                line_path.moveTo(x, y)
            else:
                line_path.lineTo(x, y)
        p.setPen(QPen(stroke, 1.3))
        p.drawPath(line_path)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(r, 6.0, 6.0)
        p.fillPath(clip, QColor(self._palette.bg_canvas))
        p.save()
        p.setClipPath(clip)

        show_axis = r.height() >= 76.0
        plot = r.adjusted(4, 5, -4, -14 if show_axis else -4)
        grid_pen = QPen(QColor(self._palette.border), 1, Qt.PenStyle.DotLine)
        p.setPen(grid_pen)
        for q in (0.25, 0.50, 0.75):
            gx = plot.left() + plot.width() * q
            p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))

        active_hist = self._orig_hist if self._source_mode == "Original" else self._res_hist
        if active_hist is None:
            p.restore()
            p.setPen(QColor(self._palette.text_muted))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No image loaded")
            p.setPen(QPen(QColor(self._palette.border), 1))
            p.drawRoundedRect(r, 6.0, 6.0)
            return

        if self._source_mode == "Overlay" and self._orig_hist is not None and self._res_hist is not None:
            o_vals = self._transform_vals(self._orig_hist["lum"])
            r_vals = self._transform_vals(self._res_hist["lum"])
            mx = max(float(o_vals.max()), float(r_vals.max()), 1.0)
            self._draw_filled_curve(p, plot, o_vals, mx, QColor("#8B95A5"), QColor(139, 149, 165, 45))
            acc = QColor(self._palette.accent)
            acc_fill = QColor(acc)
            acc_fill.setAlpha(75)
            self._draw_filled_curve(p, plot, r_vals, mx, acc, acc_fill)
        else:
            ch = self._channel_mode
            if ch == "Auto":
                if "r" in active_hist:
                    rv = self._transform_vals(active_hist["r"])
                    gv = self._transform_vals(active_hist["g"])
                    bv = self._transform_vals(active_hist["b"])
                    mx = max(float(rv.max()), float(gv.max()), float(bv.max()), 1.0)
                    self._draw_filled_curve(p, plot, rv, mx, QColor("#E05A5A"), QColor(224, 90, 90, 42))
                    self._draw_filled_curve(p, plot, gv, mx, QColor("#53B877"), QColor(83, 184, 119, 42))
                    self._draw_filled_curve(p, plot, bv, mx, QColor("#5493F7"), QColor(84, 147, 247, 42))
                else:
                    lv = self._transform_vals(active_hist["lum"])
                    acc = QColor(self._palette.accent)
                    fill = QColor(acc)
                    fill.setAlpha(85)
                    self._draw_filled_curve(p, plot, lv, max(float(lv.max()), 1.0), acc, fill)
            else:
                key = {"Lum": "lum", "R": "r", "G": "g", "B": "b"}.get(ch, "lum")
                arr = active_hist.get(key, active_hist["lum"])
                vals = self._transform_vals(arr)
                color_map = {
                    "lum": self._palette.accent,
                    "r": "#E05A5A",
                    "g": "#53B877",
                    "b": "#5493F7",
                }
                stroke = QColor(color_map.get(key, self._palette.accent))
                fill = QColor(stroke)
                fill.setAlpha(78)
                self._draw_filled_curve(p, plot, vals, max(float(vals.max()), 1.0), stroke, fill)

        if self._show_cdf:
            lum = active_hist["lum"].astype(np.float64)
            cdf = np.cumsum(lum)
            if cdf[-1] > 0:
                cdf /= cdf[-1]
                cdf_path = QPainterPath()
                for i in range(256):
                    x = plot.left() + (i / 255.0) * plot.width()
                    y = plot.bottom() - cdf[i] * (plot.height() - 4.0)
                    if i == 0:
                        cdf_path.moveTo(x, y)
                    else:
                        cdf_path.lineTo(x, y)
                p.setPen(QPen(QColor("#E6B84C"), 1.3, Qt.PenStyle.DashLine))
                p.drawPath(cdf_path)

        if show_axis:
            p.setPen(QColor(self._palette.text_muted))
            f = p.font()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(QRectF(plot.left(), plot.bottom() + 1, 24, 12), Qt.AlignmentFlag.AlignLeft, "0")
            p.drawText(QRectF(plot.center().x() - 14, plot.bottom() + 1, 28, 12), Qt.AlignmentFlag.AlignCenter, "128")
            p.drawText(QRectF(plot.right() - 24, plot.bottom() + 1, 24, 12), Qt.AlignmentFlag.AlignRight, "255")

        p.restore()
        p.setPen(QPen(QColor(self._palette.border), 1))
        p.drawRoundedRect(r, 6.0, 6.0)


class HistogramPanel(QWidget):
    """Responsive Histogram & Image Statistics inspector panel."""

    _STAT_KEYS = (
        ("mean", "MEAN"),
        ("std", "STD"),
        ("range", "RANGE"),
        ("median", "MEDIAN"),
        ("entropy", "ENTROPY"),
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._orig_hist: dict[str, np.ndarray] | None = None
        self._res_hist: dict[str, np.ndarray] | None = None
        self._orig_stats: ImageStats | None = None
        self._res_stats: ImageStats | None = None
        self._stat_cols: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(5)

        # Row 1: Title + Log / CDF pills (always fits down to ~180px width)
        top = QHBoxLayout()
        top.setSpacing(4)
        title = QLabel("HISTOGRAM")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        top.addStretch(1)

        self._log_chk = QPushButton("Log")
        self._log_chk.setObjectName("SegmentedBtn")
        self._log_chk.setCheckable(True)
        self._log_chk.setFixedHeight(20)
        self._log_chk.setToolTip("Logarithmic intensity scale")
        self._log_chk.toggled.connect(self._refresh_plot)
        top.addWidget(self._log_chk)

        self._cdf_chk = QPushButton("CDF")
        self._cdf_chk.setObjectName("SegmentedBtn")
        self._cdf_chk.setCheckable(True)
        self._cdf_chk.setFixedHeight(20)
        self._cdf_chk.setToolTip("Overlay cumulative distribution curve")
        self._cdf_chk.toggled.connect(self._refresh_plot)
        top.addWidget(self._cdf_chk)
        lay.addLayout(top)

        # Row 2: Equal-width Source + Channel selectors
        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._source_combo = configure_combobox_popup(QComboBox())
        self._source_combo.addItems(["Result", "Original", "Overlay"])
        self._source_combo.setMinimumWidth(0)
        self._source_combo.currentTextChanged.connect(self._refresh_plot)
        controls.addWidget(self._source_combo, 1)

        self._chan_combo = configure_combobox_popup(QComboBox())
        self._chan_combo.addItems(["Auto", "Lum", "R", "G", "B"])
        self._chan_combo.setMinimumWidth(0)
        self._chan_combo.currentTextChanged.connect(self._refresh_plot)
        controls.addWidget(self._chan_combo, 1)
        lay.addLayout(controls)

        self._plot = _HistogramPlot()
        lay.addWidget(self._plot, 1)

        # Responsive stat labels without background card container
        self._stats_frame = QWidget()
        self._stats_frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._stats_grid = QGridLayout(self._stats_frame)
        self._stats_grid.setContentsMargins(2, 2, 2, 0)
        self._stats_grid.setHorizontalSpacing(10)
        self._stats_grid.setVerticalSpacing(5)

        self._stat_cells: list[QWidget] = []
        self._stat_labels: dict[str, QLabel] = {}
        for k, caption in self._STAT_KEYS:
            cell = QWidget()
            c_lay = QVBoxLayout(cell)
            c_lay.setContentsMargins(0, 0, 0, 0)
            c_lay.setSpacing(1)
            cap = QLabel(caption)
            cap.setObjectName("MutedLabel")
            cap.setStyleSheet("font-size: 9px; letter-spacing: 0.4px;")
            val_lbl = QLabel("-")
            val_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
            c_lay.addWidget(cap)
            c_lay.addWidget(val_lbl)
            self._stat_cells.append(cell)
            self._stat_labels[k] = val_lbl
        lay.addWidget(self._stats_frame)
        self._reflow_stats(4)
        self._quality_metrics: dict[str, float] | None = None

    def _reflow_stats(self, cols: int) -> None:
        if cols == self._stat_cols:
            return
        self._stat_cols = cols
        while self._stats_grid.count():
            self._stats_grid.takeAt(0)
        for idx, cell in enumerate(self._stat_cells):
            self._stats_grid.addWidget(cell, idx // cols, idx % cols)
        for c in range(4):
            self._stats_grid.setColumnStretch(c, 1 if c < cols else 0)
        self._stats_grid.activate()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = self.width()
        cols = 2 if w < 250 else (3 if w < 305 else 4)
        self._reflow_stats(cols)
        need_h = 62 + self._plot.minimumHeight() + self._stats_frame.sizeHint().height()
        self._stats_frame.setVisible(self.height() >= need_h)

    def set_palette(self, palette: ThemePalette) -> None:
        self._plot.set_palette(palette)

    def update_images(self, orig: np.ndarray | None, res: np.ndarray | None) -> None:
        if orig is None:
            self._orig_hist = None
            self._res_hist = None
            self._orig_stats = None
            self._res_stats = None
            self._quality_metrics = None
        else:
            self._orig_hist = compute_histogram(orig)
            self._orig_stats = compute_stats(orig)
            if res is not None:
                self._res_hist = compute_histogram(res)
                self._res_stats = compute_stats(res)
                self._quality_metrics = compute_quality_metrics(orig, res)
            else:
                self._res_hist = self._orig_hist
                self._res_stats = self._orig_stats
                self._quality_metrics = None
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        src = self._source_combo.currentText()
        ch = self._chan_combo.currentText()
        self._plot.set_data(
            self._orig_hist,
            self._res_hist,
            src,
            ch,
            self._log_chk.isChecked(),
            self._cdf_chk.isChecked(),
        )
        st = self._orig_stats if src == "Original" else self._res_stats
        if st is None:
            for lbl in self._stat_labels.values():
                lbl.setText("-")
        else:
            self._stat_labels["mean"].setText(f"{st.mean:.1f}")
            self._stat_labels["std"].setText(f"{st.std:.1f}")
            self._stat_labels["range"].setText(f"{st.min_val}-{st.max_val}")
            self._stat_labels["median"].setText(f"{st.median:.0f}")
            self._stat_labels["entropy"].setText(f"{st.entropy:.2f}")
            qm = self._quality_metrics
            if qm is None:
                self._stat_labels["psnr"].setText("-")
                self._stat_labels["ssim"].setText("-")
            else:
                psnr_txt = "∞" if qm["psnr"] >= 99.9 else f"{qm['psnr']:.1f} dB"
                self._stat_labels["psnr"].setText(psnr_txt)
                self._stat_labels["ssim"].setText(f"{qm['ssim']:.3f}")
