"""Export dialog with format selection, encoder quality, and optional Original vs Result comparison export."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from ...core.io_utils import SUPPORTED_SAVE_FORMATS, save_image, to_bgr
from ..theme import configure_combobox_popup


class ExportDialog(QDialog):
    """Export processed result or side-by-side comparison sheet."""

    def __init__(
        self,
        orig: np.ndarray,
        result: np.ndarray,
        suggested_name: str = "processed.png",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Image - DIP Studio")
        self.setMinimumWidth(440)
        self._orig = orig
        self._result = result
        self.exported_path: Path | None = None

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit(str(Path.home() / suggested_name))
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_btn)
        form.addRow("Output File:", path_row)

        self._format_combo = configure_combobox_popup(QComboBox())
        self._format_combo.addItems(list(SUPPORTED_SAVE_FORMATS.keys()))
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._format_combo)

        q_row = QHBoxLayout()
        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(1, 100)
        self._quality_slider.setValue(95)
        self._quality_lbl = QLabel("95")
        self._quality_slider.valueChanged.connect(lambda v: self._quality_lbl.setText(str(v)))
        q_row.addWidget(self._quality_slider, 1)
        q_row.addWidget(self._quality_lbl)
        form.addRow("Quality / Level:", q_row)

        self._sbs_chk = QCheckBox("Export Original vs Result Side-by-Side Comparison")
        form.addRow("", self._sbs_chk)

        lay.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._do_export)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._on_format_changed("PNG")

    def _on_format_changed(self, fmt: str) -> None:
        ext = SUPPORTED_SAVE_FORMATS[fmt]
        cur = Path(self._path_edit.text().strip() or "output.png")
        self._path_edit.setText(str(cur.with_suffix(ext)))
        if fmt == "PNG":
            self._quality_slider.setRange(0, 9)
            self._quality_slider.setValue(3)
        elif fmt in ("JPEG", "WebP"):
            self._quality_slider.setRange(1, 100)
            self._quality_slider.setValue(95)
        self._quality_slider.setEnabled(fmt in ("PNG", "JPEG", "WebP"))

    def _browse(self) -> None:
        fmt = self._format_combo.currentText()
        ext = SUPPORTED_SAVE_FORMATS[fmt]
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed Image",
            self._path_edit.text(),
            f"{fmt} (*{ext});;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _do_export(self) -> None:
        out = Path(self._path_edit.text().strip())
        img = self._result
        if self._sbs_chk.isChecked():
            o_bgr = to_bgr(self._orig)
            r_bgr = to_bgr(self._result)
            if o_bgr.shape[0] != r_bgr.shape[0]:
                scale = o_bgr.shape[0] / max(r_bgr.shape[0], 1)
                r_bgr = cv2.resize(r_bgr, (max(1, int(round(r_bgr.shape[1] * scale))), o_bgr.shape[0]))
            sep = np.full((o_bgr.shape[0], 4, 3), 40, dtype=np.uint8)
            img = np.hstack([o_bgr, sep, r_bgr])

        fmt = self._format_combo.currentText()
        val = self._quality_slider.value()
        self.exported_path = save_image(
            img,
            out,
            jpeg_quality=val if fmt == "JPEG" else 95,
            png_compression=val if fmt == "PNG" else 3,
            webp_quality=val if fmt == "WebP" else 92,
        )
        self.accept()
