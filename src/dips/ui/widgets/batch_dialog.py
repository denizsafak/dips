"""Batch folder processing dialog applying the current non-destructive pipeline."""

from __future__ import annotations

from pathlib import Path
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from ...core.io_utils import SUPPORTED_SAVE_FORMATS
from ...core.pipeline import ImagePipeline


class BatchDialog(QDialog):
    """Apply active pipeline across every supported image in a chosen folder."""

    def __init__(self, pipeline: ImagePipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Process Folder - DIP Studio")
        self.setMinimumWidth(460)
        self._pipeline = pipeline

        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        in_row = QHBoxLayout()
        self._in_edit = QLineEdit()
        in_btn = QPushButton("Browse...")
        in_btn.clicked.connect(lambda: self._pick_dir(self._in_edit))
        in_row.addWidget(self._in_edit, 1)
        in_row.addWidget(in_btn)
        form.addRow("Input Folder:", in_row)

        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(lambda: self._pick_dir(self._out_edit))
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_btn)
        form.addRow("Output Folder:", out_row)

        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(list(SUPPORTED_SAVE_FORMATS.keys()))
        form.addRow("Output Format:", self._fmt_combo)
        lay.addLayout(form)

        self._status = QLabel(f"Pipeline steps: {len(pipeline.steps)}")
        self._status.setObjectName("MutedLabel")
        lay.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        lay.addWidget(self._progress)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.clicked.connect(self._run_batch)
        btns.addWidget(close_btn)
        btns.addWidget(self._run_btn)
        lay.addLayout(btns)

    def _pick_dir(self, target: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Folder", target.text() or str(Path.home()))
        if d:
            target.setText(d)

    def _run_batch(self) -> None:
        in_dir = Path(self._in_edit.text().strip())
        out_dir = Path(self._out_edit.text().strip())
        if not in_dir.is_dir():
            QMessageBox.warning(self, "Invalid Input", "Please select a valid input folder.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Invalid Output", "Please specify an output folder.")
            return

        ext = SUPPORTED_SAVE_FORMATS[self._fmt_combo.currentText()]
        self._run_btn.setEnabled(False)

        def _on_progress(idx: int, total: int, name: str) -> None:
            self._progress.setMaximum(max(1, total))
            self._progress.setValue(idx)
            self._status.setText(f"Processed {idx}/{total}: {name}")
            QApplication.processEvents()

        outputs = self._pipeline.batch_process(in_dir, out_dir, out_ext=ext, progress_cb=_on_progress)
        self._run_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Batch Complete",
            f"Successfully exported {len(outputs)} processed images to:\n{out_dir}",
        )
        self.accept()
