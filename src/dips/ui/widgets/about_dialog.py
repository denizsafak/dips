"""About & Keyboard Shortcuts dialog for DIPS."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ... import __app_name__, __author__, __email__, __version__
from ...assets import asset_path


class AboutDialog(QDialog):
    """Displays DIP Studio identity, licensing, author credits, and keyboard shortcuts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {__app_name__}")
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(22, 20, 22, 18)

        hdr = QHBoxLayout()
        hdr.setSpacing(14)
        icon_lbl = QLabel()
        pm = QPixmap(str(asset_path("dips.png")))
        if not pm.isNull():
            icon_lbl.setPixmap(pm.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        hdr.addWidget(icon_lbl)

        meta = QVBoxLayout()
        title = QLabel(f"{__app_name__} v{__version__}")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        sub = QLabel("Cross-Platform Digital Image Processing Studio")
        sub.setObjectName("MutedLabel")
        author = QLabel(f"Author: {__author__} ({__email__})\nLicense: MIT License")
        author.setStyleSheet("font-size: 11px;")
        meta.addWidget(title)
        meta.addWidget(sub)
        meta.addWidget(author)
        hdr.addLayout(meta, 1)
        lay.addLayout(hdr)

        shortcuts = QLabel(
            "<b>Keyboard Shortcuts</b><br>"
            "• <b>Ctrl+O</b>: Open Image &nbsp;&nbsp;• <b>Ctrl+S</b>: Quick Save Result<br>"
            "• <b>Ctrl+Shift+E</b>: Export Dialog &nbsp;&nbsp;• <b>Ctrl+C / Ctrl+V</b>: Copy / Paste Image<br>"
            "• <b>Ctrl+Z / Ctrl+Y</b>: Undo / Redo &nbsp;&nbsp;• <b>Ctrl+R</b>: Reset Pipeline<br>"
            "• <b>Ctrl+0 / Ctrl+1</b>: Fit / 100% Zoom &nbsp;&nbsp;• <b>1–4</b>: Switch View Mode"
        )
        shortcuts.setStyleSheet("font-size: 11px; line-height: 1.45;")
        lay.addWidget(shortcuts)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        lay.addWidget(btns)
