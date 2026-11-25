"""Non-destructive pipeline stack manager widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...assets.icons import get_icon
from ...core.registry import OperationRegistry
from ...core.types import PipelineStep, StepResult
from ..theme import DARK_PALETTE, ThemePalette
from .tool_sidebar import _NoFocusDelegate


class _StepRowWidget(QFrame):
    toggleClicked = pyqtSignal(int)
    moveUpClicked = pyqtSignal(int)
    moveDownClicked = pyqtSignal(int)
    duplicateClicked = pyqtSignal(int)
    deleteClicked = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        name: str,
        summary: str,
        enabled: bool,
        elapsed_ms: float | None,
        palette: ThemePalette,
        *,
        is_editing: bool = False,
    ) -> None:
        super().__init__()
        self._index = index
        self.setObjectName("SurfaceCard")
        if is_editing:
            self.setStyleSheet(
                f"QFrame#SurfaceCard {{ border: 1px solid {palette.accent}; background-color: {palette.accent_subtle}; }}"
            )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 6, 6)
        lay.setSpacing(5)

        eye_btn = QToolButton()
        eye_btn.setObjectName("RowBtn")
        eye_btn.setFixedSize(22, 22)
        eye_btn.setIcon(get_icon("eye" if enabled else "eye_off", palette.accent if enabled else palette.text_muted, 14))
        eye_btn.setToolTip("Toggle step visibility")
        eye_btn.clicked.connect(lambda: self.toggleClicked.emit(self._index))
        lay.addWidget(eye_btn)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        title = QLabel(f"{index + 1}. {name}")
        title.setMinimumWidth(0)
        title.setStyleSheet(
            f"font-weight: 600; font-size: 11px; background: transparent; color: {palette.text_primary if enabled else palette.text_muted};"
        )
        sub_txt = summary
        if len(sub_txt) > 22:
            sub_txt = sub_txt[:21] + "…"
        if elapsed_ms is not None and enabled:
            sub_txt = f"{sub_txt} ({elapsed_ms:.1f} ms)" if sub_txt else f"{elapsed_ms:.1f} ms"
        sub = QLabel(sub_txt)
        sub.setMinimumWidth(0)
        sub.setStyleSheet(f"font-size: 10px; background: transparent; color: {palette.text_muted};")
        text_col.addWidget(title)
        if sub_txt:
            text_col.addWidget(sub)
        lay.addLayout(text_col, 1)

        for icon_name, tip, sig in (
            ("up", "Move Up", self.moveUpClicked),
            ("down", "Move Down", self.moveDownClicked),
            ("copy", "Duplicate Step", self.duplicateClicked),
            ("trash", "Remove Step", self.deleteClicked),
        ):
            btn = QToolButton()
            btn.setObjectName("RowBtn")
            btn.setIcon(get_icon(icon_name, palette.text_secondary, 13))
            btn.setToolTip(tip)
            btn.setFixedSize(22, 22)
            btn.clicked.connect(lambda _c=False, s=sig: s.emit(self._index))
            lay.addWidget(btn)


class PipelinePanel(QWidget):
    """Displays and manages the ordered non-destructive operation stack."""

    stepEditRequested = pyqtSignal(int)
    stepToggleRequested = pyqtSignal(int)
    stepMoveRequested = pyqtSignal(int, int)
    stepDuplicateRequested = pyqtSignal(int)
    stepDeleteRequested = pyqtSignal(int)
    clearAllRequested = pyqtSignal()
    savePresetRequested = pyqtSignal()
    loadPresetRequested = pyqtSignal()
    batchRequested = pyqtSignal()

    def __init__(self, registry: OperationRegistry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._palette: ThemePalette = DARK_PALETTE

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        title = QLabel("PIPELINE")
        title.setObjectName("SectionTitle")
        hdr.addWidget(title)

        self._count_lbl = QLabel("(0)")
        self._count_lbl.setObjectName("MutedLabel")
        hdr.addWidget(self._count_lbl)
        hdr.addStretch(1)

        self._save_btn = QToolButton()
        self._save_btn.setObjectName("RowBtn")
        self._save_btn.setFixedSize(24, 24)
        self._save_btn.setToolTip("Save Pipeline Preset (.json)")
        self._save_btn.clicked.connect(self.savePresetRequested.emit)
        hdr.addWidget(self._save_btn)

        self._load_btn = QToolButton()
        self._load_btn.setObjectName("RowBtn")
        self._load_btn.setFixedSize(24, 24)
        self._load_btn.setToolTip("Load Pipeline Preset (.json)")
        self._load_btn.clicked.connect(self.loadPresetRequested.emit)
        hdr.addWidget(self._load_btn)

        self._batch_btn = QToolButton()
        self._batch_btn.setObjectName("RowBtn")
        self._batch_btn.setFixedSize(24, 24)
        self._batch_btn.setToolTip("Batch Process Folder with Pipeline")
        self._batch_btn.clicked.connect(self.batchRequested.emit)
        hdr.addWidget(self._batch_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("LinkButton")
        self._clear_btn.setFlat(True)
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self.clearAllRequested.emit)
        hdr.addWidget(self._clear_btn)
        lay.addLayout(hdr)

        self._stack = QStackedLayout()
        empty_box = QWidget()
        empty_lay = QVBoxLayout(empty_box)
        empty_lay.setContentsMargins(8, 16, 8, 16)
        empty_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_lbl = QLabel("No pipeline steps yet.\nSelect a tool and click Apply to build your stack.")
        empty_lbl.setObjectName("MutedLabel")
        empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_lbl.setWordWrap(True)
        empty_lay.addWidget(empty_lbl)
        self._stack.addWidget(empty_box)

        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setItemDelegate(_NoFocusDelegate(self._list))
        self._list.setMinimumHeight(110)
        self._list.setSpacing(4)
        self._list.setStyleSheet("QListWidget { background: transparent; border: none; } QListWidget::item { padding: 0px; background: transparent; } QListWidget::item:selected { background: transparent; }")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._stack.addWidget(self._list)
        lay.addLayout(self._stack, 1)

        self._refresh_toolbar_icons()

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self._refresh_toolbar_icons()

    def _refresh_toolbar_icons(self) -> None:
        c = self._palette.text_secondary
        self._save_btn.setIcon(get_icon("preset_save", c, 15))
        self._load_btn.setIcon(get_icon("preset_load", c, 15))
        self._batch_btn.setIcon(get_icon("batch", c, 15))

    def set_steps(
        self,
        steps: list[PipelineStep],
        results: list[StepResult] | None = None,
        editing_index: int | None = None,
    ) -> None:
        self._list.clear()
        self._count_lbl.setText(f"({len(steps)})")
        self._clear_btn.setEnabled(bool(steps))
        self._stack.setCurrentIndex(1 if steps else 0)
        res_by_id = {r.step_id: r for r in (results or [])}

        for idx in range(len(steps) - 1, -1, -1):
            step = steps[idx]
            spec = self._registry.get(step.op_id)
            sr = res_by_id.get(step.step_id)
            summary = self._summarize_params(step)
            row = _StepRowWidget(
                idx,
                spec.name,
                summary,
                step.enabled,
                sr.elapsed_ms if sr else None,
                self._palette,
                is_editing=(editing_index == idx),
            )
            row.toggleClicked.connect(self.stepToggleRequested.emit)
            row.moveUpClicked.connect(lambda i: self.stepMoveRequested.emit(i, 1))
            row.moveDownClicked.connect(lambda i: self.stepMoveRequested.emit(i, -1))
            row.duplicateClicked.connect(self.stepDuplicateRequested.emit)
            row.deleteClicked.connect(self.stepDeleteRequested.emit)

            item = QListWidgetItem(self._list)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            if editing_index == idx:
                item.setSelected(True)

    @staticmethod
    def _summarize_params(step: PipelineStep) -> str:
        parts: list[str] = []
        for k, v in step.params.items():
            if k == "kernel":
                parts.append(f"{len(v)}×{len(v)}")
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            elif isinstance(v, bool):
                if v:
                    parts.append(k)
            else:
                parts.append(f"{v}")
            if len(parts) >= 3:
                break
        return ", ".join(parts)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.stepEditRequested.emit(int(idx))
