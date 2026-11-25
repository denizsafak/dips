"""Searchable categorized tool browser with monochrome operation & category icons."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...assets.icons import get_icon
from ...core.registry import OperationRegistry
from ...core.types import OperationCategory, OperationSpec
from ..theme import DARK_PALETTE, ThemePalette


class _NoFocusDelegate(QStyledItemDelegate):
    """Strip Qt focus-frame state so selection never renders corner focus ticks."""

    def initStyleOption(self, option: QStyleOptionViewItem, index) -> None:
        super().initStyleOption(option, index)
        option.state &= ~QStyle.StateFlag.State_HasFocus
        option.state &= ~QStyle.StateFlag.State_KeyboardFocusChange

_CATEGORY_ICONS = {
    OperationCategory.AI: "cat_ai",
    OperationCategory.BASICS: "cat_basics",
    OperationCategory.AFFINE: "cat_affine",
    OperationCategory.GEOMETRY: "cat_geometry",
    OperationCategory.INTENSITY: "cat_intensity",
    OperationCategory.SPATIAL: "cat_spatial",
    OperationCategory.FREQUENCY: "cat_frequency",
    OperationCategory.HISTOGRAM: "cat_histogram",
    OperationCategory.MORPHOLOGY: "cat_morphology",
    OperationCategory.RESTORATION: "cat_restoration",
    OperationCategory.COLOR: "cat_color",
    OperationCategory.ARTISTIC: "cat_artistic",
    OperationCategory.FEATURES: "cat_features",
}

_OP_ICONS = {
    "ai.upscale": "op_upscale",
    "ai.bg_remove": "op_cutout",
    "ai.sticker_cutout": "op_cutout",
    "ai.bokeh": "op_bokeh",
    "ai.color_splash": "op_color",
    "ai.face_detect": "op_face",
    "ai.face_privacy": "op_shield",
    "ai.face_auto_crop": "op_crop",
    "ai.face_enhance": "op_wand",
    "ai.style_transfer": "op_wand",
    "ai.smart_enhance": "cat_ai",
    "ai.depth_map": "op_cube",
    "ai.depth_focus": "op_bokeh",
    "ai.depth_fog": "op_cloud",
    "ai.depth_relight": "op_sun",
    "ai.anime_style": "op_wand",
    "ai.object_detect": "cat_features",
    "ai.object_privacy": "op_shield",
    "ai.low_light": "op_moon",
    "basics.grayscale": "op_mono",
    "basics.flip": "op_flip",
    "basics.rotate90": "op_rotate",
    "basics.quantize": "op_layers",
    "affine.rotate": "op_rotate",
    "affine.scale": "op_upscale",
    "affine.translate": "op_move",
    "affine.shear": "op_shear",
    "affine.combined": "op_grid",
    "geom.crop": "op_crop",
    "geom.perspective": "op_shear",
    "geom.lens": "op_lens",
    "geom.pad": "op_frame",
    "geom.polar": "op_polar",
    "geom.tilt_shift": "op_lens",
    "intensity.contrast_stretch": "op_curve",
    "intensity.gamma": "op_curve",
    "intensity.negative": "op_invert",
    "intensity.log": "op_curve",
    "intensity.brightness_contrast": "op_sun",
    "intensity.slicing": "op_layers",
    "intensity.bit_plane": "op_layers",
    "spatial.mean": "op_blur",
    "spatial.gaussian": "op_blur",
    "spatial.median": "op_blur",
    "spatial.bilateral": "op_blur",
    "spatial.laplacian": "op_edges",
    "spatial.sobel": "op_edges",
    "spatial.gradient_classic": "op_edges",
    "spatial.canny": "op_edges",
    "spatial.unsharp": "op_wand",
    "spatial.custom_kernel": "op_grid",
    "spatial.point_line_detect": "op_edges",
    "freq.spectrum": "op_wave",
    "freq.lowpass": "op_wave",
    "freq.highpass": "op_wave",
    "freq.band": "op_wave",
    "freq.homomorphic": "op_wave",
    "hist.equalize": "op_bars",
    "hist.clahe": "op_bars",
    "hist.match": "op_bars",
    "morph.threshold": "op_thresh",
    "morph.otsu": "op_thresh",
    "morph.adaptive": "op_thresh",
    "morph.multi_threshold": "op_thresh",
    "morph.niblack_sauvola": "op_thresh",
    "morph.erode": "op_morph",
    "morph.dilate": "op_morph",
    "morph.open": "op_morph",
    "morph.close": "op_morph",
    "morph.gradient": "op_morph",
    "morph.tophat": "op_morph",
    "morph.blackhat": "op_morph",
    "morph.boundary": "op_morph",
    "morph.skeleton": "op_morph",
    "morph.hole_fill": "op_morph",
    "morph.convex_hull": "op_shapes",
    "morph.hit_or_miss": "op_morph",
    "morph.thinning": "op_morph",
    "restore.noise": "op_noise",
    "restore.nlm": "op_wand",
    "restore.motion_blur": "op_blur",
    "restore.wiener": "op_repair",
    "restore.inpaint": "op_repair",
    "restore.dehaze": "op_cloud",
    "color.channel": "op_color",
    "color.hsv_adjust": "op_color",
    "color.balance": "op_sun",
    "color.temperature_tint": "op_sun",
    "color.pseudocolor": "op_color",
    "color.hue_slice": "op_color",
    "color.skin_detect": "op_face",
    "color.color_edges": "op_edges",
    "art.clarity": "op_wand",
    "art.hdr_tone": "op_sun",
    "art.vignette": "op_bokeh",
    "art.toning": "op_color",
    "art.pencil": "op_repair",
    "art.stylize": "op_wand",
    "art.cartoon": "op_shapes",
    "art.chromatic": "op_lens",
    "feat.components": "op_shapes",
    "feat.watershed": "op_shapes",
    "feat.kmeans": "op_layers",
    "feat.corners": "op_shapes",
    "feat.contours": "op_shapes",
    "feat.hough": "op_lines",
    "feat.document_scan": "op_frame",
    "feat.region_segment": "op_grid",
    "feat.blobs": "op_shapes",
    "feat.texture": "op_layers",
    "feat.hog": "op_lines",
    "feat.local_descriptors": "cat_features",
}


class ToolSidebar(QWidget):
    """Left panel listing all categorized operations with monochrome icons and green active checkmarks."""

    operationSelected = pyqtSignal(str)
    operationDeselected = pyqtSignal()
    operationQuickApplied = pyqtSignal(str)

    def __init__(self, registry: OperationRegistry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelPane")
        self._registry = registry
        self._palette: ThemePalette = DARK_PALETTE
        self._cat_items: dict[OperationCategory, QTreeWidgetItem] = {}
        self._op_items: dict[str, QTreeWidgetItem] = {}
        self._active_draft_op: str | None = None
        self._committed_ops: set[str] = set()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        hdr = QFrame()
        hdr.setObjectName("HeaderBar")
        hlay = QVBoxLayout(hdr)
        hlay.setContentsMargins(10, 8, 10, 8)
        hlay.setSpacing(6)

        top_row = QHBoxLayout()
        title = QLabel("OPERATIONS")
        title.setObjectName("SectionTitle")
        top_row.addWidget(title)
        top_row.addStretch(1)
        count_lbl = QLabel(f"{len(registry.all_specs())}")
        count_lbl.setObjectName("MutedLabel")
        top_row.addWidget(count_lbl)
        self._expand_btn = QPushButton("Collapse")
        self._expand_btn.setObjectName("SegmentedBtn")
        self._expand_btn.setToolTip("Expand or collapse all operation categories")
        self._expand_btn.clicked.connect(self._toggle_expand_all)
        self._all_expanded = True
        top_row.addWidget(self._expand_btn)
        hlay.addLayout(top_row)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter operations (e.g. upscale, fft, otsu)...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._populate)
        hlay.addWidget(self._search)
        lay.addWidget(hdr)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tree.setAllColumnsShowFocus(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setItemDelegate(_NoFocusDelegate(self._tree))
        self._tree.setIndentation(14)
        self._tree.setIconSize(QSize(14, 14))
        self._tree.setAnimated(True)
        self._tree.setExpandsOnDoubleClick(False)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        lay.addWidget(self._tree, 1)

        hint = QLabel("Click to preview & tune • Double-click to apply")
        hint.setObjectName("MutedLabel")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setContentsMargins(6, 5, 6, 6)
        lay.addWidget(hint)

        self._populate("")

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        self._populate(self._search.text())

    def set_active_operations(self, draft_op_id: str | None, committed_op_ids: set[str]) -> None:
        """Update sidebar icons: green checkmark for draft/committed ops, monochrome otherwise."""
        self._active_draft_op = draft_op_id
        self._committed_ops = set(committed_op_ids)
        self._apply_item_icons()
        if draft_op_id is None:
            self._tree.clearSelection()
            self._tree.setCurrentItem(None)
        elif draft_op_id in self._op_items:
            item = self._op_items[draft_op_id]
            self._tree.setCurrentItem(item)

    def _apply_item_icons(self) -> None:
        for op_id, item in self._op_items.items():
            if op_id == self._active_draft_op or op_id in self._committed_ops:
                item.setIcon(0, get_icon("check", self._palette.success, 13, stroke_width=2.1))
            else:
                spec = self._registry.get(op_id)
                icon_key = _OP_ICONS.get(op_id, _CATEGORY_ICONS.get(spec.category, "info"))
                item.setIcon(0, get_icon(icon_key, self._palette.text_muted, 13))

    def _populate(self, query: str) -> None:
        self._tree.clear()
        self._cat_items.clear()
        self._op_items.clear()
        matched = self._registry.search(query)
        by_cat: dict[OperationCategory, list[OperationSpec]] = {}
        for spec in matched:
            by_cat.setdefault(spec.category, []).append(spec)

        for cat in self._registry.categories():
            specs = by_cat.get(cat, [])
            if not specs:
                continue
            cat_item = QTreeWidgetItem([cat.value])
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setIcon(0, get_icon(_CATEGORY_ICONS.get(cat, "info"), self._palette.accent, 14))
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(cat_item)
            self._cat_items[cat] = cat_item

            for spec in specs:
                child = QTreeWidgetItem([spec.name])
                child.setData(0, Qt.ItemDataRole.UserRole, spec.op_id)
                child.setToolTip(0, spec.description)
                cat_item.addChild(child)
                self._op_items[spec.op_id] = child
            cat_item.setExpanded(True if query.strip() else self._all_expanded)
        self._apply_item_icons()
        if self._active_draft_op and self._active_draft_op in self._op_items:
            self._tree.setCurrentItem(self._op_items[self._active_draft_op])

    def _toggle_expand_all(self) -> None:
        self._all_expanded = not self._all_expanded
        self._expand_btn.setText("Collapse" if self._all_expanded else "Expand")
        for item in self._cat_items.values():
            item.setExpanded(self._all_expanded)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if op_id:
            if op_id == self._active_draft_op:
                self._tree.clearSelection()
                self._tree.setCurrentItem(None)
                self.operationDeselected.emit()
            else:
                self.operationSelected.emit(str(op_id))
        else:
            item.setExpanded(not item.isExpanded())

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if op_id:
            self.operationQuickApplied.emit(str(op_id))
