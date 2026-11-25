"""Dynamic parameter inspector with live preview, transfer curve / SE / kernel / affine visualizers."""

from __future__ import annotations

from typing import Any
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.ops_intensity import compute_lut
from ...core.types import OperationSpec, ParamSpec, ParamType
from ..theme import DARK_PALETTE, ThemePalette, configure_combobox_popup
from .preview_widgets import (
    AffineMatrixView,
    KernelMatrixWidget,
    StructuringElementPreview,
    TransferCurveWidget,
)


class _WheelScrollForwarder(QObject):
    """Prevents sliders/spinboxes/combos from stealing mouse-wheel scroll unless focused."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and hasattr(watched, "hasFocus") and not watched.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(watched, event)


class ParamPanel(QWidget):
    """Generates dynamic parameter controls for any OperationSpec and emits live updates."""

    paramsChanged = pyqtSignal(dict)
    applyRequested = pyqtSignal()
    cancelRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette: ThemePalette = DARK_PALETTE
        self._spec: OperationSpec | None = None
        self._values: dict[str, Any] = {}
        self._editing_mode: bool = False
        self._editing_index: int | None = None
        self._building: bool = False
        self._syncing_widgets: bool = False

        self._curve_widget: TransferCurveWidget | None = None
        self._kernel_widget: KernelMatrixWidget | None = None
        self._se_widget: StructuringElementPreview | None = None
        self._affine_widget: AffineMatrixView | None = None
        self._widget_rows: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        hdr = QHBoxLayout()
        self._section_lbl = QLabel("PARAMETERS")
        self._section_lbl.setObjectName("SectionTitle")
        hdr.addWidget(self._section_lbl)
        hdr.addStretch(1)
        self._reset_btn = QPushButton("Defaults")
        self._reset_btn.setObjectName("LinkButton")
        self._reset_btn.setFlat(True)
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.setVisible(False)
        self._reset_btn.clicked.connect(self.reset_defaults)
        hdr.addWidget(self._reset_btn)
        root.addLayout(hdr)

        self._op_title = QLabel("Select an operation from the left panel")
        self._op_title.setObjectName("TitleLabel")
        self._op_title.setWordWrap(True)
        root.addWidget(self._op_title)

        self._op_desc = QLabel("Live preview appears immediately in the Result viewport before you commit a step.")
        self._op_desc.setObjectName("MutedLabel")
        self._op_desc.setWordWrap(True)
        root.addWidget(self._op_desc)

        self._scroll = QScrollArea()
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._scroll.viewport().setStyleSheet("background: transparent;")
        self._wheel_guard = _WheelScrollForwarder(self)
        self._form_host = QWidget()
        self._form_host.setObjectName("FormHost")
        self._form_host.setStyleSheet("QWidget#FormHost { background: transparent; }")
        self._form_lay = QVBoxLayout(self._form_host)
        self._form_lay.setContentsMargins(0, 4, 10, 4)
        self._form_lay.setSpacing(6)
        self._form_lay.addStretch(1)
        self._scroll.setWidget(self._form_host)
        root.addWidget(self._scroll, 1)

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("MutedLabel")
        self._meta_label.setWordWrap(True)
        self._meta_label.setVisible(False)
        root.addWidget(self._meta_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._cancel_btn = QPushButton("Discard")
        self._cancel_btn.clicked.connect(self.cancelRequested.emit)
        self._cancel_btn.setVisible(False)
        btn_row.addWidget(self._cancel_btn)

        self._apply_btn = QPushButton("Apply to Pipeline")
        self._apply_btn.setObjectName("PrimaryButton")
        self._apply_btn.clicked.connect(self.applyRequested.emit)
        self._apply_btn.setVisible(False)
        btn_row.addWidget(self._apply_btn, 1)
        root.addLayout(btn_row)

    def set_palette(self, palette: ThemePalette) -> None:
        self._palette = palette
        if self._curve_widget:
            self._curve_widget.set_palette(palette)
        if self._se_widget:
            self._se_widget.set_palette(palette)

    def set_values_externally(self, updates: dict[str, Any]) -> None:
        """Update parameter values from interactive canvas overlays without rebuilding the form."""
        if self._spec is None:
            return
        self._values.update(updates)
        self._syncing_widgets = True
        try:
            for name, val in updates.items():
                row = self._widget_rows.get(name)
                if row is None:
                    continue
                for child in row.findChildren(QDoubleSpinBox):
                    child.setValue(float(val))
                for child in row.findChildren(QSpinBox):
                    child.setValue(int(round(float(val))))
                for child in row.findChildren(QSlider):
                    p = next((ps for ps in self._spec.params if ps.name == name), None)
                    if p and p.param_type == ParamType.FLOAT:
                        scale = 10 ** max(1, min(3, p.decimals))
                        child.setValue(int(round(float(val) * scale)))
                    else:
                        child.setValue(int(round(float(val))))
        finally:
            self._syncing_widgets = False
        self.paramsChanged.emit(dict(self._values))

    def clear_operation(self) -> None:
        self._spec = None
        self._values = {}
        self._clear_form()
        self._op_title.setText("Select an operation from the left panel")
        self._op_desc.setText("Live preview appears immediately in the Result viewport before you commit a step.")
        self._reset_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._apply_btn.setVisible(False)
        self._meta_label.setVisible(False)

    def set_operation(
        self,
        spec: OperationSpec,
        initial_params: dict[str, Any] | None = None,
        *,
        editing_index: int | None = None,
    ) -> None:
        self._building = True
        self._spec = spec
        self._editing_mode = editing_index is not None
        self._editing_index = editing_index
        self._values = spec.default_params()
        if initial_params:
            self._values.update(initial_params)

        self._clear_form()
        self._op_title.setText(
            f"{spec.name}  (Editing Step #{editing_index + 1})" if editing_index is not None else spec.name
        )
        self._op_desc.setText(spec.description)
        self._reset_btn.setVisible(bool(spec.params))
        self._cancel_btn.setVisible(True)
        self._apply_btn.setVisible(True)
        self._apply_btn.setText("Update Step" if self._editing_mode else "Apply to Pipeline")
        self._meta_label.setVisible(False)

        # Insert preview widget if applicable
        if spec.preview_widget == "curve":
            self._curve_widget = TransferCurveWidget()
            self._curve_widget.set_palette(self._palette)
            self._form_lay.insertWidget(self._form_lay.count() - 1, self._curve_widget)
        elif spec.preview_widget == "se":
            self._se_widget = StructuringElementPreview()
            self._se_widget.set_palette(self._palette)
            self._form_lay.insertWidget(self._form_lay.count() - 1, self._se_widget)
        elif spec.preview_widget == "affine":
            self._affine_widget = AffineMatrixView()
            self._form_lay.insertWidget(self._form_lay.count() - 1, self._affine_widget)

        for p in spec.params:
            row_widget = self._create_param_control(p)
            self._widget_rows[p.name] = row_widget
            self._form_lay.insertWidget(self._form_lay.count() - 1, row_widget)

        self._building = False
        self._refresh_visualizers()
        self._refresh_conditional_visibility()

    def set_metadata(self, meta: dict[str, Any]) -> None:
        if self._affine_widget and "affine_matrix" in meta:
            self._affine_widget.set_matrix(meta["affine_matrix"])
        display = {k: v for k, v in meta.items() if k != "affine_matrix"}
        if display:
            parts = [f"{k.replace('_', ' ')}: {v}" for k, v in display.items()]
            self._meta_label.setText(" • ".join(parts))
            self._meta_label.setVisible(True)
        else:
            self._meta_label.setVisible(False)

    def reset_defaults(self) -> None:
        if self._spec is None:
            return
        self.set_operation(
            self._spec,
            self._spec.default_params(),
            editing_index=self._editing_index,
        )
        self.paramsChanged.emit(dict(self._values))

    def _clear_form(self) -> None:
        self._curve_widget = None
        self._kernel_widget = None
        self._se_widget = None
        self._affine_widget = None
        self._widget_rows.clear()
        while self._form_lay.count() > 1:
            item = self._form_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _create_param_control(self, p: ParamSpec) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        if p.param_type == ParamType.BOOL:
            chk = QCheckBox(p.label)
            chk.setChecked(bool(self._values.get(p.name, p.default)))
            if p.tooltip:
                chk.setToolTip(p.tooltip)
            chk.toggled.connect(lambda val, name=p.name: self._on_value_changed(name, val))
            lay.addWidget(chk)
            return box

        if p.param_type == ParamType.CHOICE:
            row = QHBoxLayout()
            lbl = QLabel(p.label)
            lbl.setMinimumWidth(0)
            combo = configure_combobox_popup(QComboBox())
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            combo.installEventFilter(self._wheel_guard)
            combo.addItems(list(p.choices))
            cur = str(self._values.get(p.name, p.default))
            idx = combo.findText(cur)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentTextChanged.connect(lambda val, name=p.name: self._on_value_changed(name, val))
            row.addWidget(lbl)
            row.addWidget(combo, 1)
            lay.addLayout(row)
            return box

        if p.param_type == ParamType.KERNEL:
            lbl = QLabel(p.label)
            lay.addWidget(lbl)
            kw = KernelMatrixWidget()
            kw.set_kernel(self._values.get(p.name, p.default))
            kw.kernelChanged.connect(lambda mat, name=p.name: self._on_value_changed(name, mat))
            self._kernel_widget = kw
            lay.addWidget(kw)
            return box

        if p.param_type == ParamType.INT:
            top = QHBoxLayout()
            lbl = QLabel(p.label)
            spin = QSpinBox()
            spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            spin.installEventFilter(self._wheel_guard)
            lo = int(p.min_val if p.min_val is not None else -1000)
            hi = int(p.max_val if p.max_val is not None else 1000)
            spin.setRange(lo, hi)
            spin.setSingleStep(int(p.step))
            if p.unit:
                spin.setSuffix(f" {p.unit}")
            val = int(self._values.get(p.name, p.default))
            spin.setValue(val)
            top.addWidget(lbl)
            top.addStretch(1)
            top.addWidget(spin)
            lay.addLayout(top)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setFixedHeight(22)
            slider.setCursor(Qt.CursorShape.PointingHandCursor)
            slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            slider.installEventFilter(self._wheel_guard)
            slider.setRange(lo, hi)
            slider.setSingleStep(int(p.step))
            slider.setValue(val)
            lay.addWidget(slider)

            def _from_slider(v: int) -> None:
                if self._syncing_widgets:
                    return
                if p.odd_only and v % 2 == 0:
                    v = v + 1 if v < hi else v - 1
                self._syncing_widgets = True
                slider.setValue(v)
                spin.setValue(v)
                self._syncing_widgets = False
                self._on_value_changed(p.name, v)

            def _from_spin(v: int) -> None:
                if self._syncing_widgets:
                    return
                if p.odd_only and v % 2 == 0:
                    v = v + 1 if v < hi else v - 1
                    spin.blockSignals(True)
                    spin.setValue(v)
                    spin.blockSignals(False)
                self._syncing_widgets = True
                slider.setValue(v)
                self._syncing_widgets = False
                self._on_value_changed(p.name, v)

            slider.valueChanged.connect(_from_slider)
            spin.valueChanged.connect(_from_spin)
            return box

        # FLOAT: slider mapped with scale factor 10^decimals
        scale = 10 ** max(1, min(3, p.decimals))
        lo = float(p.min_val if p.min_val is not None else 0.0)
        hi = float(p.max_val if p.max_val is not None else 10.0)
        val = float(self._values.get(p.name, p.default))

        top = QHBoxLayout()
        lbl = QLabel(p.label)
        spin = QDoubleSpinBox()
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spin.installEventFilter(self._wheel_guard)
        spin.setRange(lo, hi)
        spin.setDecimals(p.decimals)
        spin.setSingleStep(p.step)
        if p.unit:
            spin.setSuffix(f" {p.unit}")
        spin.setValue(val)
        top.addWidget(lbl)
        top.addStretch(1)
        top.addWidget(spin)
        lay.addLayout(top)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        slider.installEventFilter(self._wheel_guard)
        slider.setRange(int(round(lo * scale)), int(round(hi * scale)))
        slider.setValue(int(round(val * scale)))
        lay.addWidget(slider)

        def _from_fslider(sv: int) -> None:
            if self._syncing_widgets:
                return
            fv = sv / scale
            self._syncing_widgets = True
            spin.setValue(fv)
            self._syncing_widgets = False
            self._on_value_changed(p.name, fv)

        def _from_fspin(fv: float) -> None:
            if self._syncing_widgets:
                return
            self._syncing_widgets = True
            slider.setValue(int(round(fv * scale)))
            self._syncing_widgets = False
            self._on_value_changed(p.name, fv)

        slider.valueChanged.connect(_from_fslider)
        spin.valueChanged.connect(_from_fspin)
        return box

    def _on_value_changed(self, name: str, value: Any) -> None:
        self._values[name] = value
        self._refresh_visualizers()
        self._refresh_conditional_visibility()
        if not self._building:
            self.paramsChanged.emit(dict(self._values))

    def _refresh_visualizers(self) -> None:
        if self._spec is None:
            return
        if self._curve_widget is not None:
            lut = compute_lut(self._spec.op_id, self._values)
            self._curve_widget.setVisible(lut is not None)
            self._curve_widget.set_lut(lut)
        if self._se_widget is not None:
            shape = str(self._values.get("se_shape", "Rectangle"))
            w = int(self._values.get("se_w", 3))
            h = w if bool(self._values.get("se_square", True)) else int(self._values.get("se_h", 3))
            self._se_widget.set_se(shape, w, h)

    def _refresh_conditional_visibility(self) -> None:
        if self._spec is None:
            return
        op_id = self._spec.op_id
        if op_id == "intensity.contrast_stretch":
            mode = str(self._values.get("mode", "Auto Min-Max"))
            is_pct = mode == "Percentile Clip"
            is_pw = mode == "Piecewise Linear (r1,s1,r2,s2)"
            for k in ("low_pct", "high_pct"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_pct)
            for k in ("r1", "s1", "r2", "s2"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_pw)
            if "per_channel" in self._widget_rows:
                self._widget_rows["per_channel"].setVisible(not is_pw)
        elif op_id == "affine.scale":
            if "scale_y" in self._widget_rows:
                self._widget_rows["scale_y"].setVisible(not bool(self._values.get("uniform", True)))
        elif op_id == "affine.shear":
            m = str(self._values.get("mode", "Both"))
            if "shear_x" in self._widget_rows:
                self._widget_rows["shear_x"].setVisible(m in ("Shear X", "Both"))
            if "shear_y" in self._widget_rows:
                self._widget_rows["shear_y"].setVisible(m in ("Shear Y", "Both"))
        elif op_id == "spatial.mean":
            if "ksize_y" in self._widget_rows:
                self._widget_rows["ksize_y"].setVisible(not bool(self._values.get("square", True)))
        elif op_id == "spatial.laplacian":
            is_sharp = str(self._values.get("mode", "")) == "Sharpen (Original - c·∇²f)"
            if "sharpen_weight" in self._widget_rows:
                self._widget_rows["sharpen_weight"].setVisible(is_sharp)
            if "normalize" in self._widget_rows:
                self._widget_rows["normalize"].setVisible(str(self._values.get("mode", "")) == "Edge Magnitude")
        elif op_id == "spatial.canny":
            auto = bool(self._values.get("auto_thresholds", False))
            for k in ("threshold1", "threshold2"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(not auto)
        elif op_id == "morph.erode" or op_id.startswith("morph."):
            if "se_h" in self._widget_rows:
                self._widget_rows["se_h"].setVisible(not bool(self._values.get("se_square", True)))
        elif op_id == "color.balance":
            auto_wb = bool(self._values.get("auto_white_balance", False))
            for k in ("r_gain", "g_gain", "b_gain"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(not auto_wb)
        elif op_id == "feat.corners":
            det = str(self._values.get("detector", "Shi-Tomasi"))
            for k in ("max_corners", "quality_level", "min_distance"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(det == "Shi-Tomasi")
            if "threshold_pct" in self._widget_rows:
                self._widget_rows["threshold_pct"].setVisible(det == "Harris")
            if "fast_threshold" in self._widget_rows:
                self._widget_rows["fast_threshold"].setVisible(det == "FAST")
        elif op_id == "feat.hough":
            is_lines = str(self._values.get("mode", "")).startswith("Lines")
            for k in ("min_line_len", "max_line_gap"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_lines)
            if "min_dist" in self._widget_rows:
                self._widget_rows["min_dist"].setVisible(not is_lines)
        elif op_id in ("freq.lowpass", "freq.highpass"):
            is_bw = str(self._values.get("filter_type", "")) == "Butterworth"
            if "order" in self._widget_rows:
                self._widget_rows["order"].setVisible(is_bw)
        elif op_id == "restore.noise":
            nt = str(self._values.get("noise_type", "Salt & Pepper"))
            if "amount" in self._widget_rows:
                self._widget_rows["amount"].setVisible(nt == "Salt & Pepper")
            if "sigma" in self._widget_rows:
                self._widget_rows["sigma"].setVisible(nt != "Salt & Pepper")
        elif op_id == "restore.wiener":
            is_motion = str(self._values.get("psf_type", "Motion")) == "Motion"
            if "angle" in self._widget_rows:
                self._widget_rows["angle"].setVisible(is_motion)
        elif op_id == "geom.crop":
            is_free = str(self._values.get("aspect", "Free")) == "Free"
            is_center = bool(self._values.get("center_crop", False))
            if "h_pct" in self._widget_rows:
                self._widget_rows["h_pct"].setVisible(is_free)
            for k in ("x_pct", "y_pct"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(not is_center)
        elif op_id == "morph.multi_threshold":
            is_3lvl = str(self._values.get("mode", "")) == "3-Level (T1, T2)"
            for k in ("t1", "t2"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_3lvl)
        elif op_id == "morph.niblack_sauvola":
            is_sauv = str(self._values.get("method", "Sauvola")) == "Sauvola"
            if "sauvola_r" in self._widget_rows:
                self._widget_rows["sauvola_r"].setVisible(is_sauv)
        elif op_id == "feat.region_segment":
            is_grow = str(self._values.get("mode", "")) == "Seeded Region Growing"
            for k in ("std_thresh", "min_block", "draw_grid"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(not is_grow)
            for k in ("seed_x_pct", "seed_y_pct", "tolerance"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_grow)
        elif op_id == "feat.texture":
            m = str(self._values.get("method", ""))
            is_gabor = m.startswith("Gabor")
            is_glcm = m.startswith("GLCM")
            for k in ("angle_deg", "wavelength", "sigma"):
                if k in self._widget_rows:
                    self._widget_rows[k].setVisible(is_gabor)
            if "glcm_dist" in self._widget_rows:
                self._widget_rows["glcm_dist"].setVisible(is_glcm)
