"""Main studio window orchestrating DIP Studio tool browser, viewport, histogram, parameters, and pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QSettings, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __app_name__, __version__
from ..assets import asset_path
from ..assets.icons import get_icon
from ..core import (
    BUILTIN_PRESETS,
    ImagePipeline,
    discover_sample_images,
    get_registry,
    load_image,
    save_image,
)
from ..core.io_utils import SUPPORTED_OPEN_EXTS
from ..core.pipeline import is_heavy_op
from .theme import DARK_PALETTE, ThemePalette, apply_theme, detect_system_theme
from .pipeline_worker import AsyncPipelineRunner
from .update_checker import (
    GITHUB_RELEASES_URL,
    SelfUpdateDialog,
    UpdateCheckerThread,
    select_platform_asset,
)
from .widgets.about_dialog import AboutDialog
from .widgets.batch_dialog import BatchDialog
from .widgets.export_dialog import ExportDialog
from .widgets.histogram_widget import HistogramPanel
from .widgets.image_canvas import DualViewport, ndarray_to_qimage
from .widgets.param_panel import ParamPanel
from .widgets.pipeline_panel import PipelinePanel
from .widgets.tool_sidebar import ToolSidebar


class MainWindow(QMainWindow):
    """Production-grade 3-pane Digital Image Processing studio window."""

    def __init__(self, initial_image: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} - Digital Image Processing Studio")
        self.resize(1440, 880)
        self.setMinimumSize(1080, 680)

        icon_file = asset_path("dips.png")
        if icon_file.exists():
            self.setWindowIcon(QIcon(str(icon_file)))

        self._registry = get_registry()
        self._pipeline = ImagePipeline(self._registry)
        self._samples = discover_sample_images()
        self._settings = QSettings("DenizSafak", "DIPS")
        self._palette: ThemePalette = DARK_PALETTE
        saved_theme = self._settings.value("ui/theme", None)
        self._theme_mode: str = str(saved_theme) if saved_theme in ("dark", "light") else detect_system_theme()
        self._recent_files: list[str] = [
            p for p in (self._settings.value("files/recent", []) or [])
            if isinstance(p, str) and Path(p).is_file()
        ][:8]

        self._orig_image: np.ndarray | None = None
        self._result_image: np.ndarray | None = None
        self._current_path: Path | None = None
        self._image_token: int = 0
        self._active_job_id: int = 0
        self._runner = AsyncPipelineRunner(self)
        self._runner.completed.connect(self._on_pipeline_completed)
        self._runner.failed.connect(self._on_pipeline_failed)

        self._loading_delay_timer = QTimer(self)
        self._loading_delay_timer.setSingleShot(True)
        self._loading_delay_timer.setInterval(120)
        self._loading_delay_timer.timeout.connect(self._show_loading_indicator)
        self._loading_op_label = "Processing..."

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(16)
        self._preview_timer.timeout.connect(self._flush_preview_timer)
        self._preview_is_heavy: bool = False
        self._last_eval_interactive: bool = False
        self._last_step_results: list = []

        self._deferred_metrics_timer = QTimer(self)
        self._deferred_metrics_timer.setSingleShot(True)
        self._deferred_metrics_timer.setInterval(110)
        self._deferred_metrics_timer.timeout.connect(self._update_deferred_metrics)

        self._build_ui()
        self._build_actions_and_menus()
        self._connect_signals()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.installEventFilter(self)
        self.set_theme(self._theme_mode, persist=False)
        self._update_action_states()
        self._refresh_recent_menu()
        self._update_thread: UpdateCheckerThread | None = None
        self._manual_update_check = False

        if initial_image:
            self.open_image_path(initial_image)

        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QTimer.singleShot(1800, lambda: self.check_for_updates(silent=True))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # 3-pane horizontal splitter
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(1)
        self._main_splitter.setChildrenCollapsible(False)

        # Left: Categorized tool browser
        self.tool_sidebar = ToolSidebar(self._registry)
        self.tool_sidebar.setMinimumWidth(220)
        self.tool_sidebar.setMaximumWidth(340)
        self._main_splitter.addWidget(self.tool_sidebar)

        # Center: Dual Viewport
        self.viewport = DualViewport(self._samples)
        self._main_splitter.addWidget(self.viewport)

        # Right: Inspector stack (Histogram + Parameters + Pipeline)
        right_host = QWidget()
        self._right_host = right_host
        right_host.setObjectName("PanelPane")
        right_host.setMinimumWidth(320)
        right_host.setMaximumWidth(460)
        right_lay = QVBoxLayout(right_host)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setHandleWidth(1)
        self._right_splitter.setChildrenCollapsible(False)

        self.histogram_panel = HistogramPanel()
        self.histogram_panel.setMinimumHeight(160)
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumHeight(180)
        self.pipeline_panel = PipelinePanel(self._registry)
        self.pipeline_panel.setMinimumHeight(140)

        self._right_splitter.addWidget(self.histogram_panel)
        self._right_splitter.addWidget(self.param_panel)
        self._right_splitter.addWidget(self.pipeline_panel)
        self._right_splitter.setSizes([235, 395, 200])

        right_lay.addWidget(self._right_splitter)
        self._main_splitter.addWidget(right_host)
        self._main_splitter.setSizes([245, 835, 360])
        self._main_splitter.setStretchFactor(1, 1)

        # Wrapper with subtle dismissible update banner at top
        central_wrap = QWidget()
        c_lay = QVBoxLayout(central_wrap)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)

        self._update_banner = QFrame()
        self._update_banner.setObjectName("HeaderBar")
        self._update_banner.setVisible(False)
        b_lay = QHBoxLayout(self._update_banner)
        b_lay.setContentsMargins(12, 5, 10, 5)
        b_lay.setSpacing(8)
        self._update_lbl = QLabel("")
        self._update_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
        b_lay.addWidget(self._update_lbl, 1)
        self._update_url = GITHUB_RELEASES_URL
        dl_btn = QPushButton("Update Now")
        dl_btn.setObjectName("PrimaryButton")
        dl_btn.clicked.connect(self._install_pending_update)
        b_lay.addWidget(dl_btn)
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(lambda: self._update_banner.setVisible(False))
        b_lay.addWidget(dismiss_btn)
        c_lay.addWidget(self._update_banner)
        c_lay.addWidget(self._main_splitter, 1)

        self.setCentralWidget(central_wrap)

        # Status Bar with Live Pixel Inspector
        sb = QStatusBar()
        sb.setContentsMargins(8, 2, 8, 2)
        self._status_file = QLabel("Ready - Drop an image or choose a Sample Image to start")
        sb.addWidget(self._status_file, 1)

        pixel_box = QWidget()
        p_lay = QHBoxLayout(pixel_box)
        p_lay.setContentsMargins(0, 0, 10, 0)
        p_lay.setSpacing(6)
        self._swatch = QFrame()
        self._swatch.setFixedSize(14, 14)
        self._swatch.setStyleSheet(f"border: 1px solid {self._palette.border_strong}; border-radius: 3px; background: transparent;")
        self._pixel_lbl = QLabel("Cursor: -")
        p_lay.addWidget(self._swatch)
        p_lay.addWidget(self._pixel_lbl)
        sb.addPermanentWidget(pixel_box)

        self._perf_lbl = QLabel("Pipeline: 0.0 ms")
        self._perf_lbl.setContentsMargins(8, 0, 6, 0)
        sb.addPermanentWidget(self._perf_lbl)
        self.setStatusBar(sb)

    def _build_actions_and_menus(self) -> None:
        self._act_open = QAction("Open Image...", self)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self.prompt_open_image)

        self._act_close_img = QAction("Close Image", self)
        self._act_close_img.setShortcut(QKeySequence("Ctrl+W"))
        self._act_close_img.triggered.connect(self.close_current_image)

        self._act_save = QAction("Save Result...", self)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self.quick_save_result)

        self._act_export = QAction("Export Options...", self)
        self._act_export.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._act_export.triggered.connect(self.open_export_dialog)

        self._act_copy = QAction("Copy Result", self)
        self._act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        self._act_copy.triggered.connect(self.copy_result_to_clipboard)

        self._act_paste = QAction("Paste Image", self)
        self._act_paste.setShortcut(QKeySequence.StandardKey.Paste)
        self._act_paste.triggered.connect(self.paste_image_from_clipboard)

        self._act_undo = QAction("Undo", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self.undo)

        self._act_redo = QAction("Redo", self)
        self._act_redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self._act_redo.triggered.connect(self.redo)

        self._act_reset = QAction("Reset to Original", self)
        self._act_reset.setShortcut(QKeySequence("Ctrl+R"))
        self._act_reset.triggered.connect(lambda _c=False: self.reset_pipeline(confirm=True))

        self._act_fit = QAction("Fit View", self)
        self._act_fit.setShortcut(QKeySequence("Ctrl+0"))
        self._act_fit.triggered.connect(self.viewport.fit_view)

        self._act_100 = QAction("100% Zoom", self)
        self._act_100.setShortcut(QKeySequence("Ctrl+1"))
        self._act_100.triggered.connect(self.viewport.zoom_100)

        self._act_theme = QAction("Toggle Light / Dark", self)
        self._act_theme.setShortcut(QKeySequence("Ctrl+T"))
        self._act_theme.triggered.connect(self.toggle_theme)

        self._act_save_preset = QAction("Save Pipeline Preset...", self)
        self._act_save_preset.triggered.connect(self.prompt_save_preset)

        self._act_load_preset = QAction("Load Pipeline Preset...", self)
        self._act_load_preset.triggered.connect(self.prompt_load_preset)

        self._act_batch = QAction("Batch Process Folder...", self)
        self._act_batch.triggered.connect(self.open_batch_dialog)

        self._act_about = QAction(f"About {__app_name__}", self)
        self._act_about.triggered.connect(self.show_about)

        self._act_updates = QAction("Check for Updates...", self)
        self._act_updates.triggered.connect(lambda: self.check_for_updates(silent=False))

        # Menu Bar
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        m_file.addAction(self._act_open)
        m_file.addAction(self._act_close_img)
        self._recent_menu = m_file.addMenu("Recent Files")
        m_file.addSeparator()
        m_file.addAction(self._act_save)
        m_file.addAction(self._act_export)
        m_file.addSeparator()
        m_file.addAction(self._act_batch)
        m_file.addSeparator()
        act_exit = m_file.addAction("Exit")
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)

        m_edit = mb.addMenu("&Edit")
        m_edit.addAction(self._act_undo)
        m_edit.addAction(self._act_redo)
        m_edit.addAction(self._act_reset)
        m_edit.addSeparator()
        m_edit.addAction(self._act_copy)
        m_edit.addAction(self._act_paste)

        m_view = mb.addMenu("&View")
        for idx, (label, key) in enumerate(
            [
                ("Side-by-Side", "1"),
                ("Split Curtain", "2"),
                ("Difference Map", "3"),
                ("Result Only", "4"),
            ]
        ):
            act = m_view.addAction(label)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _c=False, m=idx: self.viewport.set_view_mode(m))
        m_view.addSeparator()
        m_view.addAction(self._act_fit)
        m_view.addAction(self._act_100)
        m_view.addSeparator()
        self._act_show_left = m_view.addAction("Show Operations Panel")
        self._act_show_left.setCheckable(True)
        self._act_show_left.setChecked(True)
        self._act_show_left.toggled.connect(lambda on: self._toggle_side_panel("left", on))
        self._act_show_right = m_view.addAction("Show Inspector Panel")
        self._act_show_right.setCheckable(True)
        self._act_show_right.setChecked(True)
        self._act_show_right.toggled.connect(lambda on: self._toggle_side_panel("right", on))
        act_reset_panels = m_view.addAction("Reset Panel Layout")
        act_reset_panels.triggered.connect(self._reset_panel_layout)
        m_view.addSeparator()
        m_view.addAction(self._act_theme)

        if self._samples:
            m_samples = mb.addMenu("&Samples")
            for sp in self._samples:
                act = m_samples.addAction(sp.name)
                act.triggered.connect(lambda _c=False, p=sp: self.open_image_path(p))

        m_pipe = mb.addMenu("&Pipeline")
        m_presets = m_pipe.addMenu("Built-in Presets")
        for preset_name in BUILTIN_PRESETS:
            act = m_presets.addAction(preset_name)
            act.triggered.connect(lambda _c=False, pn=preset_name: self.apply_builtin_preset(pn))
        m_pipe.addSeparator()
        m_pipe.addAction(self._act_save_preset)
        m_pipe.addAction(self._act_load_preset)
        m_pipe.addAction(self._act_reset)

        m_help = mb.addMenu("&Help")
        m_help.addAction(self._act_updates)
        m_help.addAction(self._act_about)

        # Toolbar
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for act in (self._act_open, self._act_save, self._act_export):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self._act_undo, self._act_redo, self._act_reset):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self._act_fit, self._act_100):
            tb.addAction(act)

        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Policy.Expanding, spacer.sizePolicy().Policy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self._act_theme)
        tb.addAction(self._act_about)
        self.addToolBar(tb)

    def _connect_signals(self) -> None:
        self.viewport.openRequested.connect(self.prompt_open_image)
        self.viewport.closeRequested.connect(lambda: self.close_current_image(confirm=True))
        self.viewport.sampleRequested.connect(self.open_image_path)
        self.viewport.fileDropped.connect(self._on_file_dropped)
        self.viewport.pixelInspected.connect(self._on_pixel_inspected)
        self.viewport.cropRectChanged.connect(self._on_viewport_crop_changed)
        self.viewport.cancelProcessingRequested.connect(self._on_cancel_processing)

        self.tool_sidebar.operationSelected.connect(self._on_operation_selected)
        self.tool_sidebar.operationDeselected.connect(self._on_cancel_draft)
        self.tool_sidebar.operationQuickApplied.connect(self._on_operation_quick_applied)

        self.param_panel.paramsChanged.connect(self._on_draft_params_changed)
        self.param_panel.applyRequested.connect(self._on_commit_draft)
        self.param_panel.cancelRequested.connect(self._on_cancel_draft)

        self.pipeline_panel.stepEditRequested.connect(self._on_edit_step)
        self.pipeline_panel.stepToggleRequested.connect(self._on_toggle_step)
        self.pipeline_panel.stepMoveRequested.connect(self._on_move_step)
        self.pipeline_panel.stepDuplicateRequested.connect(self._on_duplicate_step)
        self.pipeline_panel.stepDeleteRequested.connect(self._on_delete_step)
        self.pipeline_panel.clearAllRequested.connect(lambda: self.reset_pipeline(confirm=True))
        self.pipeline_panel.savePresetRequested.connect(self.prompt_save_preset)
        self.pipeline_panel.loadPresetRequested.connect(self.prompt_load_preset)
        self.pipeline_panel.batchRequested.connect(self.open_batch_dialog)

    # ------------------------------------------------------------------
    # Theme & Layout
    # ------------------------------------------------------------------
    def _toggle_side_panel(self, side: str, visible: bool) -> None:
        if side == "left":
            self.tool_sidebar.setVisible(visible)
        else:
            self._right_host.setVisible(visible)
            if visible:
                for p in (self.histogram_panel, self.param_panel, self.pipeline_panel):
                    p.setVisible(True)
                self._right_splitter.setSizes([235, 395, 200])

    def _reset_panel_layout(self) -> None:
        self._act_show_left.setChecked(True)
        self._act_show_right.setChecked(True)
        self.tool_sidebar.setVisible(True)
        self._right_host.setVisible(True)
        for p in (self.histogram_panel, self.param_panel, self.pipeline_panel):
            p.setVisible(True)
        self._main_splitter.setSizes([245, 835, 360])
        self._right_splitter.setSizes([235, 395, 200])

    def set_theme(self, mode: str, *, persist: bool = False) -> None:
        self._theme_mode = mode
        if persist:
            self._settings.setValue("ui/theme", mode)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            self._palette = apply_theme(app, mode)
        self.viewport.set_palette(self._palette)
        self.histogram_panel.set_palette(self._palette)
        self.param_panel.set_palette(self._palette)
        self.tool_sidebar.set_palette(self._palette)
        self.pipeline_panel.set_palette(self._palette)

        c = self._palette.text_secondary
        self._act_open.setIcon(get_icon("open", c))
        self._act_save.setIcon(get_icon("save", c))
        self._act_export.setIcon(get_icon("export", c))
        self._act_undo.setIcon(get_icon("undo", c))
        self._act_redo.setIcon(get_icon("redo", c))
        self._act_reset.setIcon(get_icon("reset", c))
        self._act_fit.setIcon(get_icon("fit", c))
        self._act_100.setIcon(get_icon("zoom_in", c))
        self._act_theme.setIcon(get_icon("theme", c))
        self._act_about.setIcon(get_icon("info", c))
        if hasattr(self, "_swatch") and self._pixel_lbl.text() == "Cursor: -":
            self._swatch.setStyleSheet(
                f"border: 1px solid {self._palette.border_strong}; border-radius: 3px; background: transparent;"
            )
        self._refresh_pipeline_list(None)

    def toggle_theme(self) -> None:
        self.set_theme("light" if self._theme_mode == "dark" else "dark", persist=True)

    # ------------------------------------------------------------------
    # Image Loading & Saving
    # ------------------------------------------------------------------
    def _on_file_dropped(self, path_str: str) -> None:
        p = Path(path_str)
        if p.suffix.lower() == ".json":
            self._load_preset_file(p)
        else:
            self.open_image_path(p)

    def prompt_open_image(self) -> None:
        exts = " ".join(f"*{e}" for e in SUPPORTED_OPEN_EXTS)
        start_dir = str(self._current_path.parent if self._current_path else (self._samples[0].parent if self._samples else Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_dir,
            f"Images ({exts});;All Files (*)",
        )
        if path:
            self.open_image_path(path)

    def open_image_path(self, path: str | Path) -> None:
        try:
            img = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening Image", str(exc))
            return
        self._current_path = Path(path)
        self._record_recent_file(self._current_path)
        self._set_original_image(img, reset_view=True)
        self._status_file.setText(f"Loaded: {self._current_path.name}")

    def close_current_image(self, confirm: bool = False) -> None:
        """Close the active image (with optional confirmation), reset pipeline state, and return to start screen."""
        if self._orig_image is None:
            return
        if confirm:
            ans = QMessageBox.question(
                self,
                "Close Image",
                "Close the current image and discard unsaved pipeline changes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self._preview_timer.stop()
        self._deferred_metrics_timer.stop()
        self._loading_delay_timer.stop()
        self._runner.cancel_current()
        self._runner.clear_cache()
        self.viewport.set_loading(False)
        self.viewport.set_crop_overlay(False)
        self._orig_image = None
        self._result_image = None
        self._current_path = None
        self._image_token += 1
        self._pipeline = ImagePipeline(self._registry)
        self.param_panel.clear_operation()
        self.viewport.set_images(None, None)
        self.histogram_panel.update_images(None, None)
        self._refresh_pipeline_list(None)
        self._perf_lbl.setText("Pipeline: 0.0 ms")
        self._pixel_lbl.setText("Cursor: -")
        self._swatch.setStyleSheet(f"border: 1px solid {self._palette.border_strong}; border-radius: 3px; background: transparent;")
        self._status_file.setText("Ready - Drop an image or choose a Sample Image to start")
        self._update_action_states()

    def apply_builtin_preset(self, preset_name: str) -> None:
        steps = BUILTIN_PRESETS.get(preset_name)
        if not steps:
            return
        if self._orig_image is None:
            if self._samples:
                self.open_image_path(self._samples[0])
            else:
                return
        self._pipeline.load_preset_steps(steps)
        self.param_panel.clear_operation()
        self._recompute_pipeline()
        self._update_action_states()
        self._status_file.setText(f"Applied preset: {preset_name}")

    def check_for_updates(self, *, silent: bool = False) -> None:
        if self._update_thread is not None and self._update_thread.isRunning():
            return
        self._manual_update_check = not silent
        self._pending_release: dict | None = None
        self._update_thread = UpdateCheckerThread(__version__, self)
        self._update_thread.updateAvailable.connect(self._on_update_available)
        self._update_thread.upToDate.connect(self._on_up_to_date)
        self._update_thread.checkFailed.connect(self._on_update_check_failed)
        self._update_thread.start()

    def _on_update_available(self, release: dict) -> None:
        self._pending_release = release
        new_ver = release["version"]
        self._update_url = release["html_url"]
        self._update_lbl.setText(f"DIP Studio v{new_ver} is available (running v{__version__}).")
        self._update_banner.setVisible(True)

        reply = QMessageBox.question(
            self,
            "Update Available",
            f"A new version of DIP Studio (v{new_ver}) is available.\n"
            f"You are currently running v{__version__}.\n\n"
            "Would you like to download and install the update automatically now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._install_pending_update()

    def _install_pending_update(self) -> None:
        rel = getattr(self, "_pending_release", None)
        if not rel:
            QDesktopServices.openUrl(QUrl(self._update_url))
            return
        asset = select_platform_asset(rel.get("assets", []))
        if asset is None:
            QDesktopServices.openUrl(QUrl(rel.get("html_url", GITHUB_RELEASES_URL)))
            return
        dlg = SelfUpdateDialog(
            rel["version"],
            asset["name"],
            asset["browser_download_url"],
            self,
        )
        if dlg.exec() and dlg.should_quit:
            self._runner.shutdown()
            QApplication.quit()

    def _on_up_to_date(self, cur_ver: str) -> None:
        if self._manual_update_check:
            QMessageBox.information(self, "Up to Date", f"You are running the latest version of DIP Studio (v{cur_ver}).")

    def _on_update_check_failed(self, err: str) -> None:
        if self._manual_update_check:
            QMessageBox.warning(self, "Update Check Failed", f"Could not check GitHub for updates:\n{err}")

    def _record_recent_file(self, path: Path) -> None:
        s = str(path.resolve())
        self._recent_files = [s] + [p for p in self._recent_files if p != s]
        self._recent_files = self._recent_files[:8]
        self._settings.setValue("files/recent", self._recent_files)
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        if not hasattr(self, "_recent_menu"):
            return
        self._recent_menu.clear()
        if not self._recent_files:
            empty = self._recent_menu.addAction("(Empty)")
            empty.setEnabled(False)
            return
        for p_str in self._recent_files:
            p = Path(p_str)
            act = self._recent_menu.addAction(p.name)
            act.setToolTip(p_str)
            act.triggered.connect(lambda _c=False, target=p: self.open_image_path(target))
        self._recent_menu.addSeparator()
        clr = self._recent_menu.addAction("Clear Recent Files")
        clr.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self) -> None:
        self._recent_files.clear()
        self._settings.setValue("files/recent", [])
        self._refresh_recent_menu()

    def _set_original_image(self, img: np.ndarray, *, reset_view: bool = True) -> None:
        self._image_token += 1
        self._pipeline.clear_cache()
        self._runner.clear_cache()
        self._orig_image = img
        self._recompute_pipeline(reset_view=reset_view)
        self._update_action_states()

    def quick_save_result(self) -> None:
        if self._result_image is None:
            return
        default_name = f"{self._current_path.stem}_processed.png" if self._current_path else "processed.png"
        start = str((self._current_path.parent if self._current_path else Path.home()) / default_name)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image",
            start,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;WebP (*.webp)",
        )
        if path:
            try:
                saved = save_image(self._result_image, path)
                self._status_file.setText(f"Saved result: {saved.name}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", str(exc))

    def open_export_dialog(self) -> None:
        if self._orig_image is None or self._result_image is None:
            return
        suggested = f"{self._current_path.stem}_dips.png" if self._current_path else "dips_result.png"
        dlg = ExportDialog(self._orig_image, self._result_image, suggested, self)
        if dlg.exec() and dlg.exported_path is not None:
            self._status_file.setText(f"Exported: {dlg.exported_path.name}")

    def copy_result_to_clipboard(self) -> None:
        if self._result_image is None:
            return
        QApplication.clipboard().setImage(ndarray_to_qimage(self._result_image))
        self._status_file.setText("Copied Result image to clipboard")

    def paste_image_from_clipboard(self) -> None:
        qimg = QApplication.clipboard().image()
        if qimg.isNull():
            return
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()
        if int(arr[:, :, 3].min()) < 255:
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        self._current_path = None
        self._set_original_image(img, reset_view=True)
        self._status_file.setText("Loaded image from clipboard")

    # ------------------------------------------------------------------
    # Pipeline & Operation Slots
    # ------------------------------------------------------------------
    def _on_operation_selected(self, op_id: str) -> None:
        spec = self._registry.get(op_id)
        step = self._pipeline.set_draft(op_id)
        self.param_panel.set_operation(spec, step.params, editing_index=None)
        self._sync_crop_overlay()
        self._sync_sidebar_indicators()
        self._update_action_states()
        self._recompute_pipeline()

    def _on_operation_quick_applied(self, op_id: str) -> None:
        self._pipeline.clear_draft()
        self._pipeline.add_step(op_id)
        self.param_panel.clear_operation()
        self._sync_crop_overlay()
        self._recompute_pipeline()
        self._update_action_states()

    def _on_draft_params_changed(self, params: dict) -> None:
        self._pipeline.update_draft_params(params)
        self._sync_crop_overlay()
        draft = self._pipeline.draft_step
        heavy = draft is not None and is_heavy_op(draft.op_id)
        if heavy:
            self._preview_is_heavy = True
            self._preview_timer.setInterval(85)
            self._preview_timer.start()
        elif not self._preview_timer.isActive() or self._preview_is_heavy:
            # Frame-paced 16ms (~60fps) throttle that never starves continuous slider drags
            self._preview_is_heavy = False
            self._preview_timer.setInterval(16)
            self._preview_timer.start()

    def _flush_preview_timer(self) -> None:
        if self._preview_is_heavy:
            self._recompute_pipeline(preempt=False, interactive=True)
        else:
            self._recompute_pipeline(interactive=True)

    def _on_commit_draft(self) -> None:
        self._preview_timer.stop()
        self._pipeline.commit_draft()
        self.param_panel.clear_operation()
        self._sync_crop_overlay()
        self._recompute_pipeline()
        self._update_action_states()

    def _on_cancel_draft(self) -> None:
        self._preview_timer.stop()
        self._deferred_metrics_timer.stop()
        self._loading_delay_timer.stop()
        self._runner.cancel_current()
        self.viewport.set_loading(False)
        self._pipeline.clear_draft()
        self.param_panel.clear_operation()
        self._sync_crop_overlay()
        self._sync_sidebar_indicators()
        self._recompute_pipeline()

    def _on_cancel_processing(self) -> None:
        """Immediately abort active subprocess execution and discard pending draft."""
        self._preview_timer.stop()
        self._deferred_metrics_timer.stop()
        self._loading_delay_timer.stop()
        self._runner.cancel_current()
        self.viewport.set_loading(False)
        if self._pipeline.draft_step is not None:
            self._pipeline.clear_draft()
            self.param_panel.clear_operation()
            self._sync_crop_overlay()
            self._sync_sidebar_indicators()
            self._recompute_pipeline()
        self._status_file.setText("Processing cancelled")

    def _sync_crop_overlay(self) -> None:
        draft = self._pipeline.draft_step
        if draft is not None and draft.op_id == "geom.crop":
            p = draft.params
            self.viewport.set_crop_overlay(
                True,
                float(p.get("x_pct", 10.0)),
                float(p.get("y_pct", 10.0)),
                float(p.get("w_pct", 80.0)),
                float(p.get("h_pct", 80.0)),
                str(p.get("aspect", "Free")),
            )
        else:
            self.viewport.set_crop_overlay(False)

    def _on_viewport_crop_changed(self, x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> None:
        draft = self._pipeline.draft_step
        if draft is None or draft.op_id != "geom.crop":
            return
        self.param_panel.set_values_externally(
            {"x_pct": x_pct, "y_pct": y_pct, "w_pct": w_pct, "h_pct": h_pct}
        )

    def _show_loading_indicator(self) -> None:
        self.viewport.set_loading(True, self._loading_op_label)

    def _on_edit_step(self, index: int) -> None:
        steps = self._pipeline.steps
        if not (0 <= index < len(steps)):
            return
        step = steps[index]
        spec = self._registry.get(step.op_id)
        self._pipeline.set_draft(step.op_id, step.params, editing_index=index)
        self.param_panel.set_operation(spec, step.params, editing_index=index)
        self._sync_crop_overlay()
        self._sync_sidebar_indicators()
        self._recompute_pipeline()

    def _on_toggle_step(self, index: int) -> None:
        self._pipeline.toggle_step(index)
        self._recompute_pipeline()
        self._update_action_states()

    def _on_move_step(self, index: int, delta: int) -> None:
        self._pipeline.move_step(index, delta)
        self._recompute_pipeline()
        self._update_action_states()

    def _on_duplicate_step(self, index: int) -> None:
        self._pipeline.duplicate_step(index)
        self._recompute_pipeline()
        self._update_action_states()

    def _on_delete_step(self, index: int) -> None:
        self._pipeline.remove_step(index)
        if self._pipeline.draft_step is None:
            self.param_panel.clear_operation()
        self._recompute_pipeline()
        self._update_action_states()

    def undo(self) -> None:
        if self._pipeline.undo():
            self.param_panel.clear_operation()
            self._recompute_pipeline()
            self._update_action_states()

    def redo(self) -> None:
        if self._pipeline.redo():
            self.param_panel.clear_operation()
            self._recompute_pipeline()
            self._update_action_states()

    def reset_pipeline(self, *, confirm: bool = False) -> None:
        if confirm and (self._pipeline.steps or self._pipeline.draft_step is not None):
            count = len(self._pipeline.steps)
            if count > 0:
                msg = f"Discard {count} committed pipeline step(s) and reset to the original image?"
            else:
                msg = "Discard the active operation and reset the preview to the original image?"
            ans = QMessageBox.question(
                self,
                "Reset to Original",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self._pipeline.clear_all()
        self.param_panel.clear_operation()
        self._recompute_pipeline(reset_view=False)
        self._update_action_states()

    def _can_run_fast_in_process(self) -> bool:
        """Return True when interactive preview can evaluate at 60fps in-process without IPC overhead."""
        if self._orig_image is None or self._runner.is_busy:
            return False
        draft = self._pipeline.draft_step
        if draft is not None and is_heavy_op(draft.op_id):
            return False
        if self._orig_image.shape[0] * self._orig_image.shape[1] > 4_500_000:
            return False
        steps = self._pipeline.steps
        if not steps:
            return True
        if self._pipeline.editing_index is None:
            if self._pipeline.cached_prefix_snapshot(len(steps)) is not None:
                return True
            if self._runner.get_cached_committed_snapshot(self._image_token, self._pipeline) is not None:
                return True
        return all((not s.enabled) or (not is_heavy_op(s.op_id)) for s in steps)

    def _update_deferred_metrics(self) -> None:
        if self._orig_image is not None and self._result_image is not None:
            self.histogram_panel.update_images(self._orig_image, self._result_image)
            self._refresh_pipeline_list(self._last_step_results)

    def _recompute_pipeline(
        self,
        *,
        reset_view: bool = False,
        sync: bool = False,
        preempt: bool = True,
        interactive: bool = False,
    ) -> None:
        if self._orig_image is None:
            self._loading_delay_timer.stop()
            self._deferred_metrics_timer.stop()
            self.viewport.set_loading(False)
            self.viewport.set_images(None, None)
            self.histogram_panel.update_images(None, None)
            self._refresh_pipeline_list(None)
            return

        use_sync = (
            sync
            or os.environ.get("DIPS_SYNC_PIPELINE") == "1"
            or (interactive and self._can_run_fast_in_process())
        )
        self._last_eval_interactive = interactive

        if not use_sync:
            draft = self._pipeline.draft_step
            if draft is not None:
                spec = self._registry.get(draft.op_id)
                self._loading_op_label = f"Applying {spec.name}..."
            else:
                self._loading_op_label = "Processing pipeline..."
            self._loading_delay_timer.start()
        else:
            self._loading_delay_timer.stop()
            self.viewport.set_loading(False)

        self._runner.submit(
            self._orig_image,
            self._pipeline,
            image_token=self._image_token,
            reset_view=reset_view,
            force_sync=use_sync,
            preempt=preempt,
        )

    def _on_pipeline_completed(
        self,
        job_id: int,
        res: np.ndarray,
        step_results: list,
        total_ms: float,
        reset_view: bool,
    ) -> None:
        if job_id != self._runner.latest_job_id or self._orig_image is None:
            return
        self._loading_delay_timer.stop()
        self.viewport.set_loading(False)
        self._result_image = res
        self._last_step_results = step_results
        self.viewport.set_images(self._orig_image, res, reset_view=reset_view)
        self._perf_lbl.setText(f"Pipeline: {total_ms:.1f} ms")
        if self._last_eval_interactive:
            self._deferred_metrics_timer.start()
        else:
            self._deferred_metrics_timer.stop()
            self.histogram_panel.update_images(self._orig_image, res)
            self._refresh_pipeline_list(step_results)

        if self._pipeline.draft_step is not None and step_results:
            idx = self._pipeline.editing_index
            active_sr = step_results[idx] if (idx is not None and idx < len(step_results)) else step_results[-1]
            self.param_panel.set_metadata(active_sr.metadata)

    def _on_pipeline_failed(self, job_id: int, err: str) -> None:
        if job_id != self._runner.latest_job_id:
            return
        self._loading_delay_timer.stop()
        self.viewport.set_loading(False)
        self._status_file.setText(f"Operation error: {err}")

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and hasattr(self, "tool_sidebar"):
            search = self.tool_sidebar._search
            if search.hasFocus():
                pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else None
                target = QApplication.widgetAt(pos) if pos is not None else (watched if isinstance(watched, QWidget) else None)
                if target is None or (target is not search and not search.isAncestorOf(target)):
                    search.clearFocus()
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.removeEventFilter(self)
        self._preview_timer.stop()
        self._deferred_metrics_timer.stop()
        self._loading_delay_timer.stop()
        self._runner.shutdown()
        if self._update_thread is not None and self._update_thread.isRunning():
            try:
                self._update_thread.updateAvailable.disconnect()
                self._update_thread.upToDate.disconnect()
                self._update_thread.checkFailed.disconnect()
            except Exception:
                pass
            self._update_thread.requestInterruption()
            self._update_thread.quit()
            self._update_thread = None
        super().closeEvent(event)
        event.accept()
        QApplication.quit()

    def _refresh_pipeline_list(self, step_results) -> None:
        self.pipeline_panel.set_steps(
            self._pipeline.steps,
            step_results,
            editing_index=self._pipeline.editing_index,
        )

    def _sync_sidebar_indicators(self) -> None:
        draft_op = self._pipeline.draft_step.op_id if self._pipeline.draft_step is not None else None
        committed_ops = {s.op_id for s in self._pipeline.steps if s.enabled}
        self.tool_sidebar.set_active_operations(draft_op, committed_ops)

    def _update_action_states(self) -> None:
        has_img = self._orig_image is not None
        self._act_close_img.setEnabled(has_img)
        self._act_save.setEnabled(has_img)
        self._act_export.setEnabled(has_img)
        self._act_copy.setEnabled(has_img)
        self._act_fit.setEnabled(has_img)
        self._act_100.setEnabled(has_img)
        self._act_undo.setEnabled(self._pipeline.can_undo)
        self._act_redo.setEnabled(self._pipeline.can_redo)
        self._act_reset.setEnabled(bool(self._pipeline.steps) or self._pipeline.draft_step is not None)
        self._sync_sidebar_indicators()

    # ------------------------------------------------------------------
    # Presets, Batch, Pixel Inspector, About
    # ------------------------------------------------------------------
    def prompt_save_preset(self) -> None:
        if not self._pipeline.steps:
            QMessageBox.information(self, "Empty Pipeline", "Add at least one pipeline step before saving a preset.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline Preset", str(Path.home() / "pipeline.dips.json"), "DIP Studio Preset (*.json)")
        if path:
            out = self._pipeline.save_preset(path)
            self._status_file.setText(f"Saved preset: {out.name}")

    def prompt_load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline Preset", str(Path.home()), "DIP Studio Preset (*.json)")
        if path:
            self._load_preset_file(Path(path))

    def _load_preset_file(self, path: Path) -> None:
        try:
            count = self._pipeline.load_preset(path)
            self.param_panel.clear_operation()
            self._recompute_pipeline()
            self._update_action_states()
            self._status_file.setText(f"Loaded preset '{path.name}' ({count} steps)")
        except Exception as exc:
            QMessageBox.critical(self, "Preset Error", str(exc))

    def open_batch_dialog(self) -> None:
        BatchDialog(self._pipeline, self).exec()

    def show_about(self) -> None:
        AboutDialog(self).exec()

    def _on_pixel_inspected(self, x: int, y: int, source: str) -> None:
        img = self._orig_image if source == "orig" else self._result_image
        if img is None or x < 0 or y < 0 or y >= img.shape[0] or x >= img.shape[1]:
            self._pixel_lbl.setText("Cursor: -")
            self._swatch.setStyleSheet(
                f"border: 1px solid {self._palette.border_strong}; border-radius: 3px; background: transparent;"
            )
            return

        bdr = self._palette.border_strong
        if img.ndim == 2:
            v = int(img[y, x])
            self._swatch.setStyleSheet(f"border: 1px solid {bdr}; border-radius: 3px; background: rgb({v},{v},{v});")
            self._pixel_lbl.setText(f"({x}, {y})  Gray: {v}")
        else:
            b, g, r = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
            gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
            qc = QColor(r, g, b)
            h, s, v, _ = qc.getHsv()
            h_deg = max(0, h)
            self._swatch.setStyleSheet(f"border: 1px solid {bdr}; border-radius: 3px; background: rgb({r},{g},{b});")
            alpha_str = f"  A:{int(img[y, x, 3])}" if img.shape[2] == 4 else ""
            self._pixel_lbl.setText(
                f"({x}, {y})  RGB({r}, {g}, {b}){alpha_str}  Gray:{gray}  HSV({h_deg}°, {int(s/2.55)}%, {int(v/2.55)}%)"
            )
