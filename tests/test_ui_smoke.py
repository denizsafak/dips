"""Offscreen PyQt6 smoke tests for DIP Studio MainWindow, subprocess pipeline, and dialogs."""

from __future__ import annotations

import os
import time
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox
from dips.core import ImagePipeline, discover_sample_images, load_image
from dips.ui.main_window import MainWindow
from dips.ui.pipeline_worker import AsyncPipelineRunner


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()
    from dips.core.ops_ai import release_ai_sessions
    release_ai_sessions()
    app.quit()
    app.processEvents()


def test_main_window_smoke_and_operations(qapp) -> None:
    os.environ["DIPS_SYNC_PIPELINE"] = "1"
    win = None
    try:
        samples = discover_sample_images()
        assert len(samples) >= 1
        win = MainWindow(initial_image=samples[0])
        win.set_theme("dark")
        win.show()
        qapp.processEvents()

        # Select an operation -> draft preview -> commit -> switch views -> toggle theme
        win._on_operation_selected("intensity.gamma")
        win.param_panel.paramsChanged.emit({"gamma": 0.6, "gain": 1.0})
        win._recompute_pipeline(sync=True)
        win._on_commit_draft()
        assert len(win._pipeline.steps) == 1

        # Background removal returns 4-channel BGRA transparency
        win._on_operation_selected("ai.bg_remove")
        win._recompute_pipeline(sync=True)
        win._on_commit_draft()
        assert win._result_image is not None and win._result_image.shape[2] == 4

        for mode in range(4):
            win.viewport.set_view_mode(mode)
            qapp.processEvents()

        win.toggle_theme()
        qapp.processEvents()
        assert win._theme_mode == "light"
        assert qapp.palette().window().color().lightness() > 200
        assert qapp.palette().base().color().lightness() > 240

        # Search input loses focus when clicking outside it
        from PyQt6.QtCore import QEvent, QPointF, Qt
        from PyQt6.QtGui import QMouseEvent
        win.tool_sidebar._search.setFocus()
        qapp.processEvents()
        ev = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(10, 10),
            QPointF(win.viewport.mapToGlobal(win.viewport.rect().topLeft())),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        win.eventFilter(win.viewport, ev)
        qapp.processEvents()
        assert not win.tool_sidebar._search.hasFocus()

        # Responsive histogram layout reflows columns cleanly
        from dips.ui.widgets.histogram_widget import HistogramPanel
        hp = HistogramPanel()
        hp.show()
        hp.resize(220, 220)
        qapp.processEvents()
        assert hp._stat_cols == 2
        hp.resize(360, 240)
        qapp.processEvents()
        assert hp._stat_cols == 4
        hp.close()
        hp.deleteLater()

        # Interactive crop sync
        win._on_operation_selected("geom.crop")
        win._on_viewport_crop_changed(15.0, 20.0, 60.0, 50.0)
        win._preview_timer.stop()
        assert win._pipeline.draft_step is not None
        assert win._pipeline.draft_step.params["w_pct"] == 60.0

        # Descending pipeline order (latest step listed first) & Bokeh RMBG-1.4 default
        bokeh_spec = win._registry.get("ai.bokeh")
        assert bokeh_spec.default_params()["model"].startswith("RMBG-1.4")
        win._on_operation_quick_applied("intensity.gamma")
        win._on_operation_quick_applied("intensity.negative")
        qapp.processEvents()
        row_order = [
            win.pipeline_panel._list.item(i).data(0x0100)  # Qt.ItemDataRole.UserRole
            for i in range(win.pipeline_panel._list.count())
        ]
        assert row_order == sorted(row_order, reverse=True) and len(row_order) >= 3

        # Toggle-to-discard via left sidebar & splitter safety
        assert win._main_splitter.childrenCollapsible() is False
        assert win._right_splitter.childrenCollapsible() is False
        from dips.ui.theme import detect_system_theme
        assert detect_system_theme() in ("dark", "light")

        win._on_operation_selected("intensity.negative")
        assert win._pipeline.draft_step is not None
        item_neg = win.tool_sidebar._op_items["intensity.negative"]
        win.tool_sidebar._on_item_clicked(item_neg, 0)
        assert win._pipeline.draft_step is None

        # Upscaled and downscaled results both display at equal visual scale when zooming
        import numpy as np
        win.viewport.set_view_mode(0)
        orig_small = np.zeros((100, 100, 3), dtype=np.uint8)
        res_3x = np.zeros((300, 300, 3), dtype=np.uint8)
        win.viewport.set_images(orig_small, res_3x, reset_view=True)
        win.viewport.zoom_100()
        qapp.processEvents()
        assert abs(win.viewport.orig_canvas.current_zoom() * 100 - win.viewport.res_canvas.current_zoom() * 300) < 1e-3

        res_half = np.zeros((50, 50, 3), dtype=np.uint8)
        win.viewport.set_images(orig_small, res_half, reset_view=True)
        win.viewport.zoom_100()
        qapp.processEvents()
        assert abs(win.viewport.orig_canvas.current_zoom() * 100 - win.viewport.res_canvas.current_zoom() * 50) < 1e-3

        # Reset confirmation preserves pipeline on No and clears on Yes
        called_prompts: list[str] = []
        orig_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(lambda *_args, **_kw: called_prompts.append("No") or QMessageBox.StandardButton.No)
            win.reset_pipeline(confirm=True)
            assert len(called_prompts) == 1
            assert len(win._pipeline.steps) >= 3

            QMessageBox.question = staticmethod(lambda *_args, **_kw: called_prompts.append("Yes") or QMessageBox.StandardButton.Yes)
            win.reset_pipeline(confirm=True)
            assert len(called_prompts) == 2
            assert len(win._pipeline.steps) == 0
        finally:
            QMessageBox.question = orig_question

        # Close current image returns cleanly to welcome page
        win.close_current_image(confirm=False)
        qapp.processEvents()
        assert win._orig_image is None
        assert win.viewport._stack.currentIndex() == 0
    finally:
        if win is not None:
            win.close()
            win.deleteLater()
            qapp.processEvents()
        os.environ.pop("DIPS_SYNC_PIPELINE", None)


def test_subprocess_pipeline_runner_and_preemption(qapp) -> None:
    os.environ.pop("DIPS_SYNC_PIPELINE", None)
    samples = discover_sample_images()
    img = load_image(next(p for p in samples if p.name == "lena.png"))
    pipe_heavy = ImagePipeline()
    pipe_heavy.add_step("ai.upscale", {"model": "4x-UltraSharp (High Quality)", "scale": "4×"})

    pipe_fast = ImagePipeline()
    pipe_fast.add_step("intensity.negative", {})

    runner = AsyncPipelineRunner()
    done_payload: list[tuple] = []
    runner.completed.connect(lambda jid, res, sr, ms, rv: done_payload.append((jid, res, sr, ms, rv)))

    try:
        # Start heavy job, then immediately preempt with fast job
        heavy_id = runner.submit(img, pipe_heavy, reset_view=False, force_sync=False)
        time.sleep(0.05)
        fast_id = runner.submit(img, pipe_fast, reset_view=False, force_sync=False)

        deadline = time.monotonic() + 15.0
        while not done_payload and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)

        assert len(done_payload) == 1
        assert done_payload[0][0] == fast_id
        assert done_payload[0][0] != heavy_id
        assert done_payload[0][1].shape == img.shape
    finally:
        runner.shutdown()
