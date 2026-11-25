"""Instantly cancellable subprocess pipeline executor using multiprocessing Pipe."""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
import os
from typing import Any
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..core.pipeline import ImagePipeline
from ..core.registry import get_registry
from ..core.types import PipelineStep, StepResult


def _worker_loop(conn: Connection) -> None:
    """Persistent child process evaluating pipeline jobs with prefix caching until closed or terminated."""
    registry = get_registry()
    pipe = ImagePipeline(registry)
    cached_token: int | None = None
    cached_original: np.ndarray | None = None

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        (
            job_id,
            image_token,
            original_or_none,
            steps_payload,
            draft_payload,
            editing_index,
            reset_view,
            start_index,
            precomputed_results,
            need_committed_snapshot,
        ) = msg
        try:
            if start_index > 0 and original_or_none is not None:
                base_img = original_or_none
            else:
                if original_or_none is not None:
                    cached_original = original_or_none
                    cached_token = image_token
                if cached_original is None or cached_token != image_token:
                    raise RuntimeError("Worker original image missing for requested token")
                base_img = cached_original

            pipe._steps = [
                PipelineStep(
                    op_id=item["op_id"],
                    params=item["params"],
                    enabled=item["enabled"],
                    step_id=item["step_id"],
                    label=item.get("label", ""),
                )
                for item in steps_payload
            ]
            if draft_payload is not None:
                pipe.set_draft(draft_payload["op_id"], draft_payload["params"], editing_index=editing_index)
            else:
                pipe.clear_draft()

            res, sres, ms = pipe.evaluate(
                base_img,
                include_draft=True,
                image_token=image_token if start_index == 0 else None,
                start_index=start_index,
                precomputed_results=precomputed_results,
            )
            committed_snap = None
            if need_committed_snapshot and editing_index is None and len(steps_payload) > 0 and start_index == 0:
                committed_snap = pipe.cached_prefix_snapshot(len(steps_payload))
            conn.send(("ok", job_id, res, sres, ms, reset_view, committed_snap))
        except Exception as exc:
            try:
                conn.send(("err", job_id, str(exc)))
            except (BrokenPipeError, OSError):
                break


class AsyncPipelineRunner(QObject):
    """Runs pipeline evaluations in a dedicated spawned subprocess with instant preemption and prefix caching."""

    completed = pyqtSignal(int, object, list, float, bool)  # job_id, result_img, step_results, ms, reset_view
    failed = pyqtSignal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: multiprocessing.Process | None = None
        self._conn: Connection | None = None
        self._latest_job_id: int = 0
        self._busy: bool = False
        self._shutting_down: bool = False
        self._worker_image_token: int | None = None
        self._committed_cache: tuple[int, tuple[tuple[str, bool, str], ...], np.ndarray, list[StepResult]] | None = None
        self._pending_committed_meta: dict[int, tuple[int, tuple[tuple[str, bool, str], ...]]] = {}
        self._queued_job: tuple | None = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(15)
        self._poll_timer.timeout.connect(self._poll_worker)

    @property
    def latest_job_id(self) -> int:
        return self._latest_job_id

    @property
    def is_busy(self) -> bool:
        return self._busy

    def clear_cache(self) -> None:
        """Clear cached base/committed snapshots when switching or closing an image."""
        self._committed_cache = None
        self._worker_image_token = None
        self._pending_committed_meta.clear()
        self._queued_job = None

    def get_cached_committed_snapshot(
        self, image_token: int, pipeline: ImagePipeline
    ) -> tuple[np.ndarray, list[StepResult]] | None:
        """Return cached output of committed steps if image token and step signatures match."""
        if pipeline.editing_index is not None or not pipeline.steps or self._committed_cache is None:
            return None
        committed_sigs = tuple(ImagePipeline.step_signature(s) for s in pipeline.steps)
        if self._committed_cache[0] == image_token and self._committed_cache[1] == committed_sigs:
            return self._committed_cache[2], self._committed_cache[3]
        return None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            return
        self._close_worker_handles()
        self._worker_image_token = None
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_worker_loop, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        self._proc = proc
        self._conn = parent_conn

    def _close_worker_handles(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._proc = None
        self._worker_image_token = None

    def _kill_busy_worker(self) -> None:
        """Immediately terminate a busy child process so new/cancelled jobs never wait."""
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=0.2)
                    if self._proc.is_alive():
                        self._proc.kill()
                        self._proc.join(timeout=0.1)
            except Exception:
                pass
        self._close_worker_handles()
        self._busy = False
        self._poll_timer.stop()

    @staticmethod
    def _serialize_pipeline(pipeline: ImagePipeline) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int | None]:
        steps_payload = [
            {
                "op_id": s.op_id,
                "params": s.params,
                "enabled": s.enabled,
                "step_id": s.step_id,
                "label": s.label,
            }
            for s in pipeline.steps
        ]
        draft = pipeline.draft_step
        draft_payload = {"op_id": draft.op_id, "params": draft.params} if draft is not None else None
        return steps_payload, draft_payload, pipeline.editing_index

    def _dispatch_to_worker(
        self,
        job_id: int,
        original: np.ndarray,
        steps_payload: list[dict[str, Any]],
        draft_payload: dict[str, Any] | None,
        editing_idx: int | None,
        committed_sigs: tuple[tuple[str, bool, str], ...],
        image_token: int,
        reset_view: bool,
    ) -> None:
        self._ensure_worker()
        assert self._conn is not None

        start_index = 0
        precomputed_results = None
        original_or_none: np.ndarray | None = None
        need_committed_snapshot = False

        if (
            editing_idx is None
            and len(steps_payload) > 0
            and self._committed_cache is not None
            and self._committed_cache[0] == image_token
            and self._committed_cache[1] == committed_sigs
        ):
            start_index = len(steps_payload)
            original_or_none = self._committed_cache[2]
            precomputed_results = self._committed_cache[3]
        else:
            if self._worker_image_token != image_token:
                original_or_none = original
                self._worker_image_token = image_token
            if editing_idx is None and len(steps_payload) > 0:
                need_committed_snapshot = True
                self._pending_committed_meta[job_id] = (image_token, committed_sigs)

        self._busy = True
        self._conn.send(
            (
                job_id,
                image_token,
                original_or_none,
                steps_payload,
                draft_payload,
                editing_idx,
                reset_view,
                start_index,
                precomputed_results,
                need_committed_snapshot,
            )
        )
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def submit(
        self,
        original: np.ndarray,
        pipeline: ImagePipeline,
        *,
        image_token: int = 0,
        reset_view: bool = False,
        force_sync: bool = False,
        preempt: bool = True,
    ) -> int:
        """Submit pipeline evaluation; reuses cached prefixes and preempts or coalesces in-flight jobs."""
        self._latest_job_id += 1
        job_id = self._latest_job_id
        steps_payload, draft_payload, editing_idx = self._serialize_pipeline(pipeline)
        committed_sigs = tuple(ImagePipeline.step_signature(s) for s in pipeline.steps)

        if force_sync or os.environ.get("DIPS_SYNC_PIPELINE") == "1":
            self._queued_job = None
            if self._busy:
                self._kill_busy_worker()
            try:
                if (
                    editing_idx is None
                    and len(steps_payload) > 0
                    and self._committed_cache is not None
                    and self._committed_cache[0] == image_token
                    and self._committed_cache[1] == committed_sigs
                ):
                    res, sres, ms = pipeline.evaluate(
                        self._committed_cache[2],
                        include_draft=True,
                        start_index=len(steps_payload),
                        precomputed_results=self._committed_cache[3],
                    )
                else:
                    res, sres, ms = pipeline.evaluate(original, include_draft=True, image_token=image_token)
                    if editing_idx is None and len(steps_payload) > 0:
                        snap = pipeline.cached_prefix_snapshot(len(steps_payload))
                        if snap is not None:
                            self._committed_cache = (image_token, committed_sigs, snap[0], snap[1])
                self.completed.emit(job_id, res, sres, ms, reset_view)
            except Exception as exc:
                self.failed.emit(job_id, str(exc))
            return job_id

        if self._busy and not preempt:
            # Coalesce rapid slider updates onto the warm worker without killing its loaded ONNX session
            self._queued_job = (
                job_id,
                original,
                steps_payload,
                draft_payload,
                editing_idx,
                committed_sigs,
                image_token,
                reset_view,
            )
            return job_id

        self._queued_job = None
        if self._busy:
            self._kill_busy_worker()

        self._dispatch_to_worker(
            job_id,
            original,
            steps_payload,
            draft_payload,
            editing_idx,
            committed_sigs,
            image_token,
            reset_view,
        )
        return job_id

    def _poll_worker(self) -> None:
        if self._shutting_down or self._conn is None:
            self._poll_timer.stop()
            return
        try:
            while self._conn is not None and self._conn.poll(0):
                payload = self._conn.recv()
                self._busy = False
                kind = payload[0]
                job_id = payload[1]
                pending_meta = self._pending_committed_meta.pop(job_id, None)
                if kind == "ok":
                    _, _, res, sres, ms, reset_view, committed_snap = payload
                    if committed_snap is not None and pending_meta is not None:
                        img_tok, c_sigs = pending_meta
                        self._committed_cache = (img_tok, c_sigs, committed_snap[0], committed_snap[1])
                    if job_id == self._latest_job_id:
                        self.completed.emit(job_id, res, sres, ms, reset_view)
                else:
                    if job_id == self._latest_job_id:
                        self.failed.emit(job_id, payload[2])

                if self._queued_job is not None and not self._shutting_down:
                    queued = self._queued_job
                    self._queued_job = None
                    self._dispatch_to_worker(*queued)
                else:
                    self._poll_timer.stop()
        except (EOFError, OSError):
            self._busy = False
            self._queued_job = None
            self._poll_timer.stop()
            self._close_worker_handles()

    def cancel_current(self) -> None:
        """Cancel any active/pending subprocess job immediately."""
        self._latest_job_id += 1
        self._queued_job = None
        if self._busy:
            self._kill_busy_worker()
        else:
            self._poll_timer.stop()

    def shutdown(self) -> None:
        self._shutting_down = True
        self._queued_job = None
        self._poll_timer.stop()
        self._kill_busy_worker()
