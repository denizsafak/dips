"""Non-destructive image processing pipeline with undo/redo, live draft preview, presets, and batch execution."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable
import cv2
import numpy as np

from .io_utils import SUPPORTED_OPEN_EXTS, ensure_uint8, load_image, save_image
from .registry import OperationRegistry, get_registry
from .types import OperationSpec, PipelineStep, StepResult

_ALPHA_AUTHORING_OPS = frozenset({"ai.bg_remove", "ai.sticker_cutout"})
_SPATIAL_ALPHA_OPS = frozenset(
    {
        "basics.flip",
        "basics.rotate90",
        "geom.crop",
        "geom.perspective",
        "geom.lens",
        "geom.pad",
        "geom.polar",
        "affine.rotate",
        "affine.scale",
        "affine.translate",
        "affine.shear",
        "affine.combined",
        "ai.face_auto_crop",
        "feat.document_scan",
    }
)
HEAVY_OPS = frozenset(
    {
        "ai.upscale",
        "ai.bg_remove",
        "ai.bokeh",
        "ai.color_splash",
        "ai.sticker_cutout",
        "ai.face_detect",
        "ai.face_privacy",
        "ai.face_auto_crop",
        "ai.face_enhance",
        "ai.style_transfer",
        "ai.smart_enhance",
        "ai.depth_map",
        "ai.depth_focus",
        "ai.depth_fog",
        "ai.depth_relight",
        "ai.anime_style",
        "ai.object_detect",
        "ai.object_privacy",
        "ai.low_light",
        "restore.nlm",
        "restore.wiener",
        "restore.dehaze",
        "restore.inpaint",
        "freq.spectrum",
        "freq.lowpass",
        "freq.highpass",
        "freq.band",
        "freq.homomorphic",
    }
)


def is_heavy_op(op_id: str) -> bool:
    """Return True if the operation typically warrants off-thread subprocess evaluation."""
    return op_id in HEAVY_OPS


def _clone_step_result(r: StepResult) -> StepResult:
    return StepResult(
        step_id=r.step_id,
        op_id=r.op_id,
        op_name=r.op_name,
        enabled=r.enabled,
        elapsed_ms=r.elapsed_ms,
        metadata=dict(r.metadata),
    )


def _execute_step(spec: OperationSpec, img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one operation while preserving existing alpha transparency across subsequent steps."""
    if img.ndim == 3 and img.shape[2] == 4 and spec.op_id not in _ALPHA_AUTHORING_OPS:
        if spec.op_id in _SPATIAL_ALPHA_OPS:
            out, meta = spec.func(img, params)
            return ensure_uint8(out), meta
        bgr = np.ascontiguousarray(img[:, :, :3])
        alpha = img[:, :, 3]
        out, meta = spec.func(bgr, params)
        out_u8 = ensure_uint8(out)
        if out_u8.ndim == 2:
            out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)
        elif out_u8.shape[2] == 4:
            out_bgr = out_u8[:, :, :3]
        else:
            out_bgr = out_u8
        oh, ow = out_bgr.shape[:2]
        if alpha.shape[0] != oh or alpha.shape[1] != ow:
            alpha = cv2.resize(alpha, (ow, oh), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(np.dstack((out_bgr, alpha))), meta
    out, meta = spec.func(img, params)
    return ensure_uint8(out), meta


class ImagePipeline:
    """Manages an ordered stack of non-destructive operations and undo/redo history."""

    def __init__(self, registry: OperationRegistry | None = None) -> None:
        self.registry = registry or get_registry()
        self._steps: list[PipelineStep] = []
        self._undo_stack: list[list[PipelineStep]] = []
        self._redo_stack: list[list[PipelineStep]] = []
        self._draft_step: PipelineStep | None = None
        self._editing_index: int | None = None
        self._cache_token: int | None = None
        self._prefix_cache: list[tuple[tuple[str, bool, str], np.ndarray, StepResult]] = []

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------
    @property
    def steps(self) -> list[PipelineStep]:
        return list(self._steps)

    @property
    def draft_step(self) -> PipelineStep | None:
        return self._draft_step

    @property
    def editing_index(self) -> int | None:
        return self._editing_index

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[PipelineStep]:
        return [s.clone() for s in self._steps]

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 64:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._steps = self._undo_stack.pop()
        self.clear_draft()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._steps = self._redo_stack.pop()
        self.clear_draft()
        return True

    # ------------------------------------------------------------------
    # Draft & Edit workflow
    # ------------------------------------------------------------------
    def set_draft(self, op_id: str, params: dict[str, Any] | None = None, editing_index: int | None = None) -> PipelineStep:
        spec = self.registry.get(op_id)
        merged = spec.default_params()
        if params:
            merged.update(params)
        self._draft_step = PipelineStep(op_id=op_id, params=merged, enabled=True)
        self._editing_index = editing_index
        return self._draft_step

    def update_draft_params(self, params: dict[str, Any]) -> None:
        if self._draft_step is not None:
            self._draft_step.params = dict(params)

    def clear_draft(self) -> None:
        self._draft_step = None
        self._editing_index = None

    def commit_draft(self) -> int:
        """Commit active draft either as an edit in-place or as a new step appended to the stack."""
        if self._draft_step is None:
            return -1
        self._push_undo()
        committed = self._draft_step.clone()
        if self._editing_index is not None and 0 <= self._editing_index < len(self._steps):
            committed.step_id = self._steps[self._editing_index].step_id
            committed.enabled = self._steps[self._editing_index].enabled
            self._steps[self._editing_index] = committed
            idx = self._editing_index
        else:
            self._steps.append(committed)
            idx = len(self._steps) - 1
        self.clear_draft()
        return idx

    # ------------------------------------------------------------------
    # Stack mutations
    # ------------------------------------------------------------------
    def add_step(self, op_id: str, params: dict[str, Any] | None = None) -> int:
        spec = self.registry.get(op_id)
        merged = spec.default_params()
        if params:
            merged.update(params)
        self._push_undo()
        self._steps.append(PipelineStep(op_id=op_id, params=merged, enabled=True))
        return len(self._steps) - 1

    def remove_step(self, index: int) -> None:
        if 0 <= index < len(self._steps):
            self._push_undo()
            self._steps.pop(index)
            if self._editing_index == index:
                self.clear_draft()
            elif self._editing_index is not None and self._editing_index > index:
                self._editing_index -= 1

    def duplicate_step(self, index: int) -> int:
        if not (0 <= index < len(self._steps)):
            return -1
        self._push_undo()
        dup = self._steps[index].clone()
        import uuid
        dup.step_id = uuid.uuid4().hex[:8]
        self._steps.insert(index + 1, dup)
        return index + 1

    def toggle_step(self, index: int) -> None:
        if 0 <= index < len(self._steps):
            self._push_undo()
            self._steps[index].enabled = not self._steps[index].enabled

    def move_step(self, index: int, delta: int) -> int:
        target = index + delta
        if not (0 <= index < len(self._steps) and 0 <= target < len(self._steps)):
            return index
        self._push_undo()
        self._steps[index], self._steps[target] = self._steps[target], self._steps[index]
        if self._editing_index == index:
            self._editing_index = target
        elif self._editing_index == target:
            self._editing_index = index
        return target

    def clear_all(self) -> None:
        if self._steps:
            self._push_undo()
            self._steps.clear()
        self.clear_draft()

    # ------------------------------------------------------------------
    # Execution & Prefix Caching
    # ------------------------------------------------------------------
    @staticmethod
    def step_signature(step: PipelineStep) -> tuple[str, bool, str]:
        """Return a deterministic cache key for a pipeline step."""
        return (step.op_id, bool(step.enabled), json.dumps(step.params, sort_keys=True, default=str))

    def clear_cache(self) -> None:
        """Invalidate cached intermediate step outputs."""
        self._cache_token = None
        self._prefix_cache.clear()

    def cached_prefix_snapshot(self, count: int) -> tuple[np.ndarray, list[StepResult]] | None:
        """Return a copy of the cached image and StepResults after the first `count` steps."""
        if count <= 0 or len(self._prefix_cache) < count:
            return None
        img = self._prefix_cache[count - 1][1].copy()
        res = [_clone_step_result(self._prefix_cache[i][2]) for i in range(count)]
        return img, res

    def _effective_steps(self, include_draft: bool) -> list[PipelineStep]:
        effective = [s.clone() for s in self._steps]
        if include_draft and self._draft_step is not None:
            if self._editing_index is not None and 0 <= self._editing_index < len(effective):
                draft_copy = self._draft_step.clone()
                draft_copy.enabled = True
                effective[self._editing_index] = draft_copy
            else:
                effective.append(self._draft_step.clone())
        return effective

    def evaluate(
        self,
        original: np.ndarray,
        *,
        include_draft: bool = True,
        image_token: int | None = None,
        start_index: int = 0,
        precomputed_results: list[StepResult] | None = None,
    ) -> tuple[np.ndarray, list[StepResult], float]:
        """Run the pipeline non-destructively (reusing matching cached prefixes when supplied)."""
        effective = self._effective_steps(include_draft)
        t_start = time.perf_counter()

        if start_index > 0:
            current = ensure_uint8(original).copy()
            results = [_clone_step_result(r) for r in (precomputed_results or [])]
            exec_steps = effective[start_index:]
            use_cache = False
        else:
            use_cache = image_token is not None
            if use_cache and self._cache_token != image_token:
                self._prefix_cache.clear()
                self._cache_token = image_token

            match_len = 0
            if use_cache:
                while (
                    match_len < len(self._prefix_cache)
                    and match_len < len(effective)
                    and self._prefix_cache[match_len][0] == self.step_signature(effective[match_len])
                ):
                    match_len += 1
                self._prefix_cache = self._prefix_cache[:match_len]

            if match_len > 0:
                current = self._prefix_cache[match_len - 1][1].copy()
                results = [_clone_step_result(self._prefix_cache[i][2]) for i in range(match_len)]
            else:
                current = ensure_uint8(original).copy()
                results = []
            exec_steps = effective[match_len:]

        for step in exec_steps:
            spec = self.registry.get(step.op_id)
            if not step.enabled:
                sres = StepResult(
                    step_id=step.step_id,
                    op_id=step.op_id,
                    op_name=spec.name,
                    enabled=False,
                    elapsed_ms=0.0,
                    metadata={},
                )
            else:
                t0 = time.perf_counter()
                current, meta = _execute_step(spec, current, step.params)
                elapsed = (time.perf_counter() - t0) * 1000.0
                sres = StepResult(
                    step_id=step.step_id,
                    op_id=step.op_id,
                    op_name=spec.name,
                    enabled=True,
                    elapsed_ms=elapsed,
                    metadata=meta,
                )
            results.append(sres)
            if use_cache:
                self._prefix_cache.append((self.step_signature(step), current.copy(), _clone_step_result(sres)))

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return current, results, total_ms

    # ------------------------------------------------------------------
    # JSON Preset Persistence
    # ------------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        from .. import __version__

        return {
            "app": "DIP Studio",
            "version": __version__,
            "steps": [
                {"op_id": s.op_id, "enabled": s.enabled, "params": s.params, "label": s.label}
                for s in self._steps
            ],
        }

    def save_preset(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")
        return out

    def load_preset_steps(self, raw_steps: list[dict[str, Any]]) -> int:
        loaded: list[PipelineStep] = []
        for item in raw_steps:
            op_id = item.get("op_id", "")
            if op_id not in self.registry:
                continue
            spec = self.registry.get(op_id)
            params = spec.default_params()
            params.update(item.get("params", {}))
            loaded.append(
                PipelineStep(
                    op_id=op_id,
                    params=params,
                    enabled=bool(item.get("enabled", True)),
                    label=str(item.get("label", "")),
                )
            )
        self._push_undo()
        self._steps = loaded
        self.clear_draft()
        return len(loaded)

    def load_preset(self, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.load_preset_steps(data.get("steps", []))

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------
    def batch_process(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        out_ext: str = ".png",
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[Path]:
        """Apply the committed pipeline to every supported image in `input_dir`."""
        src_dir = Path(input_dir)
        dst_dir = Path(output_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_OPEN_EXTS
        )
        outputs: list[Path] = []
        total = len(files)
        for idx, file_path in enumerate(files, 1):
            img = load_image(file_path)
            processed, _, _ = self.evaluate(img, include_draft=False)
            out_file = dst_dir / f"{file_path.stem}_dips{out_ext}"
            save_image(processed, out_file)
            outputs.append(out_file)
            if progress_cb is not None:
                progress_cb(idx, total, file_path.name)
        return outputs
