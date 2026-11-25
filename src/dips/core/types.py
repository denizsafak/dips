"""Core domain types for DIP Studio image processing operations and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import uuid
import numpy as np


class OperationCategory(str, Enum):
    """Logical operation categories across the image processing pipeline."""

    AI = "AI Studio"
    BASICS = "Basics"
    AFFINE = "Affine Transforms"
    GEOMETRY = "Crop, Perspective & Lens"
    INTENSITY = "Intensity"
    SPATIAL = "Spatial Filters"
    FREQUENCY = "Frequency Domain (FFT)"
    HISTOGRAM = "Histogram"
    MORPHOLOGY = "Morphology & Threshold"
    RESTORATION = "Noise & Restoration"
    COLOR = "Color Spaces"
    ARTISTIC = "Tone & Stylization"
    FEATURES = "Segmentation & Features"


class ParamType(str, Enum):
    """Supported parameter UI control types."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    KERNEL = "kernel"


@dataclass(frozen=True)
class ParamSpec:
    """Declarative specification for a single operation parameter."""

    name: str
    label: str
    param_type: ParamType
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float = 1.0
    decimals: int = 2
    choices: tuple[str, ...] = ()
    odd_only: bool = False
    unit: str = ""
    tooltip: str = ""
    group: str = ""


OpFunc = Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, dict[str, Any]]]


@dataclass(frozen=True)
class OperationSpec:
    """Metadata and implementation handle for a registered DIP operation."""

    op_id: str
    name: str
    category: OperationCategory
    description: str
    func: OpFunc
    params: tuple[ParamSpec, ...] = ()
    preview_widget: str = ""  # "", "curve", "kernel", "se", "affine"
    keywords: tuple[str, ...] = ()

    def default_params(self) -> dict[str, Any]:
        """Return a fresh dict of default parameter values."""
        defaults: dict[str, Any] = {}
        for p in self.params:
            if isinstance(p.default, list):
                defaults[p.name] = [row[:] if isinstance(row, list) else row for row in p.default]
            else:
                defaults[p.name] = p.default
        return defaults


@dataclass
class PipelineStep:
    """A single non-destructive step in an ImagePipeline."""

    op_id: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""

    def clone(self) -> PipelineStep:
        """Create a deep copy of this pipeline step."""
        copied_params: dict[str, Any] = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                copied_params[k] = [row[:] if isinstance(row, list) else row for row in v]
            else:
                copied_params[k] = v
        return PipelineStep(
            op_id=self.op_id,
            params=copied_params,
            enabled=self.enabled,
            step_id=self.step_id,
            label=self.label,
        )


@dataclass
class StepResult:
    """Execution output and diagnostics for a single pipeline step."""

    step_id: str
    op_id: str
    op_name: str
    enabled: bool
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageStats:
    """Summary intensity statistics for an image."""

    width: int
    height: int
    channels: int
    dtype: str
    mean: float
    std: float
    min_val: int
    max_val: int
    median: float
    entropy: float
    is_binary: bool
