"""Pure NumPy/OpenCV image processing engine for DIPS (zero Qt dependencies)."""

from .types import (
    ImageStats,
    OperationCategory,
    OperationSpec,
    ParamSpec,
    ParamType,
    PipelineStep,
    StepResult,
)
from .registry import OperationRegistry, get_registry
from .pipeline import ImagePipeline
from .io_utils import discover_sample_images, load_image, save_image
from .ops_histogram import compute_quality_metrics
from .presets import BUILTIN_PRESETS

__all__ = [
    "ImageStats",
    "OperationCategory",
    "OperationSpec",
    "ParamSpec",
    "ParamType",
    "PipelineStep",
    "StepResult",
    "OperationRegistry",
    "get_registry",
    "ImagePipeline",
    "discover_sample_images",
    "load_image",
    "save_image",
    "compute_quality_metrics",
    "BUILTIN_PRESETS",
]
