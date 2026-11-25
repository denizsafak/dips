"""Basic image operations: Grayscale, Flip, 90° Rotation, Bit Quantization."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_bgr
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def op_grayscale(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert BGR/BGRA image to grayscale using selectable channel/luminance formulas."""
    img = ensure_uint8(img)
    if img.ndim == 2:
        return img.copy(), {"mode": params.get("method", "Rec.601 (OpenCV)"), "note": "Already grayscale"}

    img = to_bgr(img)

    method = params.get("method", "Rec.601 (OpenCV)")
    keep_3ch = bool(params.get("keep_3ch", False))

    b, g, r = (
        img[:, :, 0].astype(np.float32),
        img[:, :, 1].astype(np.float32),
        img[:, :, 2].astype(np.float32),
    )
    if method == "Rec.709 (HDTV)":
        gray = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0, 255).astype(np.uint8)
    elif method == "Average (R+G+B)/3":
        gray = np.clip((r + g + b) / 3.0, 0, 255).astype(np.uint8)
    elif method == "Lightness (max+min)/2":
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        gray = np.clip(0.5 * (mx + mn), 0, 255).astype(np.uint8)
    elif method == "Red Channel":
        gray = img[:, :, 2].copy()
    elif method == "Green Channel":
        gray = img[:, :, 1].copy()
    elif method == "Blue Channel":
        gray = img[:, :, 0].copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if keep_3ch else gray
    return out, {"method": method, "output_channels": 3 if keep_3ch else 1}


def op_flip(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Flip image horizontally, vertically, or both."""
    img = ensure_uint8(img)
    direction = params.get("direction", "Horizontal")
    code = {"Horizontal": 1, "Vertical": 0, "Both (180°)": -1}.get(direction, 1)
    return cv2.flip(img, code), {"direction": direction}


def op_rotate_90(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate image by exact 90°, 180°, or 270° increments without resampling loss."""
    img = ensure_uint8(img)
    angle = params.get("angle", "90° CW")
    mapping = {
        "90° CW": cv2.ROTATE_90_CLOCKWISE,
        "180°": cv2.ROTATE_180,
        "90° CCW (270°)": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    out = cv2.rotate(img, mapping.get(angle, cv2.ROTATE_90_CLOCKWISE))
    return out, {"angle": angle, "size": f"{out.shape[1]}×{out.shape[0]}"}


def op_quantize(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce intensity bit depth (1–8 bits/channel) to demonstrate quantization (Ch. 2)."""
    img = ensure_uint8(img)
    bits = int(np.clip(params.get("bits", 4), 1, 8))
    levels = 1 << bits
    if bits == 8:
        return img.copy(), {"bits": 8, "levels": 256}
    step = 256.0 / levels
    idx = np.floor(img.astype(np.float32) / step)
    scale = 255.0 / (levels - 1)
    out = np.clip(np.round(idx * scale), 0, 255).astype(np.uint8)
    return out, {"bits": bits, "levels": levels}


def get_specs() -> list[OperationSpec]:
    """Return all Basic operation specifications."""
    return [
        OperationSpec(
            op_id="basics.grayscale",
            name="Grayscale",
            category=OperationCategory.BASICS,
            description="Convert color image to single-channel luminance or channel intensity.",
            func=op_grayscale,
            keywords=("gray", "monochrome", "luminance", "desaturate", "channel"),
            params=(
                ParamSpec(
                    "method",
                    "Formula",
                    ParamType.CHOICE,
                    "Rec.601 (OpenCV)",
                    choices=(
                        "Rec.601 (OpenCV)",
                        "Rec.709 (HDTV)",
                        "Average (R+G+B)/3",
                        "Lightness (max+min)/2",
                        "Red Channel",
                        "Green Channel",
                        "Blue Channel",
                    ),
                    tooltip="Luminance weighting or individual RGB band extraction",
                ),
                ParamSpec(
                    "keep_3ch",
                    "Keep 3 Channels (BGR)",
                    ParamType.BOOL,
                    False,
                    tooltip="Keep 3 identical channels so downstream color overlays still work",
                ),
            ),
        ),
        OperationSpec(
            op_id="basics.flip",
            name="Flip (Mirror)",
            category=OperationCategory.BASICS,
            description="Mirror image horizontally, vertically, or along both axes.",
            func=op_flip,
            keywords=("mirror", "flip", "horizontal", "vertical"),
            params=(
                ParamSpec(
                    "direction",
                    "Axis",
                    ParamType.CHOICE,
                    "Horizontal",
                    choices=("Horizontal", "Vertical", "Both (180°)"),
                ),
            ),
        ),
        OperationSpec(
            op_id="basics.rotate90",
            name="Rotate 90°",
            category=OperationCategory.BASICS,
            description="Lossless 90° clockwise, 180°, or 90° counter-clockwise rotation.",
            func=op_rotate_90,
            keywords=("rotate", "90", "180", "270", "orientation"),
            params=(
                ParamSpec(
                    "angle",
                    "Rotation",
                    ParamType.CHOICE,
                    "90° CW",
                    choices=("90° CW", "180°", "90° CCW (270°)"),
                ),
            ),
        ),
        OperationSpec(
            op_id="basics.quantize",
            name="Intensity Quantization",
            category=OperationCategory.BASICS,
            description="Reduce gray levels to 2^k levels (1–8 bits) to visualize false contouring.",
            func=op_quantize,
            keywords=("bit depth", "quantize", "posterize", "levels", "chapter 2"),
            params=(
                ParamSpec(
                    "bits",
                    "Bit Depth",
                    ParamType.INT,
                    4,
                    min_val=1,
                    max_val=8,
                    step=1,
                    unit="bit",
                    tooltip="Number of bits per channel (2^k intensity levels)",
                ),
            ),
        ),
    ]
