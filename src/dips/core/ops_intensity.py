"""Point-wise intensity transformations (Chapter 3) with exact 256-entry LUT helpers."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def compute_lut(op_id: str, params: dict[str, Any]) -> np.ndarray | None:
    """Return a 256-element uint8 LUT s = T(r) for curve-capable intensity operations."""
    r = np.arange(256, dtype=np.float32)

    if op_id == "intensity.negative":
        return (255 - r).astype(np.uint8)

    if op_id == "intensity.gamma":
        gamma = max(0.01, float(params.get("gamma", 1.0)))
        gain = float(params.get("gain", 1.0))
        s = gain * np.power(r / 255.0, gamma) * 255.0
        return np.clip(np.round(s), 0, 255).astype(np.uint8)

    if op_id == "intensity.log":
        c = float(params.get("c", 1.0))
        s = c * (255.0 / np.log1p(255.0)) * np.log1p(r)
        return np.clip(np.round(s), 0, 255).astype(np.uint8)

    if op_id == "intensity.brightness_contrast":
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 1.0))
        s = contrast * (r - 127.5) + 127.5 + brightness
        return np.clip(np.round(s), 0, 255).astype(np.uint8)

    if op_id == "intensity.contrast_stretch":
        mode = params.get("mode", "Piecewise Linear (r1,s1,r2,s2)")
        if mode != "Piecewise Linear (r1,s1,r2,s2)":
            return None
        r1 = float(params.get("r1", 50))
        s1 = float(params.get("s1", 15))
        r2 = max(r1 + 1.0, float(params.get("r2", 200)))
        s2 = float(params.get("s2", 240))
        s = np.empty_like(r)
        m1 = r < r1
        m2 = (r >= r1) & (r <= r2)
        m3 = r > r2
        s[m1] = (s1 / max(r1, 1e-6)) * r[m1] if r1 > 0 else s1
        s[m2] = ((s2 - s1) / (r2 - r1)) * (r[m2] - r1) + s1
        s[m3] = ((255.0 - s2) / max(255.0 - r2, 1e-6)) * (r[m3] - r2) + s2 if r2 < 255 else s2
        return np.clip(np.round(s), 0, 255).astype(np.uint8)

    if op_id == "intensity.slicing":
        low = int(params.get("low", 100))
        high = int(params.get("high", 180))
        if low > high:
            low, high = high, low
        preserve = bool(params.get("preserve_background", True))
        hi_val = int(params.get("highlight_val", 255))
        bg_val = int(params.get("bg_val", 0))
        s = r.copy() if preserve else np.full_like(r, bg_val)
        s[(r >= low) & (r <= high)] = hi_val
        return np.clip(s, 0, 255).astype(np.uint8)

    return None


def op_negative(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute photographic negative s = 255 - r (preserving alpha if BGRA)."""
    img = ensure_uint8(img)
    if img.ndim == 3 and img.shape[2] == 4:
        out = img.copy()
        out[:, :, :3] = 255 - out[:, :, :3]
        return out, {"formula": "s = (L - 1) - r"}
    return 255 - img, {"formula": "s = (L - 1) - r"}


def op_gamma(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Power-law (gamma) transformation s = c * r^gamma."""
    img = ensure_uint8(img)
    gamma = max(0.01, float(params.get("gamma", 1.0)))
    gain = float(params.get("gain", 1.0))
    lut = compute_lut("intensity.gamma", {"gamma": gamma, "gain": gain})
    assert lut is not None
    return cv2.LUT(img, lut), {"gamma": round(gamma, 3), "gain": round(gain, 2)}


def op_log(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Logarithmic dynamic-range compression s = c * log(1 + r)."""
    img = ensure_uint8(img)
    c = float(params.get("c", 1.0))
    lut = compute_lut("intensity.log", {"c": c})
    assert lut is not None
    return cv2.LUT(img, lut), {"c": round(c, 2)}


def op_contrast_stretch(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Contrast stretching via Min-Max, Percentile Clip, or Piecewise Linear (r1,s1,r2,s2)."""
    img = ensure_uint8(img)
    mode = str(params.get("mode", "Auto Min-Max"))

    if mode == "Piecewise Linear (r1,s1,r2,s2)":
        lut = compute_lut("intensity.contrast_stretch", params)
        assert lut is not None
        r1 = int(params.get("r1", 50))
        s1 = int(params.get("s1", 15))
        r2 = max(r1 + 1, int(params.get("r2", 200)))
        s2 = int(params.get("s2", 240))
        return cv2.LUT(img, lut), {"mode": "Piecewise", "points": f"({r1},{s1}) → ({r2},{s2})"}

    p_low = 0.0 if mode == "Auto Min-Max" else float(params.get("low_pct", 2.0))
    p_high = 100.0 if mode == "Auto Min-Max" else float(params.get("high_pct", 98.0))
    if p_high <= p_low:
        p_high = min(100.0, p_low + 1.0)

    def _stretch_channel(ch: np.ndarray) -> tuple[np.ndarray, float, float]:
        lo = float(np.percentile(ch, p_low))
        hi = float(np.percentile(ch, p_high))
        if hi <= lo + 1e-6:
            return ch.copy(), lo, hi
        scaled = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
        return np.clip(np.round(scaled), 0, 255).astype(np.uint8), lo, hi

    if img.ndim == 2 or not bool(params.get("per_channel", False)):
        if img.ndim == 2:
            out, lo, hi = _stretch_channel(img)
            return out, {"in_range": f"[{lo:.1f}, {hi:.1f}]", "out_range": "[0, 255]"}
        # Stretch luminance or joint range while preserving hue
        lo = float(np.percentile(img, p_low))
        hi = float(np.percentile(img, p_high))
        if hi <= lo + 1e-6:
            return img.copy(), {"in_range": f"[{lo:.1f}, {hi:.1f}]"}
        out = np.clip(np.round((img.astype(np.float32) - lo) * (255.0 / (hi - lo))), 0, 255).astype(np.uint8)
        return out, {"in_range": f"[{lo:.1f}, {hi:.1f}]", "out_range": "[0, 255]"}

    channels = [_stretch_channel(img[:, :, c])[0] for c in range(3)]
    return cv2.merge(channels), {"mode": f"{mode} (per-channel)"}


def op_brightness_contrast(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Linear brightness (beta) and contrast (alpha) gain/bias adjustment."""
    img = ensure_uint8(img)
    lut = compute_lut("intensity.brightness_contrast", params)
    assert lut is not None
    return cv2.LUT(img, lut), {
        "brightness": int(params.get("brightness", 0)),
        "contrast": round(float(params.get("contrast", 1.0)), 2),
    }


def op_intensity_slicing(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Highlight a specific [low, high] intensity band (with or without background)."""
    img = ensure_uint8(img)
    lut = compute_lut("intensity.slicing", params)
    assert lut is not None
    low = int(params.get("low", 100))
    high = int(params.get("high", 180))
    return cv2.LUT(img, lut), {"band": f"[{min(low, high)}, {max(low, high)}]"}


def op_bit_plane(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a single bit plane (0=LSB .. 7=MSB) from the image."""
    img = ensure_uint8(img)
    bit = int(np.clip(params.get("bit", 7), 0, 7))
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plane = ((gray >> bit) & 1) * 255
    return plane.astype(np.uint8), {"bit_plane": bit, "weight": 1 << bit}


def get_specs() -> list[OperationSpec]:
    """Return all Intensity Transformation operation specifications."""
    return [
        OperationSpec(
            op_id="intensity.contrast_stretch",
            name="Contrast Stretching",
            category=OperationCategory.INTENSITY,
            description="Expand dynamic range via Min-Max, Percentile clipping, or Piecewise Linear (r1,s1,r2,s2).",
            func=op_contrast_stretch,
            preview_widget="curve",
            keywords=("contrast", "stretch", "minmax", "piecewise", "normalize"),
            params=(
                ParamSpec(
                    "mode",
                    "Mode",
                    ParamType.CHOICE,
                    "Auto Min-Max",
                    choices=("Auto Min-Max", "Percentile Clip", "Piecewise Linear (r1,s1,r2,s2)"),
                ),
                ParamSpec("low_pct", "Low Percentile", ParamType.FLOAT, 2.0, 0.0, 25.0, 0.5, 1, unit="%", group="Percentile"),
                ParamSpec("high_pct", "High Percentile", ParamType.FLOAT, 98.0, 75.0, 100.0, 0.5, 1, unit="%", group="Percentile"),
                ParamSpec("per_channel", "Stretch RGB Channels Independently", ParamType.BOOL, False),
                ParamSpec("r1", "Input r1", ParamType.INT, 50, 0, 254, 1, group="Piecewise Linear"),
                ParamSpec("s1", "Output s1", ParamType.INT, 15, 0, 255, 1, group="Piecewise Linear"),
                ParamSpec("r2", "Input r2", ParamType.INT, 200, 1, 255, 1, group="Piecewise Linear"),
                ParamSpec("s2", "Output s2", ParamType.INT, 240, 0, 255, 1, group="Piecewise Linear"),
            ),
        ),
        OperationSpec(
            op_id="intensity.gamma",
            name="Gamma Correction",
            category=OperationCategory.INTENSITY,
            description="Power-law transformation s = c · r^γ (γ < 1 brightens shadows, γ > 1 deepens contrast).",
            func=op_gamma,
            preview_widget="curve",
            keywords=("gamma", "power law", "exposure", "midtones"),
            params=(
                ParamSpec("gamma", "Gamma (γ)", ParamType.FLOAT, 1.0, 0.10, 5.00, 0.05, 2),
                ParamSpec("gain", "Gain (c)", ParamType.FLOAT, 1.0, 0.25, 2.00, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="intensity.negative",
            name="Image Negative",
            category=OperationCategory.INTENSITY,
            description="Invert pixel intensities using s = 255 − r.",
            func=op_negative,
            preview_widget="curve",
            keywords=("negative", "invert", "complement"),
            params=(),
        ),
        OperationSpec(
            op_id="intensity.log",
            name="Log Transform",
            category=OperationCategory.INTENSITY,
            description="Expand dark pixel values and compress high intensities via s = c · log(1 + r).",
            func=op_log,
            preview_widget="curve",
            keywords=("log", "logarithm", "dynamic range"),
            params=(
                ParamSpec("c", "Scale Factor (c)", ParamType.FLOAT, 1.0, 0.2, 2.5, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="intensity.brightness_contrast",
            name="Brightness & Contrast",
            category=OperationCategory.INTENSITY,
            description="Linear affine intensity adjustment: s = α(r − 127.5) + 127.5 + β.",
            func=op_brightness_contrast,
            preview_widget="curve",
            keywords=("brightness", "contrast", "exposure", "linear"),
            params=(
                ParamSpec("brightness", "Brightness (β)", ParamType.INT, 0, -150, 150, 1),
                ParamSpec("contrast", "Contrast (α)", ParamType.FLOAT, 1.0, 0.1, 3.0, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="intensity.slicing",
            name="Intensity-Level Slicing",
            category=OperationCategory.INTENSITY,
            description="Isolate or highlight pixels within [Low, High] intensity range.",
            func=op_intensity_slicing,
            preview_widget="curve",
            keywords=("slice", "band", "highlight", "window"),
            params=(
                ParamSpec("low", "Low Threshold", ParamType.INT, 100, 0, 255, 1),
                ParamSpec("high", "High Threshold", ParamType.INT, 180, 0, 255, 1),
                ParamSpec("preserve_background", "Preserve Background", ParamType.BOOL, True),
                ParamSpec("highlight_val", "Highlight Value", ParamType.INT, 255, 0, 255, 1),
                ParamSpec("bg_val", "Background Value", ParamType.INT, 0, 0, 255, 1),
            ),
        ),
        OperationSpec(
            op_id="intensity.bit_plane",
            name="Bit-Plane Slicing",
            category=OperationCategory.INTENSITY,
            description="Inspect individual binary bit planes (0 = LSB noise, 7 = MSB structure).",
            func=op_bit_plane,
            keywords=("bit plane", "msb", "lsb", "binary", "chapter 3"),
            params=(
                ParamSpec("bit", "Bit Plane (0–7)", ParamType.INT, 7, 0, 7, 1),
            ),
        ),
    ]
