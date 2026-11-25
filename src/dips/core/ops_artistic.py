"""Tone mapping, clarity, optical vignette, color toning, and artistic stylization."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_bgr
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def op_clarity(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Edge-aware detail enhancement / local micro-contrast clarity."""
    bgr = to_bgr(img)
    sigma_s = float(params.get("sigma_s", 12.0))
    sigma_r = float(params.get("sigma_r", 0.15))
    out = cv2.detailEnhance(bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    return out, {"sigma_s": sigma_s, "sigma_r": sigma_r}


def op_shadow_highlight(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover clipped shadows and highlights using luminance-guided local tone mapping."""
    bgr = to_bgr(img)
    shadows = float(params.get("shadows", 0.35))      # [-1, 1]
    highlights = float(params.get("highlights", 0.25))  # [-1, 1]
    vibrance = float(params.get("vibrance", 1.10))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    low_mask = np.clip(1.0 - L * 1.6, 0.0, 1.0) ** 2
    high_mask = np.clip((L - 0.35) * 1.6, 0.0, 1.0) ** 2
    # Smooth local masks to avoid halo artifacts
    low_mask = cv2.GaussianBlur(low_mask, (0, 0), 8.0)
    high_mask = cv2.GaussianBlur(high_mask, (0, 0), 8.0)

    L_new = L + shadows * 0.35 * low_mask * (1.0 - L) - highlights * 0.35 * high_mask * L
    lab[:, :, 0] = np.clip(np.round(L_new * 255.0), 0, 255)
    lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * vibrance, 0, 255)
    lab[:, :, 2] = np.clip(128.0 + (lab[:, :, 2] - 128.0) * vibrance, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, {"shadows": shadows, "highlights": highlights}


def op_vignette(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply smooth radial optical light-falloff vignette."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    strength = float(params.get("strength", 0.55))
    radius = max(0.2, float(params.get("radius", 0.75)))

    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = (x * x + y * y) / (radius * radius)
    falloff = np.exp(-strength * r2)
    if img.ndim == 3:
        falloff = falloff[:, :, None]
    out = np.clip(np.round(img.astype(np.float32) * falloff), 0, 255).astype(np.uint8)
    return out, {"strength": strength, "radius": radius}


def op_pencil_sketch(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Photorealistic graphite or colored pencil sketch rendering."""
    bgr = to_bgr(img)
    sigma_s = float(params.get("sigma_s", 55.0))
    sigma_r = float(params.get("sigma_r", 0.07))
    shade = float(params.get("shade_factor", 0.04))
    gray_sketch, color_sketch = cv2.pencilSketch(bgr, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade)
    color_mode = bool(params.get("color_sketch", False))
    return (color_sketch if color_mode else gray_sketch), {"color": color_mode}


def op_stylize(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Non-photorealistic watercolor / painterly edge-aware stylization."""
    bgr = to_bgr(img)
    sigma_s = float(params.get("sigma_s", 60.0))
    sigma_r = float(params.get("sigma_r", 0.45))
    return cv2.stylization(bgr, sigma_s=sigma_s, sigma_r=sigma_r), {"sigma_s": sigma_s, "sigma_r": sigma_r}


def op_cartoon(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Cel-shaded cartoon rendering combining bilateral quantization with adaptive ink edges."""
    bgr = to_bgr(img)
    smooth_passes = max(1, int(params.get("smooth_passes", 2)))
    edge_block = max(3, int(params.get("edge_block", 9)) | 1)
    edge_c = float(params.get("edge_c", 4.0))

    color = bgr.copy()
    for _ in range(smooth_passes):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_block,
        edge_c,
    )
    out = cv2.bitwise_and(color, color, mask=edges)
    return out, {"passes": smooth_passes, "edge_block": edge_block}


def op_chromatic_aberration(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate lateral RGB chromatic fringe shift."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    sx = int(params.get("shift_x", 4))
    sy = int(params.get("shift_y", 0))
    b, g, r = cv2.split(bgr)
    M_r = np.float32([[1, 0, sx], [0, 1, sy]])
    M_b = np.float32([[1, 0, -sx], [0, 1, -sy]])
    r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b_shifted, g, r_shifted]), {"shift": f"({sx}, {sy}) px"}


def op_color_toning(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply classic Sepia, Cyanotype, Warm Film, or Cool Teal color grading."""
    bgr = to_bgr(img).astype(np.float32)
    preset = str(params.get("preset", "Sepia"))
    strength = float(params.get("strength", 0.85))

    gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if preset == "Sepia":
        # Classic BGR sepia matrix
        b = 0.272 * bgr[:, :, 2] + 0.534 * bgr[:, :, 1] + 0.131 * bgr[:, :, 0]
        g = 0.349 * bgr[:, :, 2] + 0.686 * bgr[:, :, 1] + 0.168 * bgr[:, :, 0]
        r = 0.393 * bgr[:, :, 2] + 0.769 * bgr[:, :, 1] + 0.189 * bgr[:, :, 0]
        toned = np.stack([b, g, r], axis=2)
    elif preset == "Cyanotype (Prussian Blue)":
        toned = np.stack([gray * 235.0 + 35.0, gray * 175.0 + 18.0, gray * 95.0], axis=2)
    elif preset == "Warm Film (Kodak)":
        toned = np.stack([bgr[:, :, 0] * 0.88, bgr[:, :, 1] * 1.02, bgr[:, :, 2] * 1.12 + 8.0], axis=2)
    else:  # Cool Teal / Cine
        toned = np.stack([bgr[:, :, 0] * 1.12 + 10.0, bgr[:, :, 1] * 1.04, bgr[:, :, 2] * 0.86], axis=2)

    toned = np.clip(toned, 0, 255)
    blended = bgr * (1.0 - strength) + toned * strength
    return np.clip(np.round(blended), 0, 255).astype(np.uint8), {"preset": preset, "mix": f"{strength * 100:.0f}%"}


def get_specs() -> list[OperationSpec]:
    """Return all Tone & Stylization operation specifications."""
    return [
        OperationSpec(
            op_id="art.clarity",
            name="Detail Enhance (Clarity)",
            category=OperationCategory.ARTISTIC,
            description="Boost crisp local micro-contrast and surface texture without harsh haloing.",
            func=op_clarity,
            keywords=("clarity", "detail", "enhance", "texture", "pop"),
            params=(
                ParamSpec("sigma_s", "Spatial Radius", ParamType.FLOAT, 12.0, 1.0, 80.0, 2.0, 0),
                ParamSpec("sigma_r", "Range Threshold", ParamType.FLOAT, 0.15, 0.02, 0.80, 0.02, 2),
            ),
        ),
        OperationSpec(
            op_id="art.hdr_tone",
            name="Shadows, Highlights & Vibrance",
            category=OperationCategory.ARTISTIC,
            description="Recover dark shadow detail, tame bright highlights, and adjust color vibrance.",
            func=op_shadow_highlight,
            keywords=("shadows", "highlights", "hdr", "tone", "vibrance"),
            params=(
                ParamSpec("shadows", "Lift Shadows", ParamType.FLOAT, 0.35, -1.0, 1.0, 0.05, 2),
                ParamSpec("highlights", "Recover Highlights", ParamType.FLOAT, 0.25, -1.0, 1.0, 0.05, 2),
                ParamSpec("vibrance", "Color Vibrance", ParamType.FLOAT, 1.10, 0.0, 2.0, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="art.vignette",
            name="Optical Vignette",
            category=OperationCategory.ARTISTIC,
            description="Smooth radial corner falloff to draw focus toward the frame center.",
            func=op_vignette,
            keywords=("vignette", "falloff", "lens", "dark corners"),
            params=(
                ParamSpec("strength", "Falloff Strength", ParamType.FLOAT, 0.55, 0.05, 2.5, 0.05, 2),
                ParamSpec("radius", "Clear Radius", ParamType.FLOAT, 0.75, 0.25, 1.5, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="art.toning",
            name="Film & Split Toning",
            category=OperationCategory.ARTISTIC,
            description="Apply Sepia, Cyanotype, Warm Film, or Cool Teal color grading.",
            func=op_color_toning,
            keywords=("sepia", "cyanotype", "film", "teal", "grade", "tone"),
            params=(
                ParamSpec("preset", "Look Preset", ParamType.CHOICE, "Sepia", choices=("Sepia", "Cyanotype (Prussian Blue)", "Warm Film (Kodak)", "Cool Teal (Cine)")),
                ParamSpec("strength", "Mix Strength", ParamType.FLOAT, 0.85, 0.0, 1.0, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="art.pencil",
            name="Pencil Sketch",
            category=OperationCategory.ARTISTIC,
            description="Render photo as a shaded graphite or colored pencil drawing.",
            func=op_pencil_sketch,
            keywords=("pencil", "sketch", "drawing", "graphite"),
            params=(
                ParamSpec("color_sketch", "Colored Pencil Mode", ParamType.BOOL, False),
                ParamSpec("sigma_s", "Stroke Smoothness", ParamType.FLOAT, 55.0, 5.0, 150.0, 5.0, 0),
                ParamSpec("sigma_r", "Edge Sensitivity", ParamType.FLOAT, 0.07, 0.01, 0.40, 0.01, 2),
                ParamSpec("shade_factor", "Shading Density", ParamType.FLOAT, 0.04, 0.01, 0.15, 0.01, 2),
            ),
        ),
        OperationSpec(
            op_id="art.stylize",
            name="Watercolor Stylization",
            category=OperationCategory.ARTISTIC,
            description="Painterly edge-preserving smoothing producing watercolor / oil regions.",
            func=op_stylize,
            keywords=("watercolor", "paint", "stylize", "oil"),
            params=(
                ParamSpec("sigma_s", "Brush Scale", ParamType.FLOAT, 60.0, 5.0, 150.0, 5.0, 0),
                ParamSpec("sigma_r", "Color Merge", ParamType.FLOAT, 0.45, 0.05, 0.95, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="art.cartoon",
            name="Cartoon / Cel Shader",
            category=OperationCategory.ARTISTIC,
            description="Combine bilateral color flattening with crisp adaptive ink contours.",
            func=op_cartoon,
            keywords=("cartoon", "comic", "cel", "ink"),
            params=(
                ParamSpec("smooth_passes", "Bilateral Passes", ParamType.INT, 2, 1, 6, 1),
                ParamSpec("edge_block", "Ink Block Size", ParamType.INT, 9, 3, 25, 2, odd_only=True),
                ParamSpec("edge_c", "Ink Threshold C", ParamType.FLOAT, 4.0, 1.0, 15.0, 0.5, 1),
            ),
        ),
        OperationSpec(
            op_id="art.chromatic",
            name="Chromatic Aberration",
            category=OperationCategory.ARTISTIC,
            description="Introduce lateral Red/Blue optical channel dispersion.",
            func=op_chromatic_aberration,
            keywords=("chromatic", "aberration", "glitch", "rgb shift", "fringe"),
            params=(
                ParamSpec("shift_x", "Horizontal Shift", ParamType.INT, 4, -30, 30, 1, unit="px"),
                ParamSpec("shift_y", "Vertical Shift", ParamType.INT, 0, -30, 30, 1, unit="px"),
            ),
        ),
    ]
