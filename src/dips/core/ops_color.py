"""Color image processing (Chapter 6): Color Space Inspector, HSV/Balance adjustments, Pseudo-Color."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_bgr, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType

_COLORMAPS = {
    "Turbo": cv2.COLORMAP_TURBO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Magma": cv2.COLORMAP_MAGMA,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Jet": cv2.COLORMAP_JET,
    "Hot": cv2.COLORMAP_HOT,
    "Bone": cv2.COLORMAP_BONE,
}


def op_color_channel(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Inspect an individual channel from RGB, HSV, CIE LAB, YCrCb, or CMY."""
    bgr = to_bgr(img)
    space = str(params.get("space", "HSV"))
    channel_idx = int(params.get("channel", 0))
    pseudo = bool(params.get("false_color", False))

    if space == "RGB":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        labels = ("Red", "Green", "Blue")
        plane = rgb[:, :, min(channel_idx, 2)]
    elif space == "HSV":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        labels = ("Hue (0–179)", "Saturation", "Value")
        plane = hsv[:, :, min(channel_idx, 2)]
        if channel_idx == 0:
            plane = np.clip(plane.astype(np.float32) * (255.0 / 179.0), 0, 255).astype(np.uint8)
    elif space == "CIE LAB":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        labels = ("L* (Lightness)", "a* (Green-Red)", "b* (Blue-Yellow)")
        plane = lab[:, :, min(channel_idx, 2)]
    elif space == "YCrCb":
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        labels = ("Y (Luminance)", "Cr (Red-Diff)", "Cb (Blue-Diff)")
        plane = ycrcb[:, :, min(channel_idx, 2)]
    else:  # CMY
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cmy = 255 - rgb
        labels = ("Cyan", "Magenta", "Yellow")
        plane = cmy[:, :, min(channel_idx, 2)]

    label = labels[min(channel_idx, 2)]
    out = cv2.applyColorMap(plane, cv2.COLORMAP_TURBO) if pseudo else plane
    return out, {"space": space, "channel": label}


def op_hsv_adjust(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Adjust Hue shift, Saturation multiplier, and Value multiplier in HSV space."""
    bgr = to_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = float(params.get("hue_shift", 0.0)) / 2.0  # degrees [-180,180] -> OpenCV [0,180)
    s_scale = float(params.get("sat_scale", 1.0))
    v_scale = float(params.get("val_scale", 1.0))

    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, {"hue_deg": int(params.get("hue_shift", 0)), "sat": round(s_scale, 2), "val": round(v_scale, 2)}


def op_color_balance(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Adjust per-channel RGB gain multipliers or apply Gray-World auto white balance."""
    bgr = to_bgr(img).astype(np.float32)
    if bool(params.get("auto_white_balance", False)):
        means = bgr.mean(axis=(0, 1))
        avg = float(means.mean())
        gains = avg / np.maximum(means, 1.0)
        balanced = np.clip(bgr * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return balanced, {"mode": "Gray-World Auto WB"}

    r_g = float(params.get("r_gain", 1.0))
    g_g = float(params.get("g_gain", 1.0))
    b_g = float(params.get("b_gain", 1.0))
    bgr[:, :, 2] *= r_g
    bgr[:, :, 1] *= g_g
    bgr[:, :, 0] *= b_g
    return np.clip(np.round(bgr), 0, 255).astype(np.uint8), {"R/G/B": f"{r_g:.2f}/{g_g:.2f}/{b_g:.2f}"}


def op_pseudocolor(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply intensity-to-false-color LUT colormap (Chapter 6 Pseudo-Color)."""
    gray = to_gray(img)
    cmap_name = str(params.get("colormap", "Turbo"))
    cmap = _COLORMAPS.get(cmap_name, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(gray, cmap), {"colormap": cmap_name}


def op_temperature_tint(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Adjust photographic color temperature (cool-warm), green-magenta tint, and vibrance."""
    bgr = to_bgr(img).astype(np.float32)
    temp = float(params.get("temp", 0.0))
    tint = float(params.get("tint", 0.0))
    vibrance = float(params.get("vibrance", 0.0))

    # Warm shifts Red up / Blue down; Cool shifts Blue up / Red down
    r_shift = 1.0 + (temp / 100.0) * 0.22 - (tint / 100.0) * 0.08
    b_shift = 1.0 - (temp / 100.0) * 0.22 - (tint / 100.0) * 0.08
    g_shift = 1.0 + (tint / 100.0) * 0.16
    bgr[:, :, 2] *= r_shift
    bgr[:, :, 1] *= g_shift
    bgr[:, :, 0] *= b_shift
    out = np.clip(bgr, 0.0, 255.0).astype(np.uint8)

    if abs(vibrance) > 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_norm = hsv[:, :, 1] / 255.0
        # Vibrance boosts muted colors more strongly than already-saturated pixels
        amt = (vibrance / 100.0) * (1.0 - sat_norm) * 0.85
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + amt), 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out, {"temp": f"{temp:+.0f}", "tint": f"{tint:+.0f}", "vibrance": f"{vibrance:+.0f}"}


def op_hue_slice(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Color Slicing / HSV Hue thresholding keeping target hue band and muting or masking outside."""
    bgr = to_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    target_deg = float(params.get("target_hue", 15.0))
    width_deg = float(params.get("hue_width", 25.0))
    min_sat = int(params.get("min_sat", 40))
    bg_mode = str(params.get("outside_mode", "Grayscale"))

    h_center = (target_deg % 360.0) * 0.5
    h_half = max(1.0, width_deg * 0.5)
    hue = hsv[:, :, 0].astype(np.float32)
    diff = np.minimum(np.abs(hue - h_center), 180.0 - np.abs(hue - h_center))
    mask = (diff <= h_half) & (hsv[:, :, 1] >= min_sat)
    pct = round(float(np.count_nonzero(mask)) / max(mask.size, 1) * 100.0, 1)

    if bg_mode == "Binary Mask":
        return np.where(mask, 255, 0).astype(np.uint8), {"selected": f"{pct}%", "hue_deg": int(target_deg)}
    if bg_mode == "Neutral Gray (128)":
        out = bgr.copy()
        out[~mask] = (128, 128, 128)
        return out, {"selected": f"{pct}%", "hue_deg": int(target_deg)}
    if bg_mode == "Black":
        out = np.zeros_like(bgr)
        out[mask] = bgr[mask]
        return out, {"selected": f"{pct}%", "hue_deg": int(target_deg)}

    gray_bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = np.where(mask[:, :, None], bgr, gray_bgr)
    return out, {"selected": f"{pct}%", "hue_deg": int(target_deg)}


def op_skin_detect(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect skin regions in YCrCb space (133<=Cr<=173, 77<=Cb<=127) (Chapter 6)."""
    bgr = to_bgr(img)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    tol = int(params.get("tolerance", 4))
    display = str(params.get("display", "Highlight Overlay"))

    lower = np.array([16, max(0, 133 - tol), max(0, 77 - tol)], dtype=np.uint8)
    upper = np.array([245, min(255, 173 + tol), min(255, 127 + tol)], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    pct = round(float(np.count_nonzero(mask)) / max(mask.size, 1) * 100.0, 1)

    if display == "Binary Mask":
        return mask, {"skin_pct": f"{pct}%"}
    if display == "Skin Cutout":
        return cv2.bitwise_and(bgr, bgr, mask=mask), {"skin_pct": f"{pct}%"}

    out = bgr.copy()
    tint = out.copy()
    tint[mask > 0] = (90, 215, 145)
    cv2.addWeighted(tint, 0.42, out, 0.58, 0.0, out)
    return out, {"skin_pct": f"{pct}%"}


def op_color_edges(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute vector color edge magnitude sqrt(|∇R|² + |∇G|² + |∇B|²) (Chapter 6)."""
    bgr = to_bgr(img).astype(np.float64)
    k = max(1, int(params.get("ksize", 3)) | 1)
    mode = str(params.get("mode", "Vector Gradient Magnitude"))
    normalize = bool(params.get("normalize", True))

    gx = np.zeros_like(bgr)
    gy = np.zeros_like(bgr)
    mags = []
    for c in range(3):
        cx = cv2.Sobel(bgr[:, :, c], cv2.CV_64F, 1, 0, ksize=k)
        cy = cv2.Sobel(bgr[:, :, c], cv2.CV_64F, 0, 1, ksize=k)
        gx[:, :, c] = cx
        gy[:, :, c] = cy
        mags.append(np.hypot(cx, cy))

    if mode == "Per-Channel RGB Edges":
        stacked = np.stack(mags, axis=2)
        if normalize and stacked.max() > 1e-6:
            stacked = cv2.normalize(stacked, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(np.round(stacked), 0, 255).astype(np.uint8), {"mode": "RGB Edges"}

    vec_mag = np.sqrt(mags[0] ** 2 + mags[1] ** 2 + mags[2] ** 2) / np.sqrt(3.0)
    if normalize and vec_mag.max() > 1e-6:
        vec_mag = cv2.normalize(vec_mag, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(np.round(vec_mag), 0, 255).astype(np.uint8), {"mode": "Vector Magnitude", "ksize": k}


def get_specs() -> list[OperationSpec]:
    """Return all Color Space operation specifications."""
    return [
        OperationSpec(
            op_id="color.channel",
            name="Color Space Channel Split",
            category=OperationCategory.COLOR,
            description="Inspect individual channels across RGB, HSV, CIE LAB, YCrCb, or CMY.",
            func=op_color_channel,
            keywords=("color space", "hsv", "lab", "ycrcb", "cmy", "channel"),
            params=(
                ParamSpec("space", "Color Model", ParamType.CHOICE, "HSV", choices=("RGB", "HSV", "CIE LAB", "YCrCb", "CMY")),
                ParamSpec("channel", "Channel Index (0/1/2)", ParamType.INT, 0, 0, 2, 1),
                ParamSpec("false_color", "Apply Turbo Heatmap", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="color.hsv_adjust",
            name="HSV Hue / Saturation",
            category=OperationCategory.COLOR,
            description="Shift hue and scale saturation or brightness in perceptual HSV space.",
            func=op_hsv_adjust,
            keywords=("hue", "saturation", "vibrance", "hsv"),
            params=(
                ParamSpec("hue_shift", "Hue Shift", ParamType.INT, 0, -180, 180, 2, unit="°"),
                ParamSpec("sat_scale", "Saturation Scale", ParamType.FLOAT, 1.0, 0.0, 2.5, 0.05, 2, unit="×"),
                ParamSpec("val_scale", "Value Scale", ParamType.FLOAT, 1.0, 0.1, 2.5, 0.05, 2, unit="×"),
            ),
        ),
        OperationSpec(
            op_id="color.balance",
            name="White Balance & RGB Gain",
            category=OperationCategory.COLOR,
            description="Apply Gray-World automatic white balance or manual Red/Green/Blue channel gains.",
            func=op_color_balance,
            keywords=("white balance", "rgb", "gain", "tint", "temperature"),
            params=(
                ParamSpec("auto_white_balance", "Auto White Balance (Gray-World)", ParamType.BOOL, False),
                ParamSpec("r_gain", "Red Gain", ParamType.FLOAT, 1.0, 0.2, 2.5, 0.05, 2),
                ParamSpec("g_gain", "Green Gain", ParamType.FLOAT, 1.0, 0.2, 2.5, 0.05, 2),
                ParamSpec("b_gain", "Blue Gain", ParamType.FLOAT, 1.0, 0.2, 2.5, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="color.temperature_tint",
            name="Temperature, Tint & Vibrance",
            category=OperationCategory.COLOR,
            description="Photographic warm/cool temperature, green/magenta tint, and selective vibrance.",
            func=op_temperature_tint,
            keywords=("temperature", "kelvin", "tint", "vibrance", "warm", "cool", "white balance"),
            params=(
                ParamSpec("temp", "Temperature (Cool <-> Warm)", ParamType.FLOAT, 0.0, -100.0, 100.0, 2.0, 0),
                ParamSpec("tint", "Tint (Magenta <-> Green)", ParamType.FLOAT, 0.0, -100.0, 100.0, 2.0, 0),
                ParamSpec("vibrance", "Vibrance", ParamType.FLOAT, 0.0, -100.0, 100.0, 2.0, 0),
            ),
        ),
        OperationSpec(
            op_id="color.pseudocolor",
            name="Pseudo-Color Colormap",
            category=OperationCategory.COLOR,
            description="Map grayscale intensity to perceptual scientific false-color palettes.",
            func=op_pseudocolor,
            keywords=("pseudocolor", "colormap", "heatmap", "turbo", "viridis"),
            params=(
                ParamSpec("colormap", "Palette", ParamType.CHOICE, "Turbo", choices=tuple(_COLORMAPS.keys())),
            ),
        ),
        OperationSpec(
            op_id="color.hue_slice",
            name="Color Slicing (HSV Hue Mask)",
            category=OperationCategory.COLOR,
            description="Isolate a target HSV hue range via cv2.inRange color slicing (Chapter 6).",
            func=op_hue_slice,
            keywords=("color slicing", "hue threshold", "inrange", "hsv", "selective"),
            params=(
                ParamSpec("target_hue", "Target Hue", ParamType.FLOAT, 15.0, 0.0, 360.0, 5.0, 0, unit="°"),
                ParamSpec("hue_width", "Hue Tolerance (+/-)", ParamType.FLOAT, 25.0, 4.0, 90.0, 2.0, 0, unit="°"),
                ParamSpec("min_sat", "Min Saturation", ParamType.INT, 40, 0, 255, 5),
                ParamSpec("outside_mode", "Outside Pixels", ParamType.CHOICE, "Grayscale", choices=("Grayscale", "Neutral Gray (128)", "Black", "Binary Mask")),
            ),
        ),
        OperationSpec(
            op_id="color.skin_detect",
            name="Skin Region Detector (YCbCr)",
            category=OperationCategory.COLOR,
            description="Segment skin-tone pixels using YCrCb chrominance bounds (Chapter 6).",
            func=op_skin_detect,
            keywords=("skin", "ycbcr", "ycrcb", "segmentation", "chrominance"),
            params=(
                ParamSpec("tolerance", "Bounds Tolerance", ParamType.INT, 4, 0, 25, 1),
                ParamSpec("display", "Display Mode", ParamType.CHOICE, "Highlight Overlay", choices=("Highlight Overlay", "Skin Cutout", "Binary Mask")),
            ),
        ),
        OperationSpec(
            op_id="color.color_edges",
            name="Color Edge Vector Gradient",
            category=OperationCategory.COLOR,
            description="Compute multi-channel color gradient sqrt(|∇R|²+|∇G|²+|∇B|²) (Chapter 6).",
            func=op_color_edges,
            keywords=("color edges", "vector gradient", "rgb edges", "sobel", "chapter 6"),
            params=(
                ParamSpec("mode", "Edge Mode", ParamType.CHOICE, "Vector Gradient Magnitude", choices=("Vector Gradient Magnitude", "Per-Channel RGB Edges")),
                ParamSpec("ksize", "Sobel Size", ParamType.INT, 3, 1, 7, 2, odd_only=True),
                ParamSpec("normalize", "Normalize [0-255]", ParamType.BOOL, True),
            ),
        ),
    ]
