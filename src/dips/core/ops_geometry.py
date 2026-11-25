"""Crop, Perspective/Keystone, Lens Distortion, Padding, and Polar Coordinate Warps."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import BORDER_MODES, border_flag, ensure_bgra, ensure_uint8
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def op_crop(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Percentage-based region crop with aspect ratio lock and optional center lock."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    w_pct = float(np.clip(params.get("w_pct", 80.0), 5.0, 100.0))
    h_pct = float(np.clip(params.get("h_pct", 80.0), 5.0, 100.0))
    aspect = str(params.get("aspect", "Free"))

    cw = max(1, int(round(w * w_pct / 100.0)))
    ch = max(1, int(round(h * h_pct / 100.0)))

    ratios = {
        "1:1 (Square)": 1.0,
        "4:3": 4.0 / 3.0,
        "16:9": 16.0 / 9.0,
        "3:2": 3.0 / 2.0,
        "4:5 (Portrait)": 4.0 / 5.0,
        "9:16 (Vertical)": 9.0 / 16.0,
        "21:9 (Cinema)": 21.0 / 9.0,
    }
    if aspect in ratios:
        ch = max(1, int(round(cw / ratios[aspect])))
        if ch > h:
            ch = h
            cw = max(1, min(w, int(round(ch * ratios[aspect]))))

    if bool(params.get("center_crop", False)):
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
    else:
        x_pct = float(np.clip(params.get("x_pct", 10.0), 0.0, 95.0))
        y_pct = float(np.clip(params.get("y_pct", 10.0), 0.0, 95.0))
        x0 = int(round(w * x_pct / 100.0))
        y0 = int(round(h * y_pct / 100.0))

    x1 = min(w, x0 + cw)
    y1 = min(h, y0 + ch)
    x0 = min(x0, max(0, x1 - 1))
    y0 = min(y0, max(0, y1 - 1))
    cropped = img[y0:y1, x0:x1].copy()
    return cropped, {"crop_rect": f"({x0},{y0}) {cropped.shape[1]}×{cropped.shape[0]}"}


def op_perspective(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """3D projective / keystone tilt warp along horizontal and vertical axes."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    tx = float(params.get("tilt_x", 0.15))
    ty = float(params.get("tilt_y", 0.0))
    border_name = str(params.get("border", "Transparent"))
    border = border_flag(border_name, cv2.BORDER_CONSTANT)

    src_img = ensure_bgra(img) if border_name == "Transparent" else img
    bval: Any = (0, 0, 0, 0) if border_name == "Transparent" else ((0, 0, 0, 255) if (src_img.ndim == 3 and src_img.shape[2] == 4) else ((0, 0, 0) if src_img.ndim == 3 else 0))

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = tx * w * 0.25
    dy = ty * h * 0.25
    dst = np.float32(
        [
            [max(0.0, dx), max(0.0, dy)],
            [min(float(w), w - dx), max(0.0, -dy)],
            [min(float(w), w + dx), min(float(h), h + dy)],
            [max(0.0, -dx), min(float(h), h - dy)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=bval)
    return out, {"tilt_x": tx, "tilt_y": ty}


def op_lens_distortion(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply or correct radial barrel / pincushion lens distortion (k1, k2)."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    k1 = float(params.get("k1", -0.25))
    k2 = float(params.get("k2", 0.0))
    border_name = str(params.get("border", "Transparent"))
    border = border_flag(border_name, cv2.BORDER_CONSTANT)

    src_img = ensure_bgra(img) if border_name == "Transparent" else img
    bval: Any = (0, 0, 0, 0) if border_name == "Transparent" else ((0, 0, 0, 255) if (src_img.ndim == 3 and src_img.shape[2] == 4) else ((0, 0, 0) if src_img.ndim == 3 else 0))

    fx = fy = float(max(w, h))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    out = cv2.remap(src_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=border, borderValue=bval)
    kind = "Barrel" if k1 > 0 else ("Pincushion" if k1 < 0 else "Neutral")
    return out, {"k1": k1, "k2": k2, "profile": kind}


def op_pad_border(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Add uniform or custom border padding using Transparent, Constant, Replicate, Reflect, or Wrap modes."""
    img = ensure_uint8(img)
    pad = max(0, int(params.get("pad_px", 24)))
    mode_name = str(params.get("border", "Transparent"))
    mode = border_flag(mode_name, cv2.BORDER_CONSTANT)
    fill = int(params.get("fill_gray", 24))
    if mode_name == "Transparent":
        src_img = ensure_bgra(img)
        val: Any = (0, 0, 0, 0)
    else:
        src_img = img
        val = (fill, fill, fill, 255) if (src_img.ndim == 3 and src_img.shape[2] == 4) else ((fill, fill, fill) if src_img.ndim == 3 else fill)
    out = cv2.copyMakeBorder(src_img, pad, pad, pad, pad, mode, value=val)
    return out, {"padded_size": f"{out.shape[1]}×{out.shape[0]}", "mode": mode_name}


def op_polar_warp(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Remap between Cartesian and Linear-Polar or Log-Polar coordinates."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    mode = str(params.get("mode", "Log-Polar"))
    inverse = bool(params.get("inverse", False))
    flags = cv2.WARP_POLAR_LOG if mode == "Log-Polar" else cv2.WARP_POLAR_LINEAR
    flags |= cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    if inverse:
        flags |= cv2.WARP_INVERSE_MAP
    max_r = float(min(w, h) / 2.0)
    out = cv2.warpPolar(img, (w, h), (w / 2.0, h / 2.0), max_r, flags)
    return out, {"mode": mode, "inverse": inverse}


def op_tilt_shift(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate a tilt-shift miniature optical lens with a tilted focal band and color pop."""
    img = ensure_uint8(img)
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    alpha_ch = img[:, :, 3] if has_alpha else None
    bgr = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = bgr.shape[:2]
    focus_y = float(np.clip(params.get("focus_y", 50.0), 5.0, 95.0)) / 100.0
    band_w = float(np.clip(params.get("band_width", 22.0), 5.0, 75.0)) / 100.0
    angle_deg = float(params.get("angle", 0.0))
    blur_k = max(3, int(params.get("blur_strength", 19)) | 1)
    sat_pop = float(params.get("saturation_pop", 1.20))

    if abs(sat_pop - 1.0) > 1e-3:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_pop, 0, 255)
        bgr_pop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    else:
        bgr_pop = bgr

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / max(1.0, float(h))
    ny = (yy - h * focus_y) / max(1.0, float(h))
    rad = np.deg2rad(angle_deg)
    dist = np.abs(ny * float(np.cos(rad)) - nx * float(np.sin(rad)))
    weight = np.clip((dist - band_w * 0.5) / max(0.08, 0.55 - band_w * 0.35), 0.0, 1.0)

    mid_k = max(3, (blur_k // 2) | 1)
    blur_mid = cv2.GaussianBlur(bgr_pop, (mid_k, mid_k), 0).astype(np.float32)
    blur_far = cv2.GaussianBlur(bgr_pop, (blur_k, blur_k), 0).astype(np.float32)
    w_mid = np.clip(weight / 0.5, 0.0, 1.0)[:, :, None]
    w_far = np.clip((weight - 0.35) / 0.65, 0.0, 1.0)[:, :, None]
    blurred = blur_mid * (1.0 - w_far) + blur_far * w_far
    out_bgr = np.clip(np.round(bgr_pop.astype(np.float32) * (1.0 - w_mid) + blurred * w_mid), 0, 255).astype(np.uint8)

    if has_alpha and alpha_ch is not None:
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = alpha_ch
        return out, {"focus_y": f"{focus_y * 100:.0f}%", "tilt": f"{angle_deg:+.0f}°"}
    return out_bgr, {"focus_y": f"{focus_y * 100:.0f}%", "tilt": f"{angle_deg:+.0f}°"}


def get_specs() -> list[OperationSpec]:
    """Return all Crop, Perspective & Lens operation specifications."""
    border_param = ParamSpec(
        "border",
        "Border Fill",
        ParamType.CHOICE,
        "Transparent",
        choices=tuple(BORDER_MODES.keys()),
    )
    return [
        OperationSpec(
            op_id="geom.crop",
            name="Crop & Aspect Frame",
            category=OperationCategory.GEOMETRY,
            description="Drag the interactive frame on the Original image or tune coordinates and aspect ratio.",
            func=op_crop,
            keywords=("crop", "trim", "roi", "aspect", "16:9", "square"),
            params=(
                ParamSpec(
                    "aspect",
                    "Aspect Ratio",
                    ParamType.CHOICE,
                    "Free",
                    choices=("Free", "1:1 (Square)", "4:3", "16:9", "3:2", "4:5 (Portrait)", "9:16 (Vertical)", "21:9 (Cinema)"),
                ),
                ParamSpec("center_crop", "Center Frame Automatically", ParamType.BOOL, False),
                ParamSpec("x_pct", "Left X", ParamType.FLOAT, 10.0, 0.0, 95.0, 1.0, 0, unit="%"),
                ParamSpec("y_pct", "Top Y", ParamType.FLOAT, 10.0, 0.0, 95.0, 1.0, 0, unit="%"),
                ParamSpec("w_pct", "Width", ParamType.FLOAT, 80.0, 5.0, 100.0, 1.0, 0, unit="%"),
                ParamSpec("h_pct", "Height", ParamType.FLOAT, 80.0, 5.0, 100.0, 1.0, 0, unit="%"),
            ),
        ),
        OperationSpec(
            op_id="geom.perspective",
            name="Perspective / Keystone",
            category=OperationCategory.GEOMETRY,
            description="Apply 3D keystone tilt warp to correct or simulate camera perspective.",
            func=op_perspective,
            keywords=("perspective", "keystone", "homography", "tilt"),
            params=(
                ParamSpec("tilt_x", "Horizontal Tilt", ParamType.FLOAT, 0.15, -1.0, 1.0, 0.02, 2),
                ParamSpec("tilt_y", "Vertical Tilt", ParamType.FLOAT, 0.00, -1.0, 1.0, 0.02, 2),
                border_param,
            ),
        ),
        OperationSpec(
            op_id="geom.lens",
            name="Lens Distortion / Undistort",
            category=OperationCategory.GEOMETRY,
            description="Correct or simulate radial barrel and pincushion optical distortion.",
            func=op_lens_distortion,
            keywords=("lens", "barrel", "pincushion", "fisheye", "undistort"),
            params=(
                ParamSpec("k1", "Radial Coefficient (k1)", ParamType.FLOAT, -0.25, -1.5, 1.5, 0.05, 2),
                ParamSpec("k2", "Secondary Coeff (k2)", ParamType.FLOAT, 0.00, -1.0, 1.0, 0.05, 2),
                border_param,
            ),
        ),
        OperationSpec(
            op_id="geom.pad",
            name="Border / Frame Padding",
            category=OperationCategory.GEOMETRY,
            description="Expand image canvas with Constant, Replicate, Reflect, or Wrap boundary extension.",
            func=op_pad_border,
            keywords=("pad", "border", "frame", "margin", "reflect"),
            params=(
                ParamSpec("pad_px", "Border Width", ParamType.INT, 24, 1, 200, 2, unit="px"),
                ParamSpec("border", "Extension Mode", ParamType.CHOICE, "Transparent", choices=tuple(BORDER_MODES.keys())),
                ParamSpec("fill_gray", "Constant Gray Level", ParamType.INT, 24, 0, 255, 1),
            ),
        ),
        OperationSpec(
            op_id="geom.polar",
            name="Polar / Log-Polar Warp",
            category=OperationCategory.GEOMETRY,
            description="Transform image between Cartesian (x,y) and Polar / Log-Polar (r,θ) space.",
            func=op_polar_warp,
            keywords=("polar", "log-polar", "radial", "warp"),
            params=(
                ParamSpec("mode", "Mapping", ParamType.CHOICE, "Log-Polar", choices=("Log-Polar", "Linear Polar")),
                ParamSpec("inverse", "Inverse Transform (Polar → Cartesian)", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="geom.tilt_shift",
            name="Tilt-Shift Miniature Lens",
            category=OperationCategory.GEOMETRY,
            description="Create miniature shallow depth-of-field photography with a tilted sharp plane.",
            func=op_tilt_shift,
            keywords=("tilt shift", "miniature", "lens", "blur", "focal band", "toy"),
            params=(
                ParamSpec("focus_y", "Focal Band Center Y", ParamType.FLOAT, 50.0, 5.0, 95.0, 2.0, 0, unit="%"),
                ParamSpec("band_width", "Sharp Band Width", ParamType.FLOAT, 22.0, 5.0, 75.0, 2.0, 0, unit="%"),
                ParamSpec("angle", "Plane Tilt Angle", ParamType.FLOAT, 0.0, -45.0, 45.0, 2.0, 0, unit="°"),
                ParamSpec("blur_strength", "Blur Strength", ParamType.INT, 19, 3, 55, 2, odd_only=True),
                ParamSpec("saturation_pop", "Miniature Color Pop", ParamType.FLOAT, 1.20, 0.80, 1.80, 0.05, 2),
            ),
        ),
    ]
