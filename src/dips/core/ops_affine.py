"""Affine geometric transformations with configurable interpolation and border modes."""

from __future__ import annotations

from typing import Any
import math
import cv2
import numpy as np

from .io_utils import BORDER_MODES, border_flag, ensure_bgra, ensure_uint8
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType

_INTERP_MODES = {
    "Bilinear": cv2.INTER_LINEAR,
    "Nearest Neighbor": cv2.INTER_NEAREST,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos4": cv2.INTER_LANCZOS4,
}


def _interp_flag(name: str) -> int:
    return _INTERP_MODES.get(name, cv2.INTER_LINEAR)


def _format_matrix(M: np.ndarray) -> list[list[float]]:
    return [[round(float(M[r, c]), 4) for c in range(3)] for r in range(2)]


def _warp_with_bounds(
    img: np.ndarray,
    M: np.ndarray,
    out_w: int,
    out_h: int,
    interp_name: str,
    border_name: str,
) -> np.ndarray:
    if border_name == "Transparent":
        src = ensure_bgra(img)
        border_val: Any = (0, 0, 0, 0)
    else:
        src = img
        border_val = (0, 0, 0, 255) if (src.ndim == 3 and src.shape[2] == 4) else ((0, 0, 0) if src.ndim == 3 else 0)
    return cv2.warpAffine(
        src,
        M.astype(np.float64),
        (max(1, int(out_w)), max(1, int(out_h))),
        flags=_interp_flag(interp_name),
        borderMode=border_flag(border_name),
        borderValue=border_val,
    )


def op_rotate(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Arbitrary-angle rotation with optional canvas expansion."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    angle = float(params.get("angle", 15.0))
    expand = bool(params.get("expand", False))
    interp = str(params.get("interpolation", "Bilinear"))
    border = str(params.get("border", "Transparent"))

    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    out_w, out_h = w, h
    if expand:
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        out_w = int(math.ceil(h * sin_a + w * cos_a))
        out_h = int(math.ceil(h * cos_a + w * sin_a))
        M[0, 2] += (out_w / 2.0) - cx
        M[1, 2] += (out_h / 2.0) - cy

    out = _warp_with_bounds(img, M, out_w, out_h, interp, border)
    return out, {"affine_matrix": _format_matrix(M), "output_size": f"{out_w}×{out_h}"}


def op_scale(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale image uniformly or independently along X and Y axes."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    uniform = bool(params.get("uniform", True))
    sx = max(0.05, float(params.get("scale_x", 1.0)))
    sy = sx if uniform else max(0.05, float(params.get("scale_y", 1.0)))
    keep_canvas = bool(params.get("keep_canvas", False))
    interp = str(params.get("interpolation", "Bilinear"))
    border = str(params.get("border", "Transparent"))

    if keep_canvas:
        cx, cy = w / 2.0, h / 2.0
        M = np.array([[sx, 0.0, cx * (1.0 - sx)], [0.0, sy, cy * (1.0 - sy)]], dtype=np.float64)
        out = _warp_with_bounds(img, M, w, h, interp, border)
        return out, {"affine_matrix": _format_matrix(M), "output_size": f"{w}×{h}"}

    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))
    M = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64)
    out = cv2.resize(img, (new_w, new_h), interpolation=_interp_flag(interp))
    return out, {"affine_matrix": _format_matrix(M), "output_size": f"{new_w}×{new_h}"}


def op_translate(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate image by (dx, dy) pixels."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    dx = float(params.get("dx", 0))
    dy = float(params.get("dy", 0))
    border = str(params.get("border", "Transparent"))
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    out = _warp_with_bounds(img, M, w, h, "Nearest Neighbor", border)
    return out, {"affine_matrix": _format_matrix(M)}


def op_shear(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply horizontal (X), vertical (Y), or combined shear centered on the image."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    mode = str(params.get("mode", "Both"))
    shx = float(params.get("shear_x", 0.20))
    shy = float(params.get("shear_y", 0.0))
    if mode == "Shear X":
        shy = 0.0
    elif mode == "Shear Y":
        shx = 0.0

    expand = bool(params.get("expand", True))
    interp = str(params.get("interpolation", "Bilinear"))
    border = str(params.get("border", "Transparent"))

    cx, cy = w / 2.0, h / 2.0
    # Center-anchored shear: T(c) * Sh * T(-c)
    T_neg = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    Sh = np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T_pos = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    M3 = T_pos @ Sh @ T_neg

    out_w, out_h = w, h
    if expand:
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float64).T
        transformed = M3[:2, :] @ corners
        min_x, max_x = transformed[0].min(), transformed[0].max()
        min_y, max_y = transformed[1].min(), transformed[1].max()
        out_w = max(1, int(math.ceil(max_x - min_x)))
        out_h = max(1, int(math.ceil(max_y - min_y)))
        M3[0, 2] -= min_x
        M3[1, 2] -= min_y

    M = M3[:2, :]
    out = _warp_with_bounds(img, M, out_w, out_h, interp, border)
    return out, {"affine_matrix": _format_matrix(M), "output_size": f"{out_w}×{out_h}"}


def op_combined_affine(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Combined Rotate + Scale + Shear + Translate matrix in a single resampling pass."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    angle = math.radians(float(params.get("angle", 0.0)))
    sx = max(0.05, float(params.get("scale_x", 1.0)))
    sy = max(0.05, float(params.get("scale_y", 1.0)))
    shx = float(params.get("shear_x", 0.0))
    shy = float(params.get("shear_y", 0.0))
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    interp = str(params.get("interpolation", "Bilinear"))
    border = str(params.get("border", "Transparent"))

    cx, cy = w / 2.0, h / 2.0
    T_neg = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    Sh = np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c, s = math.cos(-angle), math.sin(-angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T_pos = np.array([[1.0, 0.0, cx + dx], [0.0, 1.0, cy + dy], [0.0, 0.0, 1.0]])

    M3 = T_pos @ R @ Sh @ S @ T_neg
    M = M3[:2, :]
    out = _warp_with_bounds(img, M, w, h, interp, border)
    return out, {"affine_matrix": _format_matrix(M)}


def get_specs() -> list[OperationSpec]:
    """Return all Affine Transformation operation specifications."""
    interp_param = ParamSpec(
        "interpolation",
        "Interpolation",
        ParamType.CHOICE,
        "Bilinear",
        choices=tuple(_INTERP_MODES.keys()),
        group="Resampling",
    )
    border_param = ParamSpec(
        "border",
        "Border Fill",
        ParamType.CHOICE,
        "Transparent",
        choices=tuple(BORDER_MODES.keys()),
        group="Resampling",
    )
    return [
        OperationSpec(
            op_id="affine.rotate",
            name="Rotate",
            category=OperationCategory.AFFINE,
            description="Rotate by an arbitrary angle with selectable interpolation and border mode.",
            func=op_rotate,
            preview_widget="affine",
            keywords=("rotate", "angle", "interpolation", "warp"),
            params=(
                ParamSpec("angle", "Angle", ParamType.FLOAT, 15.0, -180.0, 180.0, 0.5, 1, unit="°"),
                ParamSpec("expand", "Expand Canvas to Fit", ParamType.BOOL, False),
                interp_param,
                border_param,
            ),
        ),
        OperationSpec(
            op_id="affine.scale",
            name="Scale / Resize",
            category=OperationCategory.AFFINE,
            description="Resize uniformly or independently along X and Y axes.",
            func=op_scale,
            preview_widget="affine",
            keywords=("scale", "resize", "zoom", "aspect"),
            params=(
                ParamSpec("uniform", "Uniform Scaling (Lock X=Y)", ParamType.BOOL, True),
                ParamSpec("scale_x", "Scale X", ParamType.FLOAT, 1.0, 0.1, 4.0, 0.05, 2, unit="×"),
                ParamSpec("scale_y", "Scale Y", ParamType.FLOAT, 1.0, 0.1, 4.0, 0.05, 2, unit="×"),
                ParamSpec("keep_canvas", "Keep Original Canvas Size", ParamType.BOOL, False),
                interp_param,
                border_param,
            ),
        ),
        OperationSpec(
            op_id="affine.translate",
            name="Translate",
            category=OperationCategory.AFFINE,
            description="Shift image horizontally (dx) and vertically (dy) in pixels.",
            func=op_translate,
            preview_widget="affine",
            keywords=("translate", "shift", "offset", "move"),
            params=(
                ParamSpec("dx", "Shift X (dx)", ParamType.INT, 0, -500, 500, 1, unit="px"),
                ParamSpec("dy", "Shift Y (dy)", ParamType.INT, 0, -500, 500, 1, unit="px"),
                border_param,
            ),
        ),
        OperationSpec(
            op_id="affine.shear",
            name="Shear",
            category=OperationCategory.AFFINE,
            description="Apply X, Y, or combined horizontal/vertical shear transformation.",
            func=op_shear,
            preview_widget="affine",
            keywords=("shear", "skew", "affine"),
            params=(
                ParamSpec("mode", "Shear Mode", ParamType.CHOICE, "Both", choices=("Shear X", "Shear Y", "Both")),
                ParamSpec("shear_x", "Shear X", ParamType.FLOAT, 0.20, -1.5, 1.5, 0.02, 2),
                ParamSpec("shear_y", "Shear Y", ParamType.FLOAT, 0.00, -1.5, 1.5, 0.02, 2),
                ParamSpec("expand", "Expand Canvas to Fit", ParamType.BOOL, True),
                interp_param,
                border_param,
            ),
        ),
        OperationSpec(
            op_id="affine.combined",
            name="Combined Affine Matrix",
            category=OperationCategory.AFFINE,
            description="Compose Rotation, Scale, Shear, and Translation into a single 2×3 warp.",
            func=op_combined_affine,
            preview_widget="affine",
            keywords=("affine", "matrix", "compose", "transform"),
            params=(
                ParamSpec("angle", "Angle", ParamType.FLOAT, 0.0, -180.0, 180.0, 0.5, 1, unit="°"),
                ParamSpec("scale_x", "Scale X", ParamType.FLOAT, 1.0, 0.1, 3.0, 0.05, 2, unit="×"),
                ParamSpec("scale_y", "Scale Y", ParamType.FLOAT, 1.0, 0.1, 3.0, 0.05, 2, unit="×"),
                ParamSpec("shear_x", "Shear X", ParamType.FLOAT, 0.0, -1.0, 1.0, 0.02, 2),
                ParamSpec("shear_y", "Shear Y", ParamType.FLOAT, 0.0, -1.0, 1.0, 0.02, 2),
                ParamSpec("dx", "Shift X (dx)", ParamType.INT, 0, -400, 400, 1, unit="px"),
                ParamSpec("dy", "Shift Y (dy)", ParamType.INT, 0, -400, 400, 1, unit="px"),
                interp_param,
                border_param,
            ),
        ),
    ]
