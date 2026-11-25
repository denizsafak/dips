"""Spatial domain filtering: smoothing, sharpening, gradients, edge detectors, custom convolution."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def _odd(k: Any, minimum: int = 3, maximum: int = 51) -> int:
    val = int(round(float(k)))
    val = max(minimum, min(maximum, val))
    return val if val % 2 == 1 else val + 1


def _finalize_deriv(arr: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        norm = cv2.normalize(np.abs(arr), None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(norm, 0, 255).astype(np.uint8)
    return np.clip(np.abs(arr), 0, 255).astype(np.uint8)


def op_mean_blur(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Box / Mean averaging filter."""
    img = ensure_uint8(img)
    kx = _odd(params.get("ksize_x", 5), 1, 51)
    ky = kx if bool(params.get("square", True)) else _odd(params.get("ksize_y", 5), 1, 51)
    out = cv2.blur(img, (kx, ky))
    return out, {"kernel": f"{kx}×{ky}"}


def op_gaussian_blur(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Gaussian smoothing filter with kernel size and sigma control."""
    img = ensure_uint8(img)
    k = _odd(params.get("ksize", 5), 3, 51)
    sigma = float(params.get("sigma", 0.0))
    out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    return out, {"kernel": f"{k}×{k}", "sigma": sigma if sigma > 0 else "auto"}


def op_median_blur(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Non-linear median filter (ideal for salt-and-pepper impulse noise)."""
    img = ensure_uint8(img)
    k = _odd(params.get("ksize", 5), 3, 31)
    out = cv2.medianBlur(img, k)
    return out, {"kernel": f"{k}×{k}"}


def op_bilateral(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Edge-preserving bilateral smoothing filter."""
    img = ensure_uint8(img)
    d = max(3, int(params.get("d", 9)))
    sc = float(params.get("sigma_color", 75.0))
    ss = float(params.get("sigma_space", 75.0))
    out = cv2.bilateralFilter(img, d, sc, ss)
    return out, {"diameter": d, "sigma_color": sc, "sigma_space": ss}


def op_laplacian(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Second-order Laplacian edge response or Laplacian image sharpening."""
    img = ensure_uint8(img)
    k = _odd(params.get("ksize", 3), 1, 15)
    mode = str(params.get("mode", "Edge Magnitude"))
    use_gray = bool(params.get("grayscale_edges", True))
    src = to_gray(img) if (use_gray and mode != "Sharpen (Original - c·∇²f)") else img

    lap = cv2.Laplacian(src, cv2.CV_64F, ksize=k)
    if mode == "Sharpen (Original - c·∇²f)":
        strength = float(params.get("sharpen_weight", 0.5))
        # OpenCV Laplacian uses positive center for ksize=1,3? Wait: cv2.Laplacian center is positive (+4/-1),
        # so sharpening is f - c * lap. Verify: f - (-4f + neighbors) = 5f - neighbors! Wait:
        # Actually cv2.Laplacian kernel for ksize=1 is [[0,1,0],[1,-4,1],[0,1,0]] (negative center)!
        # With negative center, f - c * lap adds center and subtracts neighbors!
        sharpened = img.astype(np.float64) - strength * lap
        return np.clip(np.round(sharpened), 0, 255).astype(np.uint8), {"mode": "Sharpen", "ksize": k, "weight": strength}

    if mode == "Zero-Centered (+128)":
        shifted = np.clip(np.round(lap + 128.0), 0, 255).astype(np.uint8)
        return shifted, {"mode": "Zero-Centered", "ksize": k}

    return _finalize_deriv(lap, bool(params.get("normalize", True))), {"mode": "Magnitude", "ksize": k}


def op_sobel(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """First-order Sobel gradient operator along X, Y, Magnitude, or Orientation."""
    img = ensure_uint8(img)
    k = _odd(params.get("ksize", 3), 1, 7)
    direction = str(params.get("direction", "Magnitude (√Gx²+Gy²)"))
    normalize = bool(params.get("normalize", True))
    src = to_gray(img) if bool(params.get("grayscale", True)) else img

    gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=k)

    if direction == "Sobel X (Vertical Edges)":
        return _finalize_deriv(gx, normalize), {"direction": "X", "ksize": k}
    if direction == "Sobel Y (Horizontal Edges)":
        return _finalize_deriv(gy, normalize), {"direction": "Y", "ksize": k}
    if direction == "Gradient Angle (Hue)":
        g_src = to_gray(img)
        gx_g = cv2.Sobel(g_src, cv2.CV_64F, 1, 0, ksize=k)
        gy_g = cv2.Sobel(g_src, cv2.CV_64F, 0, 1, ksize=k)
        mag = np.hypot(gx_g, gy_g)
        ang = (np.arctan2(gy_g, gx_g) + np.pi) * (179.0 / (2.0 * np.pi))
        hsv = np.zeros((*g_src.shape, 3), dtype=np.uint8)
        hsv[:, :, 0] = np.clip(ang, 0, 179).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.clip(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), {"direction": "Angle", "ksize": k}

    mag = np.hypot(gx, gy)
    return _finalize_deriv(mag, normalize), {"direction": "Magnitude", "ksize": k}


def op_gradient_classic(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Classic first-order derivative edge detectors: Prewitt, Scharr, and Roberts Cross."""
    img = ensure_uint8(img)
    operator = str(params.get("operator", "Prewitt"))
    direction = str(params.get("direction", "Magnitude"))
    normalize = bool(params.get("normalize", True))
    src = to_gray(img) if bool(params.get("grayscale", True)) else img

    if operator == "Scharr":
        gx = cv2.Scharr(src, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(src, cv2.CV_64F, 0, 1)
    elif operator == "Roberts Cross":
        kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        gx = cv2.filter2D(src, cv2.CV_64F, kx)
        gy = cv2.filter2D(src, cv2.CV_64F, ky)
    else:  # Prewitt
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        gx = cv2.filter2D(src, cv2.CV_64F, kx)
        gy = cv2.filter2D(src, cv2.CV_64F, ky)

    if direction == "X":
        res = _finalize_deriv(gx, normalize)
    elif direction == "Y":
        res = _finalize_deriv(gy, normalize)
    else:
        res = _finalize_deriv(np.hypot(gx, gy), normalize)
    return res, {"operator": operator, "direction": direction}


def op_canny(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-stage Canny edge detector with Gaussian pre-smoothing and hysteresis thresholds."""
    img = ensure_uint8(img)
    gray = to_gray(img)
    pre_k = int(params.get("blur_ksize", 3))
    if pre_k >= 3:
        pre_k = _odd(pre_k, 3, 15)
        gray = cv2.GaussianBlur(gray, (pre_k, pre_k), 0)

    if bool(params.get("auto_thresholds", False)):
        med = float(np.median(gray))
        sigma = 0.33
        t1 = int(max(0, (1.0 - sigma) * med))
        t2 = int(min(255, (1.0 + sigma) * med))
    else:
        t1 = int(params.get("threshold1", 60))
        t2 = int(params.get("threshold2", 160))
        if t1 > t2:
            t1, t2 = t2, t1

    l2 = bool(params.get("l2_gradient", True))
    edges = cv2.Canny(gray, t1, t2, apertureSize=3, L2gradient=l2)
    return edges, {"t1": t1, "t2": t2}


def op_unsharp_mask(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Unsharp masking / high-boost sharpening: g = f + k * (f - blur(f))."""
    img = ensure_uint8(img)
    ksize = _odd(params.get("ksize", 5), 3, 31)
    sigma = max(0.1, float(params.get("sigma", 1.5)))
    amount = float(params.get("amount", 1.2))
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    detail = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + amount * detail
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8), {"amount": amount, "sigma": sigma}


CUSTOM_KERNEL_PRESETS: dict[str, list[list[float]]] = {
    "Sharpen": [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
    "Identity": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    "Box Blur (3×3)": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "Edge Detect (8-conn)": [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
    "Emboss": [[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]],
    "Sobel X": [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
}


def op_custom_kernel(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply user-defined 3×3 or 5×5 spatial convolution kernel."""
    img = ensure_uint8(img)
    raw_kernel = params.get("kernel", CUSTOM_KERNEL_PRESETS["Sharpen"])
    arr = np.array(raw_kernel, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] not in (3, 5):
        arr = np.array(CUSTOM_KERNEL_PRESETS["Identity"], dtype=np.float64)

    if bool(params.get("normalize_sum", False)):
        s = float(arr.sum())
        if abs(s) > 1e-6:
            arr = arr / s

    bias = float(params.get("bias", 0.0))
    if bool(params.get("abs_output", False)):
        resp = cv2.filter2D(img, cv2.CV_64F, arr) + bias
        out = np.clip(np.abs(resp), 0, 255).astype(np.uint8)
    else:
        resp = cv2.filter2D(img, cv2.CV_64F, arr) + bias
        out = np.clip(np.round(resp), 0, 255).astype(np.uint8)
    return out, {"kernel_size": f"{arr.shape[0]}×{arr.shape[1]}", "sum": round(float(arr.sum()), 3)}


_POINT_LINE_KERNELS: dict[str, np.ndarray] = {
    "Isolated Point (Laplacian)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64),
    "Horizontal Line": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float64),
    "Vertical Line": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float64),
    "+45° Diagonal Line": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float64),
    "-45° Diagonal Line": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float64),
}


def op_point_line_detect(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect isolated points or oriented 1-pixel lines via Chapter 5 spatial masks."""
    gray = to_gray(img)
    mask_name = str(params.get("mask_type", "Isolated Point (Laplacian)"))
    thresh_pct = float(params.get("threshold_pct", 0.35))
    overlay = bool(params.get("overlay", True))

    kernel = _POINT_LINE_KERNELS.get(mask_name, _POINT_LINE_KERNELS["Isolated Point (Laplacian)"])
    resp = np.abs(cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel))
    mx = float(resp.max()) if resp.size else 0.0
    hits = (resp >= thresh_pct * mx) & (mx > 1e-6)
    count = int(np.count_nonzero(hits))

    if overlay:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else ensure_uint8(img)[:, :, :3].copy()
        vis[hits] = (76, 175, 255)
        return vis, {"mask": mask_name, "detected_px": count}
    return np.where(hits, 255, 0).astype(np.uint8), {"mask": mask_name, "detected_px": count}


def get_specs() -> list[OperationSpec]:
    """Return all Spatial Filtering operation specifications."""
    return [
        OperationSpec(
            op_id="spatial.mean",
            name="Mean / Box Filter",
            category=OperationCategory.SPATIAL,
            description="Uniform neighborhood averaging filter for smoothing and noise reduction.",
            func=op_mean_blur,
            keywords=("mean", "box", "average", "blur", "smooth"),
            params=(
                ParamSpec("square", "Square Kernel", ParamType.BOOL, True),
                ParamSpec("ksize_x", "Kernel Width", ParamType.INT, 5, 1, 51, 2, odd_only=True),
                ParamSpec("ksize_y", "Kernel Height", ParamType.INT, 5, 1, 51, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="spatial.gaussian",
            name="Gaussian Blur",
            category=OperationCategory.SPATIAL,
            description="Weighted bell-curve smoothing with configurable kernel size and σ.",
            func=op_gaussian_blur,
            keywords=("gaussian", "blur", "smooth", "sigma", "lowpass"),
            params=(
                ParamSpec("ksize", "Kernel Size", ParamType.INT, 5, 3, 51, 2, odd_only=True),
                ParamSpec("sigma", "Sigma (0 = Auto)", ParamType.FLOAT, 0.0, 0.0, 15.0, 0.2, 1),
            ),
        ),
        OperationSpec(
            op_id="spatial.median",
            name="Median Filter",
            category=OperationCategory.SPATIAL,
            description="Order-statistic filter that removes salt-and-pepper noise while preserving edges.",
            func=op_median_blur,
            keywords=("median", "impulse", "salt pepper", "denoise"),
            params=(
                ParamSpec("ksize", "Kernel Size", ParamType.INT, 5, 3, 31, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="spatial.bilateral",
            name="Bilateral Filter",
            category=OperationCategory.SPATIAL,
            description="Edge-preserving non-linear smoothing combining spatial and intensity weights.",
            func=op_bilateral,
            keywords=("bilateral", "edge preserving", "smooth", "skin"),
            params=(
                ParamSpec("d", "Neighborhood Diameter", ParamType.INT, 9, 3, 25, 2),
                ParamSpec("sigma_color", "Sigma Color", ParamType.FLOAT, 75.0, 5.0, 200.0, 5.0, 0),
                ParamSpec("sigma_space", "Sigma Space", ParamType.FLOAT, 75.0, 5.0, 200.0, 5.0, 0),
            ),
        ),
        OperationSpec(
            op_id="spatial.laplacian",
            name="Laplacian",
            category=OperationCategory.SPATIAL,
            description="Isotropic 2nd-order derivative operator for edge detection or sharpening.",
            func=op_laplacian,
            keywords=("laplacian", "second derivative", "sharpen", "edges"),
            params=(
                ParamSpec(
                    "mode",
                    "Output Mode",
                    ParamType.CHOICE,
                    "Edge Magnitude",
                    choices=("Edge Magnitude", "Zero-Centered (+128)", "Sharpen (Original - c·∇²f)"),
                ),
                ParamSpec("ksize", "Aperture Size", ParamType.INT, 3, 1, 15, 2, odd_only=True),
                ParamSpec("sharpen_weight", "Sharpen Weight (c)", ParamType.FLOAT, 0.5, 0.1, 3.0, 0.1, 2),
                ParamSpec("normalize", "Normalize Magnitude [0–255]", ParamType.BOOL, True),
                ParamSpec("grayscale_edges", "Compute on Grayscale", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="spatial.sobel",
            name="Sobel Gradient",
            category=OperationCategory.SPATIAL,
            description="First-order directional derivative (Sobel X, Y, Magnitude, or Angle).",
            func=op_sobel,
            keywords=("sobel", "gradient", "dx", "dy", "edges"),
            params=(
                ParamSpec(
                    "direction",
                    "Component",
                    ParamType.CHOICE,
                    "Magnitude (√Gx²+Gy²)",
                    choices=(
                        "Magnitude (√Gx²+Gy²)",
                        "Sobel X (Vertical Edges)",
                        "Sobel Y (Horizontal Edges)",
                        "Gradient Angle (Hue)",
                    ),
                ),
                ParamSpec("ksize", "Kernel Size", ParamType.INT, 3, 1, 7, 2, odd_only=True),
                ParamSpec("normalize", "Normalize [0–255]", ParamType.BOOL, True),
                ParamSpec("grayscale", "Convert to Grayscale First", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="spatial.gradient_classic",
            name="Prewitt / Scharr / Roberts",
            category=OperationCategory.SPATIAL,
            description="Classic first-order gradient edge operators from Chapter 5 & 7.",
            func=op_gradient_classic,
            keywords=("prewitt", "scharr", "roberts", "edge", "gradient"),
            params=(
                ParamSpec("operator", "Operator", ParamType.CHOICE, "Prewitt", choices=("Prewitt", "Scharr", "Roberts Cross")),
                ParamSpec("direction", "Component", ParamType.CHOICE, "Magnitude", choices=("Magnitude", "X", "Y")),
                ParamSpec("normalize", "Normalize [0–255]", ParamType.BOOL, True),
                ParamSpec("grayscale", "Convert to Grayscale First", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="spatial.canny",
            name="Canny Edge Detector",
            category=OperationCategory.SPATIAL,
            description="Optimal multi-stage edge detector with non-max suppression and hysteresis.",
            func=op_canny,
            keywords=("canny", "edges", "hysteresis", "threshold"),
            params=(
                ParamSpec("auto_thresholds", "Auto Median Thresholds", ParamType.BOOL, False),
                ParamSpec("threshold1", "Low Threshold", ParamType.INT, 60, 0, 255, 1),
                ParamSpec("threshold2", "High Threshold", ParamType.INT, 160, 0, 255, 1),
                ParamSpec("blur_ksize", "Pre-Blur Kernel (0 = Off)", ParamType.INT, 3, 0, 15, 1),
                ParamSpec("l2_gradient", "Accurate L2 Gradient", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="spatial.unsharp",
            name="Unsharp Mask (Sharpen)",
            category=OperationCategory.SPATIAL,
            description="High-boost sharpening by amplifying high-frequency detail above Gaussian blur.",
            func=op_unsharp_mask,
            keywords=("sharpen", "unsharp", "highboost", "crisp"),
            params=(
                ParamSpec("amount", "Amount (k)", ParamType.FLOAT, 1.2, 0.1, 4.0, 0.1, 2),
                ParamSpec("sigma", "Radius (σ)", ParamType.FLOAT, 1.5, 0.2, 10.0, 0.2, 1),
                ParamSpec("ksize", "Kernel Size", ParamType.INT, 5, 3, 31, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="spatial.custom_kernel",
            name="Custom Convolution Kernel",
            category=OperationCategory.SPATIAL,
            description="Edit a custom 3×3 or 5×5 convolution matrix directly or load presets.",
            func=op_custom_kernel,
            preview_widget="kernel",
            keywords=("custom", "kernel", "convolution", "filter2d", "matrix"),
            params=(
                ParamSpec("kernel", "Convolution Matrix", ParamType.KERNEL, CUSTOM_KERNEL_PRESETS["Sharpen"]),
                ParamSpec("normalize_sum", "Normalize by Kernel Sum", ParamType.BOOL, False),
                ParamSpec("abs_output", "Take Absolute Value", ParamType.BOOL, False),
                ParamSpec("bias", "Intensity Bias", ParamType.INT, 0, -255, 255, 1),
            ),
        ),
        OperationSpec(
            op_id="spatial.point_line_detect",
            name="Point & Line Mask Detection",
            category=OperationCategory.SPATIAL,
            description="Detect isolated points or Horizontal / Vertical / ±45° lines with Chapter 5 masks.",
            func=op_point_line_detect,
            keywords=("point detect", "line detect", "mask", "laplacian", "diagonal"),
            params=(
                ParamSpec("mask_type", "Detector Mask", ParamType.CHOICE, "Isolated Point (Laplacian)", choices=tuple(_POINT_LINE_KERNELS.keys())),
                ParamSpec("threshold_pct", "Peak Ratio Threshold", ParamType.FLOAT, 0.35, 0.05, 0.95, 0.05, 2),
                ParamSpec("overlay", "Highlight Over Image", ParamType.BOOL, True),
            ),
        ),
    ]
