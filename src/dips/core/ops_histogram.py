"""Histogram computation, statistical metrics, equalization, CLAHE, and histogram specification."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_gray
from .types import ImageStats, OperationCategory, OperationSpec, ParamSpec, ParamType


def compute_histogram(img: np.ndarray) -> dict[str, np.ndarray]:
    """Compute 256-bin histograms for luminance and (if color) B, G, R channels."""
    img = ensure_uint8(img)
    if img.ndim == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.int64)
        return {"lum": hist}

    gray = to_gray(img)
    lum = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    b = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    g = cv2.calcHist([img], [1], None, [256], [0, 256]).ravel().astype(np.int64)
    r = cv2.calcHist([img], [2], None, [256], [0, 256]).ravel().astype(np.int64)
    return {"lum": lum, "r": r, "g": g, "b": b}


def compute_stats(img: np.ndarray) -> ImageStats:
    """Compute image dimensions and intensity statistics (mean, std, min, max, median, entropy)."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    gray = to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = float(gray.size)
    probs = hist / max(total, 1.0)
    nz = probs[probs > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    unique_vals = np.unique(gray)
    is_binary = len(unique_vals) <= 2 and set(unique_vals.tolist()).issubset({0, 255})
    return ImageStats(
        width=w,
        height=h,
        channels=ch,
        dtype=str(img.dtype),
        mean=float(gray.mean()),
        std=float(gray.std()),
        min_val=int(gray.min()),
        max_val=int(gray.max()),
        median=float(np.median(gray)),
        entropy=entropy,
        is_binary=is_binary,
    )


def compute_quality_metrics(orig: np.ndarray, res: np.ndarray) -> dict[str, float] | None:
    """Compute PSNR (dB), SSIM, MSE, and MAE between same-resolution Original and Result."""
    o = to_gray(orig).astype(np.float64)
    r = to_gray(res).astype(np.float64)
    if o.shape != r.shape:
        return None
    diff = o - r
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = 99.99 if mse < 1e-10 else float(10.0 * np.log10((255.0 * 255.0) / mse))

    # Standard Gaussian-windowed SSIM (Wang et al. 2004)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu1 = cv2.GaussianBlur(o, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(r, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(o * o, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(r * r, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(o * r, (11, 11), 1.5) - mu1_mu2
    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = float(np.mean(num / np.maximum(den, 1e-12)))
    return {"psnr": psnr, "ssim": ssim, "mse": mse, "mae": mae}


def op_hist_equalize(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Global histogram equalization on Grayscale or Color (YCrCb-Y / LAB-L / HSV-V / RGB)."""
    img = ensure_uint8(img)
    if img.ndim == 2:
        return cv2.equalizeHist(img), {"space": "Grayscale"}

    space = str(params.get("color_space", "YCrCb (Luminance Y)"))
    if space == "Convert to Grayscale":
        return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), {"space": "Grayscale"}
    if space == "LAB (Lightness L)":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), {"space": "LAB-L"}
    if space == "HSV (Value V)":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), {"space": "HSV-V"}
    if space == "RGB Per-Channel":
        chans = [cv2.equalizeHist(img[:, :, c]) for c in range(3)]
        return cv2.merge(chans), {"space": "RGB (Independent)"}

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {"space": "YCrCb-Y"}


def op_clahe(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) with clip limit and tile grid."""
    img = ensure_uint8(img)
    clip = max(0.1, float(params.get("clip_limit", 2.5)))
    grid = max(2, int(params.get("tile_grid", 8)))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    if img.ndim == 2:
        return clahe.apply(img), {"clip_limit": clip, "tile_grid": f"{grid}×{grid}"}

    space = str(params.get("color_space", "LAB (Lightness L)"))
    if space == "YCrCb (Luminance Y)":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif space == "HSV (Value V)":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out, {"clip_limit": clip, "tile_grid": f"{grid}×{grid}", "space": space}


def op_hist_match(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Histogram specification/matching toward a target synthetic distribution (Ch. 3)."""
    img = ensure_uint8(img)
    target = str(params.get("target", "Gaussian (Balanced)"))
    x = np.arange(256, dtype=np.float64)

    if target == "Bimodal (High Contrast)":
        pdf = np.exp(-0.5 * ((x - 55.0) / 28.0) ** 2) + np.exp(-0.5 * ((x - 200.0) / 28.0) ** 2)
    elif target == "Bright Key (High-Key)":
        pdf = np.exp(-0.5 * ((x - 190.0) / 42.0) ** 2) + 0.05
    elif target == "Dark Key (Low-Key)":
        pdf = np.exp(-0.5 * ((x - 65.0) / 42.0) ** 2) + 0.05
    else:
        mean = float(params.get("mean", 128.0))
        sigma = max(10.0, float(params.get("sigma", 50.0)))
        pdf = np.exp(-0.5 * ((x - mean) / sigma) ** 2) + 1e-4

    target_cdf = np.cumsum(pdf)
    target_cdf /= target_cdf[-1]

    def _match_ch(ch: np.ndarray) -> np.ndarray:
        src_hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        src_cdf = np.cumsum(src_hist)
        if src_cdf[-1] > 0:
            src_cdf /= src_cdf[-1]
        lut = np.interp(src_cdf, target_cdf, x)
        return cv2.LUT(ch, np.clip(np.round(lut), 0, 255).astype(np.uint8))

    if img.ndim == 2:
        return _match_ch(img), {"target": target}
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _match_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {"target": target}


def get_specs() -> list[OperationSpec]:
    """Return all Histogram operation specifications."""
    return [
        OperationSpec(
            op_id="hist.equalize",
            name="Histogram Equalization",
            category=OperationCategory.HISTOGRAM,
            description="Redistribute intensities to flatten the histogram and maximize global contrast.",
            func=op_hist_equalize,
            keywords=("histogram", "equalization", "cdf", "contrast"),
            params=(
                ParamSpec(
                    "color_space",
                    "Color Mode",
                    ParamType.CHOICE,
                    "YCrCb (Luminance Y)",
                    choices=(
                        "YCrCb (Luminance Y)",
                        "LAB (Lightness L)",
                        "HSV (Value V)",
                        "RGB Per-Channel",
                        "Convert to Grayscale",
                    ),
                ),
            ),
        ),
        OperationSpec(
            op_id="hist.clahe",
            name="CLAHE (Adaptive Equalize)",
            category=OperationCategory.HISTOGRAM,
            description="Tile-based local histogram equalization with contrast limiting to prevent noise blow-up.",
            func=op_clahe,
            keywords=("clahe", "adaptive", "local", "histogram", "contrast"),
            params=(
                ParamSpec("clip_limit", "Clip Limit", ParamType.FLOAT, 2.5, 0.5, 20.0, 0.25, 2),
                ParamSpec("tile_grid", "Tile Grid Size", ParamType.INT, 8, 2, 32, 1),
                ParamSpec(
                    "color_space",
                    "Color Channel",
                    ParamType.CHOICE,
                    "LAB (Lightness L)",
                    choices=("LAB (Lightness L)", "YCrCb (Luminance Y)", "HSV (Value V)"),
                ),
            ),
        ),
        OperationSpec(
            op_id="hist.match",
            name="Histogram Specification",
            category=OperationCategory.HISTOGRAM,
            description="Match luminance distribution to a target Gaussian, Bimodal, High-Key, or Low-Key PDF.",
            func=op_hist_match,
            keywords=("specification", "matching", "pdf", "cdf", "histogram"),
            params=(
                ParamSpec(
                    "target",
                    "Target Distribution",
                    ParamType.CHOICE,
                    "Gaussian (Balanced)",
                    choices=("Gaussian (Balanced)", "Bimodal (High Contrast)", "Bright Key (High-Key)", "Dark Key (Low-Key)"),
                ),
                ParamSpec("mean", "Gaussian Mean", ParamType.FLOAT, 128.0, 20.0, 235.0, 2.0, 0),
                ParamSpec("sigma", "Gaussian Spread (σ)", ParamType.FLOAT, 50.0, 15.0, 120.0, 2.0, 0),
            ),
        ),
    ]
