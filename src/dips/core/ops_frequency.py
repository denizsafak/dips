"""Frequency-domain (2D FFT) operations: Magnitude Spectrum, Low-Pass, High-Pass, Band/Notch, Homomorphic."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def _distance_grid(h: int, w: int) -> np.ndarray:
    cy, cx = h / 2.0, w / 2.0
    y = np.arange(h, dtype=np.float64)[:, None] - cy
    x = np.arange(w, dtype=np.float64)[None, :] - cx
    return np.hypot(y, x)


def _transfer_mask(
    h: int,
    w: int,
    cutoff: float,
    kind: str,
    order: int,
    *,
    highpass: bool = False,
) -> np.ndarray:
    d = _distance_grid(h, w)
    d0 = max(1.0, float(cutoff))
    if kind == "Ideal":
        mask = (d <= d0).astype(np.float64)
    elif kind == "Butterworth":
        n = max(1, int(order))
        # Avoid division by zero at DC when computing highpass directly
        if highpass:
            ratio = np.where(d > 1e-6, (d0 / np.maximum(d, 1e-6)) ** (2 * n), 1e12)
            return 1.0 / (1.0 + ratio)
        mask = 1.0 / (1.0 + (d / d0) ** (2 * n))
    else:  # Gaussian
        mask = np.exp(-(d * d) / (2.0 * d0 * d0))
    return (1.0 - mask) if highpass else mask


def _apply_freq_mask_channel(ch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
    filtered = f * mask
    spatial = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return np.clip(np.round(spatial), 0, 255).astype(np.uint8)


def _apply_freq_mask(img: np.ndarray, mask: np.ndarray, lum_only: bool) -> np.ndarray:
    img = ensure_uint8(img)
    if img.ndim == 2:
        return _apply_freq_mask_channel(img, mask)
    if lum_only:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = _apply_freq_mask_channel(ycrcb[:, :, 0], mask)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    chans = [_apply_freq_mask_channel(img[:, :, c], mask) for c in range(3)]
    return cv2.merge(chans)


def op_fft_spectrum(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute centered 2D FFT log-magnitude spectrum."""
    gray = to_gray(img).astype(np.float64)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fshift))
    gamma = float(params.get("gamma", 1.0))
    norm = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    if abs(gamma - 1.0) > 1e-3:
        norm = np.power(norm, max(0.2, gamma))
    u8 = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    if bool(params.get("colormap", True)):
        u8 = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return u8, {"dc_magnitude": round(float(np.abs(fshift[gray.shape[0] // 2, gray.shape[1] // 2])), 1)}


def op_freq_lowpass(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Ideal, Butterworth, or Gaussian frequency-domain low-pass smoothing."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    kind = str(params.get("filter_type", "Gaussian"))
    cutoff = float(params.get("cutoff", 45.0))
    order = int(params.get("order", 2))
    lum_only = bool(params.get("luminance_only", True))
    mask = _transfer_mask(h, w, cutoff, kind, order, highpass=False)
    out = _apply_freq_mask(img, mask, lum_only)
    return out, {"type": kind, "D0": cutoff}


def op_freq_highpass(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Frequency-domain high-pass edge extraction or high-frequency emphasis."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    kind = str(params.get("filter_type", "Gaussian"))
    cutoff = float(params.get("cutoff", 30.0))
    order = int(params.get("order", 2))
    emphasis = float(params.get("emphasis", 0.0))
    lum_only = bool(params.get("luminance_only", True))

    hp = _transfer_mask(h, w, cutoff, kind, order, highpass=True)
    if emphasis > 0.0:
        # High-frequency emphasis: a + b * H_hp(u,v)
        mask = 1.0 + emphasis * hp
        out = _apply_freq_mask(img, mask, lum_only)
        return out, {"type": f"{kind} Emphasis", "D0": cutoff, "boost": emphasis}

    gray = to_gray(img).astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(gray))
    spatial = np.abs(np.real(np.fft.ifft2(np.fft.ifftshift(f * hp))))
    out = np.clip(cv2.normalize(spatial, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
    return out, {"type": kind, "D0": cutoff}


def op_freq_band(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Butterworth Band-Pass or Band-Reject (ring notch) frequency filter."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    mode = str(params.get("mode", "Band-Reject (Notch)"))
    d_low = float(params.get("d_low", 20.0))
    d_high = max(d_low + 2.0, float(params.get("d_high", 70.0)))
    order = max(1, int(params.get("order", 2)))
    lum_only = bool(params.get("luminance_only", True))

    d = _distance_grid(h, w)
    center = 0.5 * (d_low + d_high)
    width = max(1.0, d_high - d_low)
    denom = np.where(np.abs(d * d - center * center) > 1e-6, d * d - center * center, 1e-6)
    br = 1.0 / (1.0 + ((d * width) / denom) ** (2 * order))
    mask = br if mode.startswith("Band-Reject") else (1.0 - br)
    out = _apply_freq_mask(img, mask, lum_only)
    return out, {"mode": mode, "band": f"[{d_low:.0f}, {d_high:.0f}]"}


def op_homomorphic(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Homomorphic illumination/reflectance filter (compresses uneven lighting, boosts detail)."""
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    g_low = float(params.get("gamma_low", 0.5))
    g_high = float(params.get("gamma_high", 1.8))
    cutoff = max(2.0, float(params.get("cutoff", 35.0)))
    c_sharp = float(params.get("c", 1.0))

    d2 = _distance_grid(h, w) ** 2
    hp = 1.0 - np.exp(-c_sharp * (d2 / (cutoff * cutoff)))
    H = (g_high - g_low) * hp + g_low

    def _run_ch(ch: np.ndarray) -> np.ndarray:
        log_img = np.log1p(ch.astype(np.float64))
        f = np.fft.fftshift(np.fft.fft2(log_img))
        spatial = np.real(np.fft.ifft2(np.fft.ifftshift(f * H)))
        exp_img = np.expm1(spatial)
        norm = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(np.round(norm), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _run_ch(img), {"gammaL": g_low, "gammaH": g_high}
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _run_ch(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {"gammaL": g_low, "gammaH": g_high, "D0": cutoff}


def get_specs() -> list[OperationSpec]:
    """Return all Frequency Domain (2D FFT) operation specifications."""
    return [
        OperationSpec(
            op_id="freq.spectrum",
            name="2D FFT Magnitude Spectrum",
            category=OperationCategory.FREQUENCY,
            description="Visualize centered 2D Fourier log-magnitude frequency spectrum.",
            func=op_fft_spectrum,
            keywords=("fft", "dft", "fourier", "spectrum", "frequency"),
            params=(
                ParamSpec("colormap", "Inferno Colormap", ParamType.BOOL, True),
                ParamSpec("gamma", "Display Gamma", ParamType.FLOAT, 1.0, 0.3, 2.5, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="freq.lowpass",
            name="Frequency Low-Pass Filter",
            category=OperationCategory.FREQUENCY,
            description="Attenuate high frequencies in the 2D FFT domain (Ideal, Butterworth, or Gaussian).",
            func=op_freq_lowpass,
            keywords=("fft", "lowpass", "butterworth", "ideal", "smooth"),
            params=(
                ParamSpec("filter_type", "Transfer Function", ParamType.CHOICE, "Gaussian", choices=("Gaussian", "Butterworth", "Ideal")),
                ParamSpec("cutoff", "Cutoff Radius (D0)", ParamType.FLOAT, 45.0, 4.0, 300.0, 2.0, 0, unit="px"),
                ParamSpec("order", "Butterworth Order (n)", ParamType.INT, 2, 1, 8, 1),
                ParamSpec("luminance_only", "Filter Luminance Only", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="freq.highpass",
            name="Frequency High-Pass Filter",
            category=OperationCategory.FREQUENCY,
            description="Isolate fine detail/edges or perform High-Frequency Emphasis in 2D FFT space.",
            func=op_freq_highpass,
            keywords=("fft", "highpass", "edges", "emphasis", "sharpen"),
            params=(
                ParamSpec("filter_type", "Transfer Function", ParamType.CHOICE, "Gaussian", choices=("Gaussian", "Butterworth", "Ideal")),
                ParamSpec("cutoff", "Cutoff Radius (D0)", ParamType.FLOAT, 30.0, 2.0, 250.0, 2.0, 0, unit="px"),
                ParamSpec("order", "Butterworth Order (n)", ParamType.INT, 2, 1, 8, 1),
                ParamSpec("emphasis", "High-Freq Boost (0 = Edges Only)", ParamType.FLOAT, 0.0, 0.0, 3.0, 0.1, 1),
                ParamSpec("luminance_only", "Filter Luminance Only", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="freq.band",
            name="Band-Pass / Band-Reject Filter",
            category=OperationCategory.FREQUENCY,
            description="Suppress or isolate periodic frequency rings [D_low, D_high] via Butterworth transfer.",
            func=op_freq_band,
            keywords=("bandpass", "bandreject", "notch", "periodic noise", "fft"),
            params=(
                ParamSpec("mode", "Mode", ParamType.CHOICE, "Band-Reject (Notch)", choices=("Band-Reject (Notch)", "Band-Pass")),
                ParamSpec("d_low", "Inner Radius (D_low)", ParamType.FLOAT, 20.0, 2.0, 200.0, 2.0, 0, unit="px"),
                ParamSpec("d_high", "Outer Radius (D_high)", ParamType.FLOAT, 70.0, 5.0, 300.0, 2.0, 0, unit="px"),
                ParamSpec("order", "Butterworth Order", ParamType.INT, 2, 1, 6, 1),
                ParamSpec("luminance_only", "Filter Luminance Only", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="freq.homomorphic",
            name="Homomorphic Illumination Filter",
            category=OperationCategory.FREQUENCY,
            description="Suppress low-frequency uneven lighting while enhancing high-frequency surface reflectance.",
            func=op_homomorphic,
            keywords=("homomorphic", "illumination", "reflectance", "fft", "shadows"),
            params=(
                ParamSpec("gamma_low", "Low-Freq Gain (γL)", ParamType.FLOAT, 0.50, 0.1, 1.2, 0.05, 2),
                ParamSpec("gamma_high", "High-Freq Gain (γH)", ParamType.FLOAT, 1.80, 1.0, 4.0, 0.1, 2),
                ParamSpec("cutoff", "Cutoff Frequency (D0)", ParamType.FLOAT, 35.0, 5.0, 200.0, 2.0, 0, unit="px"),
                ParamSpec("c", "Slope Sharpness (c)", ParamType.FLOAT, 1.0, 0.2, 4.0, 0.1, 1),
            ),
        ),
    ]
