"""Noise synthesis and image restoration: Gaussian/S&P noise, Non-Local Means, Motion Blur, Wiener Deconvolution, Inpainting, Dehazing."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_bgr, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def _motion_psf(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length))
    size = length if length % 2 == 1 else length + 1
    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    cv2.line(psf, (0, center), (size - 1, center), 1.0, 1)
    M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    psf = cv2.warpAffine(psf, M, (size, size), flags=cv2.INTER_LINEAR)
    s = float(psf.sum())
    if s > 1e-8:
        psf /= s
    return psf


def op_add_noise(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Add synthetic Gaussian, Salt & Pepper, Speckle, or Poisson noise."""
    img = ensure_uint8(img)
    kind = str(params.get("noise_type", "Salt & Pepper"))
    amount = float(params.get("amount", 0.05))
    sigma = float(params.get("sigma", 22.0))
    seed = int(params.get("seed", 42))
    rng = np.random.default_rng(seed)

    if kind == "Gaussian":
        noise = rng.normal(0.0, sigma, size=img.shape)
        out = np.clip(np.round(img.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        return out, {"type": "Gaussian", "sigma": sigma}

    if kind == "Speckle (Multiplicative)":
        noise = rng.normal(0.0, max(0.01, sigma / 100.0), size=img.shape)
        out = np.clip(np.round(img.astype(np.float32) * (1.0 + noise)), 0, 255).astype(np.uint8)
        return out, {"type": "Speckle", "std": round(sigma / 100.0, 2)}

    if kind == "Poisson (Shot Noise)":
        scale = max(1.0, 255.0 / max(1.0, sigma))
        vals = rng.poisson(np.maximum(img.astype(np.float64), 0.0) / 255.0 * scale) / scale * 255.0
        return np.clip(np.round(vals), 0, 255).astype(np.uint8), {"type": "Poisson"}

    # Salt & Pepper
    out = img.copy()
    prob = float(np.clip(amount, 0.001, 0.6))
    rand = rng.random(img.shape[:2])
    salt = rand < (prob * 0.5)
    pepper = (rand >= (prob * 0.5)) & (rand < prob)
    if out.ndim == 2:
        out[salt] = 255
        out[pepper] = 0
    elif out.shape[2] == 4:
        out[salt, :3] = 255
        out[pepper, :3] = 0
    else:
        out[salt] = [255, 255, 255]
        out[pepper] = [0, 0, 0]
    return out, {"type": "Salt & Pepper", "density": f"{prob * 100:.1f}%"}


def op_nlm_denoise(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Non-Local Means patch-based denoising."""
    img = ensure_uint8(img)
    h_lum = float(params.get("h", 10.0))
    h_col = float(params.get("h_color", 10.0))
    tmpl = int(params.get("template_window", 7)) | 1
    srch = int(params.get("search_window", 21)) | 1
    if img.ndim == 2:
        out = cv2.fastNlMeansDenoising(img, None, h_lum, tmpl, srch)
    else:
        out = cv2.fastNlMeansDenoisingColored(img, None, h_lum, h_col, tmpl, srch)
    return out, {"h": h_lum, "window": f"{tmpl}/{srch}"}


def op_motion_blur(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate linear camera/subject motion blur at given length and angle."""
    img = ensure_uint8(img)
    length = int(params.get("length", 15))
    angle = float(params.get("angle", 25.0))
    psf = _motion_psf(length, angle)
    out = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REPLICATE)
    return out, {"length_px": length, "angle": f"{angle:.0f}°"}


def op_wiener_deconv(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Frequency-domain Wiener deconvolution for Gaussian or Motion blur restoration."""
    img = ensure_uint8(img)
    psf_type = str(params.get("psf_type", "Motion"))
    psf_size = max(3, int(params.get("psf_size", 15)))
    angle = float(params.get("angle", 25.0))
    nsr = max(1e-5, float(params.get("nsr", 0.02)))

    if psf_type == "Gaussian":
        k = psf_size if psf_size % 2 == 1 else psf_size + 1
        g1 = cv2.getGaussianKernel(k, max(0.5, k / 5.0))
        psf = g1 @ g1.T
    else:
        psf = _motion_psf(psf_size, angle)

    h, w = img.shape[:2]
    pad_psf = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    if kh > h or kw > w:
        psf = cv2.resize(psf, (min(kw, w), min(kh, h)))
        psf /= max(float(psf.sum()), 1e-8)
        kh, kw = psf.shape
    pad_psf[:kh, :kw] = psf
    # Center PSF at (0,0) via circular roll for zero-phase FFT alignment
    pad_psf = np.roll(pad_psf, -(kh // 2), axis=0)
    pad_psf = np.roll(pad_psf, -(kw // 2), axis=1)

    H = np.fft.fft2(pad_psf)
    W_filter = np.conj(H) / (np.abs(H) ** 2 + nsr)

    def _restore_ch(ch: np.ndarray) -> np.ndarray:
        F = np.fft.fft2(ch.astype(np.float64))
        restored = np.real(np.fft.ifft2(F * W_filter))
        return np.clip(np.round(restored), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _restore_ch(img), {"psf": psf_type, "nsr": nsr}
    return cv2.merge([_restore_ch(img[:, :, c]) for c in range(3)]), {"psf": psf_type, "size": psf_size, "nsr": nsr}


def op_auto_inpaint(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect bright or dark dust/scratch artifacts and reconstruct via Telea or Navier-Stokes inpainting."""
    bgr = to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    target = str(params.get("target", "Bright Scratches / Spots"))
    thresh = int(params.get("threshold", 235))
    radius = max(1, int(params.get("radius", 3)))
    method = cv2.INPAINT_TELEA if params.get("method", "Telea (Fast Marching)") == "Telea (Fast Marching)" else cv2.INPAINT_NS

    if target.startswith("Bright"):
        mask = (gray >= thresh).astype(np.uint8) * 255
    else:
        mask = (gray <= (255 - thresh)).astype(np.uint8) * 255

    dilate_px = int(params.get("mask_dilate", 1))
    if dilate_px > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, se)

    out = cv2.inpaint(bgr, mask, radius, method)
    pct = round(float(np.count_nonzero(mask)) / max(mask.size, 1) * 100.0, 2)
    return out, {"repaired_area": f"{pct}%", "radius": radius}


def op_dehaze(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-image Dark Channel Prior atmospheric dehazing."""
    bgr = to_bgr(img).astype(np.float64) / 255.0
    patch = max(3, int(params.get("patch_size", 15)) | 1)
    omega = float(params.get("omega", 0.90))
    t_min = float(params.get("t_min", 0.15))

    min_ch = np.min(bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(min_ch, kernel)

    # Estimate atmospheric light from top 0.1% brightest pixels in dark channel
    flat_dark = dark.ravel()
    flat_bgr = bgr.reshape(-1, 3)
    n_top = max(1, int(flat_dark.size * 0.001))
    top_idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    atmo = np.clip(flat_bgr[top_idx].mean(axis=0), 0.1, 1.0)

    norm_bgr = bgr / atmo.reshape(1, 1, 3)
    norm_dark = cv2.erode(np.min(norm_bgr, axis=2), kernel)
    trans = 1.0 - omega * norm_dark
    trans = cv2.GaussianBlur(trans, (patch * 2 + 1, patch * 2 + 1), 0)
    trans = np.maximum(trans, t_min)

    recovered = (bgr - atmo.reshape(1, 1, 3)) / trans[:, :, None] + atmo.reshape(1, 1, 3)
    out = np.clip(np.round(recovered * 255.0), 0, 255).astype(np.uint8)
    return out, {"omega": omega, "patch": f"{patch}×{patch}"}


def get_specs() -> list[OperationSpec]:
    """Return all Noise & Restoration operation specifications."""
    return [
        OperationSpec(
            op_id="restore.noise",
            name="Add Synthetic Noise",
            category=OperationCategory.RESTORATION,
            description="Inject Gaussian, Salt & Pepper, Speckle, or Poisson noise for restoration benchmarking.",
            func=op_add_noise,
            keywords=("noise", "salt", "pepper", "gaussian", "speckle", "poisson"),
            params=(
                ParamSpec(
                    "noise_type",
                    "Noise Model",
                    ParamType.CHOICE,
                    "Salt & Pepper",
                    choices=("Salt & Pepper", "Gaussian", "Speckle (Multiplicative)", "Poisson (Shot Noise)"),
                ),
                ParamSpec("amount", "S&P Density", ParamType.FLOAT, 0.05, 0.005, 0.40, 0.005, 3),
                ParamSpec("sigma", "Noise Sigma (σ)", ParamType.FLOAT, 22.0, 2.0, 80.0, 2.0, 0),
                ParamSpec("seed", "Random Seed", ParamType.INT, 42, 0, 9999, 1),
            ),
        ),
        OperationSpec(
            op_id="restore.nlm",
            name="Non-Local Means Denoise",
            category=OperationCategory.RESTORATION,
            description="Patch-similarity denoising preserving fine texture better than Gaussian/Median.",
            func=op_nlm_denoise,
            keywords=("denoise", "nlm", "non-local means", "clean"),
            params=(
                ParamSpec("h", "Luminance Filter (h)", ParamType.FLOAT, 10.0, 1.0, 35.0, 1.0, 1),
                ParamSpec("h_color", "Color Filter (hColor)", ParamType.FLOAT, 10.0, 1.0, 35.0, 1.0, 1),
                ParamSpec("template_window", "Patch Size", ParamType.INT, 7, 3, 15, 2, odd_only=True),
                ParamSpec("search_window", "Search Window", ParamType.INT, 21, 9, 35, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="restore.motion_blur",
            name="Motion Blur Simulation",
            category=OperationCategory.RESTORATION,
            description="Convolve with a directional linear Point Spread Function (PSF).",
            func=op_motion_blur,
            keywords=("motion blur", "psf", "shake", "streak"),
            params=(
                ParamSpec("length", "Blur Length", ParamType.INT, 15, 3, 75, 2, unit="px"),
                ParamSpec("angle", "Direction Angle", ParamType.FLOAT, 25.0, -180.0, 180.0, 5.0, 0, unit="°"),
            ),
        ),
        OperationSpec(
            op_id="restore.wiener",
            name="Wiener Deconvolution",
            category=OperationCategory.RESTORATION,
            description="Invert Motion or Gaussian blur PSF in the frequency domain with noise regularization.",
            func=op_wiener_deconv,
            keywords=("wiener", "deconvolution", "deblur", "restore", "psf"),
            params=(
                ParamSpec("psf_type", "PSF Model", ParamType.CHOICE, "Motion", choices=("Motion", "Gaussian")),
                ParamSpec("psf_size", "PSF Size / Length", ParamType.INT, 15, 3, 75, 2, unit="px"),
                ParamSpec("angle", "Motion Angle", ParamType.FLOAT, 25.0, -180.0, 180.0, 5.0, 0, unit="°"),
                ParamSpec("nsr", "Noise-to-Signal Ratio (K)", ParamType.FLOAT, 0.02, 0.001, 0.25, 0.005, 3),
            ),
        ),
        OperationSpec(
            op_id="restore.inpaint",
            name="Spot & Scratch Inpainting",
            category=OperationCategory.RESTORATION,
            description="Automatically mask extreme bright/dark specks and reconstruct via Telea or Navier-Stokes.",
            func=op_auto_inpaint,
            keywords=("inpaint", "repair", "scratch", "dust", "retouch"),
            params=(
                ParamSpec("target", "Artifact Target", ParamType.CHOICE, "Bright Scratches / Spots", choices=("Bright Scratches / Spots", "Dark Scratches / Spots")),
                ParamSpec("threshold", "Detection Threshold", ParamType.INT, 235, 120, 255, 2),
                ParamSpec("mask_dilate", "Mask Padding", ParamType.INT, 1, 0, 6, 1, unit="px"),
                ParamSpec("radius", "Inpaint Radius", ParamType.INT, 3, 1, 15, 1, unit="px"),
                ParamSpec("method", "Algorithm", ParamType.CHOICE, "Telea (Fast Marching)", choices=("Telea (Fast Marching)", "Navier-Stokes")),
            ),
        ),
        OperationSpec(
            op_id="restore.dehaze",
            name="Atmospheric Dehaze (DCP)",
            category=OperationCategory.RESTORATION,
            description="Remove haze/fog and restore contrast using Dark Channel Prior transmission estimation.",
            func=op_dehaze,
            keywords=("dehaze", "fog", "mist", "dark channel", "contrast"),
            params=(
                ParamSpec("omega", "Haze Removal Strength (ω)", ParamType.FLOAT, 0.90, 0.1, 1.0, 0.05, 2),
                ParamSpec("patch_size", "Dark Channel Patch", ParamType.INT, 15, 3, 35, 2, odd_only=True),
                ParamSpec("t_min", "Min Transmission Floor", ParamType.FLOAT, 0.15, 0.05, 0.50, 0.02, 2),
            ),
        ),
    ]
