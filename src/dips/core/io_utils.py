"""Unicode-safe image I/O, format conversion, and sample image discovery."""

from __future__ import annotations

from pathlib import Path
import sys
import cv2
import numpy as np


SUPPORTED_OPEN_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm")
SUPPORTED_SAVE_FORMATS = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "WebP": ".webp",
}

BORDER_MODES: dict[str, int] = {
    "Transparent": cv2.BORDER_CONSTANT,
    "Black (Constant)": cv2.BORDER_CONSTANT,
    "Replicate Edge": cv2.BORDER_REPLICATE,
    "Reflect": cv2.BORDER_REFLECT_101,
    "Wrap (Tile)": cv2.BORDER_WRAP,
}


def border_flag(name: str, default: int = cv2.BORDER_CONSTANT) -> int:
    """Resolve an OpenCV border mode flag from a UI label."""
    return BORDER_MODES.get(name, default)


def ensure_bgra(img: np.ndarray) -> np.ndarray:
    """Return a 4-channel BGRA uint8 image, preserving existing alpha or adding opaque alpha."""
    arr = ensure_uint8(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    if arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    return arr


def composite_bgra_to_bgr(img: np.ndarray, bg_bgr: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Alpha-composite a 4-channel BGRA image onto a solid BGR background."""
    arr = ensure_uint8(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 3:
        return arr
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array(bg_bgr, dtype=np.float32).reshape((1, 1, 3))
    comp = arr[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(np.round(comp), 0, 255).astype(np.uint8)


def resource_path(*parts: str) -> Path:
    """Resolve a path relative to project root or PyInstaller bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidate = Path(base).joinpath(*parts)
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize an image array to uint8 with 1 (H,W), 3 (BGR), or 4 (BGRA) channels."""
    if img is None:
        raise ValueError("Image array is None")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return np.ascontiguousarray(img)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR uint8 view/copy of an image."""
    img = ensure_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a 1-channel grayscale uint8 image."""
    img = ensure_uint8(img)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk safely (handles Unicode file paths on Windows/Linux)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    raw = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to decode image: {file_path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    return ensure_uint8(img)


def save_image(
    img: np.ndarray,
    path: str | Path,
    *,
    jpeg_quality: int = 95,
    png_compression: int = 3,
    webp_quality: int = 92,
) -> Path:
    """Save an image to disk safely (handles Unicode paths and encoder parameters)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower() or ".png"
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")

    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(np.clip(png_compression, 0, 9))]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(np.clip(webp_quality, 1, 100))]

    to_encode = ensure_uint8(img)
    if ext in (".jpg", ".jpeg", ".bmp") and to_encode.ndim == 3 and to_encode.shape[2] == 4:
        to_encode = composite_bgra_to_bgr(to_encode)

    ok, encoded = cv2.imencode(ext, to_encode, params)
    if not ok:
        raise ValueError(f"Failed to encode image as '{ext}'")
    encoded.tofile(str(out_path))
    return out_path


def discover_sample_images() -> list[Path]:
    """Locate bundled sample images shipped with DIPS."""
    candidates = [
        resource_path("dips", "samples"),
        Path(__file__).resolve().parents[1] / "samples",
    ]
    seen: set[Path] = set()
    images: list[Path] = []
    for directory in candidates:
        if directory.is_dir():
            for item in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
                if item.is_file() and item.suffix.lower() in SUPPORTED_OPEN_EXTS:
                    resolved = item.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        images.append(item)
            if images:
                break
    return images
