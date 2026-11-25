"""ONNX model catalog, resolution, and atomic on-demand / build-time downloader."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import urllib.request

MODELS_MANIFEST: dict[str, dict[str, str]] = {
    "4x-UltraSharp-fp16-opset17.onnx": {
        "url": "https://huggingface.co/Kim2091/UltraSharp/resolve/main/ONNX/4x-UltraSharp-fp16-opset17.onnx",
        "desc": "4x-UltraSharp Super-Resolution (FP16)",
    },
    "super-resolution-10.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
        "desc": "ESPCN 3x Fast Super-Resolution",
    },
    "rmbg-1.4-quantized.onnx": {
        "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
        "desc": "BRIA RMBG-1.4 Studio Background Removal (Quantized)",
    },
    "modnet.onnx": {
        "url": "https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx",
        "desc": "MODNet Studio Portrait Matting",
    },
    "u2netp.onnx": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "desc": "U2Net-p Salient Foreground Matting",
    },
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "desc": "YuNet Face & 5-Point Landmark Detector",
    },
    "mosaic-9.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "desc": "Fast Neural Style - Mosaic",
    },
    "candy-9.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        "desc": "Fast Neural Style - Candy",
    },
    "depth-anything-v2-small-q.onnx": {
        "url": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx",
        "desc": "Depth Anything V2 Small Monocular Depth (Quantized)",
    },
    "AnimeGANv2_Hayao.onnx": {
        "url": "https://huggingface.co/vumichien/AnimeGANv2_Hayao/resolve/main/AnimeGANv2_Hayao.onnx",
        "desc": "AnimeGANv2 Hayao Illustration Style",
    },
    "yolov8n.onnx": {
        "url": "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
        "desc": "YOLOv8 Nano 80-Class Object Detector",
    },
}


def package_models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def user_models_cache_dir() -> Path:
    return Path.home() / ".cache" / "dips" / "models"


def find_model_file(filename: str) -> Path | None:
    """Search frozen bundle, source models directory, and user cache."""
    base = getattr(sys, "_MEIPASS", None)
    candidates = [
        Path(base) / "dips" / "models" / filename if base else None,
        package_models_dir() / filename,
        user_models_cache_dir() / filename,
    ]
    for path in candidates:
        if path is not None and path.is_file() and path.stat().st_size > 1024:
            return path
    return None


def download_file_atomic(url: str, dest: Path) -> Path:
    """Download a file to a temporary path then atomically replace `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "DIPS/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1024 * 256)
    tmp.replace(dest)
    return dest


def ensure_model(filename: str) -> Path:
    """Return local path to `filename`, downloading it to user cache if missing."""
    existing = find_model_file(filename)
    if existing is not None:
        return existing

    meta = MODELS_MANIFEST.get(filename)
    if meta is None:
        raise FileNotFoundError(f"Unknown model requested: {filename}")

    target_dir = package_models_dir() if not getattr(sys, "_MEIPASS", None) else user_models_cache_dir()
    try:
        return download_file_atomic(meta["url"], target_dir / filename)
    except PermissionError:
        return download_file_atomic(meta["url"], user_models_cache_dir() / filename)


def download_all_models(target_dir: Path | None = None) -> list[Path]:
    """Download all manifest models into `target_dir` (defaults to src/dips/models)."""
    dest_root = target_dir or package_models_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, meta in MODELS_MANIFEST.items():
        dest = dest_root / filename
        if dest.is_file() and dest.stat().st_size > 1024:
            print(f"[ok] {filename} ({dest.stat().st_size // 1024} KB)")
            paths.append(dest)
            continue
        print(f"[downloading] {meta['desc']} -> {filename} ...")
        download_file_atomic(meta["url"], dest)
        print(f"[done] {filename} ({dest.stat().st_size // 1024} KB)")
        paths.append(dest)
    return paths
