"""AI Studio operations powered by OpenCV DNN and ONNX Runtime."""

from __future__ import annotations

import atexit
from functools import lru_cache
import gc
from typing import Any
import cv2
import numpy as np
import onnxruntime as ort

from .io_utils import ensure_uint8, to_bgr
from .model_manager import ensure_model
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


_PREFERRED_GPU_PROVIDERS = (
    ("CUDAExecutionProvider", "GPU (CUDA)"),
    ("CoreMLExecutionProvider", "GPU (CoreML)"),
    ("DmlExecutionProvider", "GPU (DirectML)"),
    ("ROCMExecutionProvider", "GPU (ROCm)"),
    ("OpenVINOExecutionProvider", "GPU (OpenVINO)"),
)


def get_preferred_ort_providers() -> list[str]:
    """Return available ONNX Runtime execution providers ordered GPU-first with CPU fallback."""
    avail = set(ort.get_available_providers())
    providers = [prov for prov, _label in _PREFERRED_GPU_PROVIDERS if prov in avail]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def session_device_label(sess: ort.InferenceSession) -> str:
    """Return human-readable active device label for an ORT session."""
    active = sess.get_providers()
    if active:
        top = active[0]
        for prov, label in _PREFERRED_GPU_PROVIDERS:
            if top == prov:
                return label
    return "CPU"


@lru_cache(maxsize=8)
def _load_cv_onnx(filename: str) -> cv2.dnn.Net:
    path = ensure_model(filename)
    return cv2.dnn.readNetFromONNX(str(path))


@lru_cache(maxsize=4)
def _load_ort_session(filename: str) -> ort.InferenceSession:
    path = ensure_model(filename)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    preferred = get_preferred_ort_providers()
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=preferred)
    except Exception:
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=4)
def _load_yunet(score_thresh: float, nms_thresh: float) -> cv2.FaceDetectorYN:
    path = ensure_model("face_detection_yunet_2023mar.onnx")
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_thresh, nms_thresh, 5000)


def release_ai_sessions() -> None:
    """Release cached ONNX Runtime / OpenCV DNN sessions before C++ atexit teardown."""
    _load_ort_session.cache_clear()
    _load_cv_onnx.cache_clear()
    _load_yunet.cache_clear()
    gc.collect()


atexit.register(release_ai_sessions)


def _predict_u2net_matte(bgr: np.ndarray) -> np.ndarray:
    """Return a [0, 1] float32 foreground matte of shape (H, W) using u2netp.onnx."""
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    blob = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    net = _load_cv_onnx("u2netp.onnx")
    net.setInput(blob)
    pred = net.forward()
    matte_small = pred[0, 0]
    mn, mx = float(matte_small.min()), float(matte_small.max())
    if mx > mn + 1e-6:
        matte_small = (matte_small - mn) / (mx - mn)
    matte = cv2.resize(matte_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def _predict_modnet_matte(bgr: np.ndarray) -> np.ndarray:
    """Return a [0, 1] float32 portrait matte of shape (H, W) using modnet.onnx."""
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm = (rgb - 127.5) / 127.5
    blob = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    sess = _load_ort_session("modnet.onnx")
    inp_name = sess.get_inputs()[0].name
    matte_small = sess.run(None, {inp_name: blob})[0][0, 0]
    matte = cv2.resize(matte_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def _predict_rmbg14_matte(bgr: np.ndarray) -> np.ndarray:
    """Return a [0, 1] float32 high-precision matte using BRIA RMBG-1.4 (rmbg-1.4-quantized.onnx)."""
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = rgb - 0.5
    blob = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    sess = _load_ort_session("rmbg-1.4-quantized.onnx")
    inp_name = sess.get_inputs()[0].name
    matte_small = sess.run(None, {inp_name: blob})[0][0, 0]
    mn, mx = float(matte_small.min()), float(matte_small.max())
    if mx > mn + 1e-6:
        matte_small = (matte_small - mn) / (mx - mn)
    matte = cv2.resize(matte_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)


def _matte_model_meta(model_choice: str) -> tuple[str, str]:
    if model_choice.startswith("RMBG-1.4"):
        return "rmbg-1.4-quantized.onnx", "RMBG-1.4"
    if model_choice.startswith("MODNet"):
        return "modnet.onnx", "MODNet"
    return "u2netp.onnx", "U2NetP"


def _predict_matte(bgr: np.ndarray, model_choice: str) -> np.ndarray:
    if model_choice.startswith("RMBG-1.4"):
        return _predict_rmbg14_matte(bgr)
    if model_choice.startswith("MODNet"):
        return _predict_modnet_matte(bgr)
    return _predict_u2net_matte(bgr)


def _apply_privacy_mask(roi: np.ndarray, mode: str, strength: int) -> np.ndarray:
    """Anonymize an image ROI using Mosaic pixelation, Gaussian blur, or solid charcoal mask."""
    if mode == "Gaussian Blur":
        k = (strength * 2 + 1) | 1
        return cv2.GaussianBlur(roi, (k, k), 0)
    if mode == "Solid Charcoal":
        out = roi.copy()
        out[:] = (28, 31, 38)
        return out
    rw, rh = roi.shape[1], roi.shape[0]
    blocks_x = max(2, rw // max(2, strength))
    blocks_y = max(2, rh // max(2, strength))
    tiny = cv2.resize(roi, (blocks_x, blocks_y), interpolation=cv2.INTER_AREA)
    return cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_NEAREST)


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return sorted(set(starts))


def _run_espcn_3x_y(y_channel: np.ndarray) -> np.ndarray:
    """Tile-based 3x super-resolution on uint8 luminance using super-resolution-10.onnx."""
    net = _load_cv_onnx("super-resolution-10.onnx")
    h, w = y_channel.shape
    tile = 224
    overlap = 16
    scale = 3

    ys = _tile_starts(h, tile, overlap)
    xs = _tile_starts(w, tile, overlap)
    req_h = max(tile, ys[-1] + tile)
    req_w = max(tile, xs[-1] + tile)
    src = cv2.copyMakeBorder(y_channel, 0, req_h - h, 0, req_w - w, cv2.BORDER_REFLECT_101)
    y_norm = src.astype(np.float32) / 255.0

    out_h, out_w = h * scale, w * scale
    acc = np.zeros((out_h, out_w), dtype=np.float64)
    wsum = np.zeros((out_h, out_w), dtype=np.float64)

    win1d = np.maximum(np.hanning(tile * scale), 0.05)
    win2d = np.outer(win1d, win1d)

    for y0 in ys:
        for x0 in xs:
            patch = np.ascontiguousarray(y_norm[y0 : y0 + tile, x0 : x0 + tile])
            net.setInput(patch[None, None, :, :])
            sr_patch = net.forward()[0, 0]
            oy, ox = y0 * scale, x0 * scale
            ey, ex = min(out_h, oy + tile * scale), min(out_w, ox + tile * scale)
            ph, pw = ey - oy, ex - ox
            acc[oy:ey, ox:ex] += sr_patch[:ph, :pw] * win2d[:ph, :pw]
            wsum[oy:ey, ox:ex] += win2d[:ph, :pw]

    merged = acc / np.maximum(wsum, 1e-8)
    return np.clip(np.round(merged * 255.0), 0, 255).astype(np.uint8)


def _run_ultrasharp_4x_bgr(bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Tiled 4x neural super-resolution using 4x-UltraSharp-fp16-opset17.onnx via ORT."""
    sess = _load_ort_session("4x-UltraSharp-fp16-opset17.onnx")
    device_label = session_device_label(sess)
    on_gpu = device_label != "CPU"
    inp = sess.get_inputs()[0]
    inp_dtype = np.float16 if "float16" in inp.type else np.float32

    h, w = bgr.shape[:2]
    max_single = 384 if on_gpu else 256
    tile_cap = 256 if on_gpu else 192
    tile = min(tile_cap, max(64, max(h, w)))
    if h <= max_single and w <= max_single:
        tile_h, tile_w = h, w
        ys, xs = [0], [0]
    else:
        tile_h = tile_w = tile_cap
        ys = _tile_starts(h, tile_h, 16)
        xs = _tile_starts(w, tile_w, 16)

    scale = 4
    req_h = max(tile_h, ys[-1] + tile_h)
    req_w = max(tile_w, xs[-1] + tile_w)
    src = cv2.copyMakeBorder(bgr, 0, req_h - h, 0, req_w - w, cv2.BORDER_REFLECT_101)
    rgb_norm = (cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).astype(inp_dtype)

    out_h, out_w = h * scale, w * scale
    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    win_y = np.maximum(np.hanning(tile_h * scale), 0.05) if len(ys) > 1 else np.ones(tile_h * scale)
    win_x = np.maximum(np.hanning(tile_w * scale), 0.05) if len(xs) > 1 else np.ones(tile_w * scale)
    win2d = np.outer(win_y, win_x).astype(np.float32)[:, :, None]

    for y0 in ys:
        for x0 in xs:
            patch = rgb_norm[y0 : y0 + tile_h, x0 : x0 + tile_w]
            blob = np.ascontiguousarray(np.transpose(patch, (2, 0, 1))[None, ...])
            sr = sess.run(None, {inp.name: blob})[0][0]
            sr_hwc = np.transpose(sr.astype(np.float32), (1, 2, 0))
            sr_bgr = cv2.cvtColor(np.clip(sr_hwc, 0.0, 1.0), cv2.COLOR_RGB2BGR)

            oy, ox = y0 * scale, x0 * scale
            ey, ex = min(out_h, oy + tile_h * scale), min(out_w, ox + tile_w * scale)
            ph, pw = ey - oy, ex - ox
            acc[oy:ey, ox:ex] += sr_bgr[:ph, :pw] * win2d[:ph, :pw]
            wsum[oy:ey, ox:ex] += win2d[:ph, :pw]

    merged = acc / np.maximum(wsum, 1e-8)
    return np.clip(np.round(merged * 255.0), 0, 255).astype(np.uint8), device_label


def op_ai_upscale(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """AI Super-Resolution (2x / 3x / 4x) via ESPCN or 4x-UltraSharp ONNX."""
    img = ensure_uint8(img)
    model_choice = str(params.get("model", "ESPCN 3× (Fast)"))
    factor_str = str(params.get("scale", "3×"))
    target_scale = 2 if factor_str.startswith("2×") else (3 if factor_str.startswith("3×") else 4)
    sharpen = float(params.get("crispness", 0.15))

    if model_choice.startswith("4x-UltraSharp"):
        bgr = to_bgr(img)
        sr, device_label = _run_ultrasharp_4x_bgr(bgr)
        if target_scale != 4:
            sr = cv2.resize(sr, (bgr.shape[1] * target_scale, bgr.shape[0] * target_scale), interpolation=cv2.INTER_AREA)
        out = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY) if img.ndim == 2 else sr
        model_tag = "4x-UltraSharp"
    else:
        device_label = "CPU"
        if img.ndim == 2:
            sr_y = _run_espcn_3x_y(img)
            if target_scale != 3:
                sr_y = cv2.resize(sr_y, (img.shape[1] * target_scale, img.shape[0] * target_scale), interpolation=cv2.INTER_CUBIC)
            out = sr_y
        else:
            bgr = to_bgr(img)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            sr_y = _run_espcn_3x_y(ycrcb[:, :, 0])
            h3, w3 = sr_y.shape
            cr = cv2.resize(ycrcb[:, :, 1], (w3, h3), interpolation=cv2.INTER_CUBIC)
            cb = cv2.resize(ycrcb[:, :, 2], (w3, h3), interpolation=cv2.INTER_CUBIC)
            out = cv2.cvtColor(cv2.merge([sr_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
            if target_scale != 3:
                out = cv2.resize(out, (bgr.shape[1] * target_scale, bgr.shape[0] * target_scale), interpolation=cv2.INTER_CUBIC)
        model_tag = "ESPCN-ONNX"

    if sharpen > 0.0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = np.clip(np.round(out.astype(np.float32) + sharpen * (out.astype(np.float32) - blur.astype(np.float32))), 0, 255).astype(np.uint8)

    return out, {"model": model_tag, "device": device_label, "output": f"{out.shape[1]}×{out.shape[0]}"}


def op_ai_bg_remove(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Background removal with true alpha transparency using RMBG-1.4, MODNet, or U2-Net-p."""
    bgr = to_bgr(img)
    model_choice = str(params.get("model", "RMBG-1.4 (Studio Quality)"))
    matte = _predict_matte(bgr, model_choice)
    thresh = float(params.get("threshold", 0.45))
    feather = int(params.get("feather", 3))
    bg_mode = str(params.get("background", "Transparent (Alpha)"))

    soft = np.clip((matte - (thresh - 0.15)) / 0.30, 0.0, 1.0)
    if feather >= 3:
        k = feather | 1
        soft = cv2.GaussianBlur(soft, (k, k), 0)

    fg_pct = round(float((soft > 0.5).mean() * 100.0), 1)
    model_file, model_tag = _matte_model_meta(model_choice)
    device_label = session_device_label(_load_ort_session(model_file))

    if bg_mode == "Transparent (Alpha)":
        alpha_u8 = np.clip(np.round(soft * 255.0), 0, 255).astype(np.uint8)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha_u8
        return bgra, {"model": model_tag, "device": device_label, "foreground": f"{fg_pct}%", "channels": "RGBA"}

    if bg_mode == "Alpha Matte (Grayscale)":
        return np.clip(np.round(soft * 255.0), 0, 255).astype(np.uint8), {"model": model_tag, "device": device_label}

    bg_colors = {
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Dark Studio": (24, 27, 34),
        "Chroma Green": (60, 210, 60),
    }
    col = bg_colors.get(bg_mode, (255, 255, 255))
    bg = np.full_like(bgr, col, dtype=np.float32)
    alpha = soft[:, :, None]
    out = bgr.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {"model": model_tag, "device": device_label, "foreground": f"{fg_pct}%"}


def op_ai_bokeh(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """AI Portrait Bokeh: blur background optically while keeping subject sharp."""
    bgr = to_bgr(img)
    model_choice = str(params.get("model", "RMBG-1.4 (Studio Quality)"))
    matte = _predict_matte(bgr, model_choice)
    blur_k = max(5, int(params.get("blur_strength", 25)) | 1)
    edge_soft = max(3, int(params.get("edge_feather", 7)) | 1)
    thresh = float(params.get("threshold", 0.45))

    alpha = np.clip((matte - (thresh - 0.12)) / 0.24, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (edge_soft, edge_soft), 0)[:, :, None]
    bg_blurred = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0)
    out = bgr.astype(np.float32) * alpha + bg_blurred.astype(np.float32) * (1.0 - alpha)
    model_tag = "RMBG-1.4" if model_choice.startswith("RMBG-1.4") else ("MODNet" if model_choice.startswith("MODNet") else "U2NetP")
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {"model": model_tag, "blur": f"{blur_k}×{blur_k}"}


def _detect_faces(bgr: np.ndarray, conf: float) -> list[np.ndarray]:
    det = _load_yunet(round(conf, 2), 0.3)
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    return [row for row in faces]


def op_ai_face_detect(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect faces and 5 facial landmarks using YuNet ONNX."""
    bgr = to_bgr(img).copy()
    conf = float(params.get("confidence", 0.60))
    show_landmarks = bool(params.get("landmarks", True))
    faces = _detect_faces(bgr, conf)

    for f in faces:
        x, y, w, h = int(round(f[0])), int(round(f[1])), int(round(f[2])), int(round(f[3]))
        score = float(f[14])
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (245, 139, 76), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"{score * 100:.0f}%",
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 139, 76),
            1,
            cv2.LINE_AA,
        )
        if show_landmarks:
            for idx in range(4, 14, 2):
                lx, ly = int(round(f[idx])), int(round(f[idx + 1]))
                cv2.circle(bgr, (lx, ly), 3, (120, 225, 160), -1, lineType=cv2.LINE_AA)

    return bgr, {"model": "YuNet-2023", "faces_found": len(faces)}


def op_ai_face_privacy(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Automatically anonymize detected faces via Gaussian blur, Mosaic pixelation, or Solid mask."""
    bgr = to_bgr(img).copy()
    h_img, w_img = bgr.shape[:2]
    conf = float(params.get("confidence", 0.55))
    mode = str(params.get("mode", "Mosaic Pixelate"))
    strength = max(4, int(params.get("strength", 18)))
    pad_pct = float(params.get("padding_pct", 10.0)) / 100.0
    faces = _detect_faces(bgr, conf)

    for f in faces:
        x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        px, py = w * pad_pct, h * pad_pct
        x0 = max(0, int(round(x - px)))
        y0 = max(0, int(round(y - py)))
        x1 = min(w_img, int(round(x + w + px)))
        y1 = min(h_img, int(round(y + h + py)))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = bgr[y0:y1, x0:x1]
        bgr[y0:y1, x0:x1] = _apply_privacy_mask(roi, mode, strength)

    return bgr, {"model": "YuNet-2023", "anonymized": len(faces)}


def op_ai_style_transfer(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fast Neural Style Transfer using ONNX Model Zoo style networks (Mosaic / Candy)."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    style_name = str(params.get("style", "Mosaic"))
    mix = float(params.get("strength", 0.85))
    keep_color = bool(params.get("preserve_color", False))

    model_file = "candy-9.onnx" if style_name == "Candy" else "mosaic-9.onnx"
    net = _load_cv_onnx(model_file)
    blob = cv2.dnn.blobFromImage(bgr, 1.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward()[0]
    rgb_out = np.transpose(raw, (1, 2, 0))
    rgb_out = np.clip(np.round(rgb_out), 0, 255).astype(np.uint8)
    bgr_styled = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    bgr_styled = cv2.resize(bgr_styled, (w, h), interpolation=cv2.INTER_CUBIC)

    if keep_color:
        s_ycrcb = cv2.cvtColor(bgr_styled, cv2.COLOR_BGR2YCrCb)
        o_ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        o_ycrcb[:, :, 0] = s_ycrcb[:, :, 0]
        bgr_styled = cv2.cvtColor(o_ycrcb, cv2.COLOR_YCrCb2BGR)

    blended = bgr.astype(np.float32) * (1.0 - mix) + bgr_styled.astype(np.float32) * mix
    return np.clip(np.round(blended), 0, 255).astype(np.uint8), {"style": style_name, "mix": f"{mix * 100:.0f}%"}


def op_ai_smart_enhance(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-Scale Retinex perceptual illumination + local contrast auto-enhancement."""
    bgr = to_bgr(img).astype(np.float64) + 1.0
    strength = float(params.get("strength", 0.65))
    sigmas = (15.0, 60.0, 140.0)

    log_bgr = np.log(bgr)
    msr = np.zeros_like(log_bgr)
    for s in sigmas:
        blur = cv2.GaussianBlur(bgr, (0, 0), s)
        msr += log_bgr - np.log(np.maximum(blur, 1.0))
    msr /= len(sigmas)

    sum_rgb = np.maximum(np.sum(bgr, axis=2, keepdims=True), 1.0)
    crf = np.log(125.0 * bgr / sum_rgb + 1.0)
    msrcr = msr * crf

    for c in range(3):
        ch = msrcr[:, :, c]
        lo, hi = np.percentile(ch, (1.5, 98.5))
        if hi > lo + 1e-6:
            msrcr[:, :, c] = np.clip((ch - lo) * (255.0 / (hi - lo)), 0, 255)
        else:
            msrcr[:, :, c] = bgr[:, :, c] - 1.0

    orig_f = bgr - 1.0
    blended = orig_f * (1.0 - strength) + msrcr * strength
    return np.clip(np.round(blended), 0, 255).astype(np.uint8), {"engine": "MSRCR", "mix": f"{strength * 100:.0f}%"}


_COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _predict_depth_map(bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Estimate normalized [0, 1] monocular relative depth using Depth Anything V2 Small (quantized)."""
    h, w = bgr.shape[:2]
    sz = 392  # multiple of 14 for ViT patch grid
    resized = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = np.ascontiguousarray(np.transpose((rgb - mean) / std, (2, 0, 1))[None, ...]).astype(np.float32)

    sess = _load_ort_session("depth-anything-v2-small-q.onnx")
    inp_name = sess.get_inputs()[0].name
    raw = sess.run(None, {inp_name: blob})[0][0].astype(np.float32)
    mn, mx = float(raw.min()), float(raw.max())
    norm = (raw - mn) / (mx - mn) if mx > mn + 1e-6 else np.zeros_like(raw)
    depth = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(depth, 0.0, 1.0).astype(np.float32), session_device_label(sess)


def op_ai_depth_map(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """AI Monocular 3D Depth Estimation using Depth Anything V2 Small ONNX."""
    bgr = to_bgr(img)
    depth, device_label = _predict_depth_map(bgr)
    if bool(params.get("invert", False)):
        depth = 1.0 - depth
    depth_u8 = np.clip(np.round(depth * 255.0), 0, 255).astype(np.uint8)

    cmap_choice = str(params.get("colormap", "Turbo"))
    cmaps = {
        "Turbo": cv2.COLORMAP_TURBO,
        "Inferno": cv2.COLORMAP_INFERNO,
        "Magma": cv2.COLORMAP_MAGMA,
        "Viridis": cv2.COLORMAP_VIRIDIS,
        "Plasma": cv2.COLORMAP_PLASMA,
    }
    if cmap_choice == "Grayscale":
        out = depth_u8
    else:
        out = cv2.applyColorMap(depth_u8, cmaps.get(cmap_choice, cv2.COLORMAP_TURBO))
    return out, {"model": "DepthAnythingV2-S", "device": device_label, "map": cmap_choice}


def op_ai_depth_focus(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Refocus image optically around a selected 3D depth plane."""
    bgr = to_bgr(img)
    depth, device_label = _predict_depth_map(bgr)
    focus_depth = float(params.get("focus_depth", 0.75))
    dof_width = float(params.get("dof_width", 0.22))
    blur_k = max(5, int(params.get("blur_strength", 25)) | 1)

    dist = np.abs(depth - focus_depth)
    blur_weight = np.clip((dist - dof_width * 0.5) / max(0.08, 1.0 - dof_width), 0.0, 1.0)
    blur_mid = cv2.GaussianBlur(bgr, (max(3, (blur_k // 2) | 1), max(3, (blur_k // 2) | 1)), 0).astype(np.float32)
    blur_far = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0).astype(np.float32)
    orig_f = bgr.astype(np.float32)

    w_far = np.clip((blur_weight - 0.35) / 0.65, 0.0, 1.0)[:, :, None]
    w_mid = np.clip(blur_weight / 0.55, 0.0, 1.0)[:, :, None]
    soft_bg = blur_mid * (1.0 - w_far) + blur_far * w_far
    out = orig_f * (1.0 - w_mid) + soft_bg * w_mid
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {
        "model": "DepthAnythingV2-S",
        "device": device_label,
        "focus": f"{focus_depth * 100:.0f}%",
    }


def op_ai_anime_style(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform photos into anime/illustration art using AnimeGANv2 Hayao ONNX."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    max_side = 768
    scale = min(1.0, max_side / max(h, w))
    nw = max(128, int(round((w * scale) / 32.0)) * 32)
    nh = max(128, int(round((h * scale) / 32.0)) * 32)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    blob = np.ascontiguousarray(rgb[None, ...]).astype(np.float32)  # NHWC

    sess = _load_ort_session("AnimeGANv2_Hayao.onnx")
    inp_name = sess.get_inputs()[0].name
    styled = sess.run(None, {inp_name: blob})[0][0]
    styled_u8 = np.clip(np.round((styled + 1.0) * 127.5), 0, 255).astype(np.uint8)
    styled_bgr = cv2.cvtColor(styled_u8, cv2.COLOR_RGB2BGR)
    if styled_bgr.shape[:2] != (h, w):
        styled_bgr = cv2.resize(styled_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    if bool(params.get("preserve_color", False)):
        orig_ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        sty_ycrcb = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2YCrCb)
        sty_ycrcb[:, :, 1:] = orig_ycrcb[:, :, 1:]
        styled_bgr = cv2.cvtColor(sty_ycrcb, cv2.COLOR_YCrCb2BGR)

    strength = float(params.get("strength", 0.90))
    out = cv2.addWeighted(bgr, 1.0 - strength, styled_bgr, strength, 0.0)
    return out, {"model": "AnimeGANv2-Hayao", "device": session_device_label(sess), "mix": f"{strength * 100:.0f}%"}


def _detect_yolo_objects(
    bgr: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
) -> tuple[list[tuple[int, int, int, int, float, int]], str]:
    """Run YOLOv8n ONNX detection and return NMS-filtered (x, y, w, h, conf, cls_id) tuples."""
    h, w = bgr.shape[:2]
    sz = 640
    r = min(sz / w, sz / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    pad_x = (sz - nw) * 0.5
    pad_y = (sz - nh) * 0.5
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
    y0, x0 = int(round(pad_y)), int(round(pad_x))
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...]).astype(np.float32)

    sess = _load_ort_session("yolov8n.onnx")
    inp_name = sess.get_inputs()[0].name
    preds = sess.run(None, {inp_name: blob})[0][0].T  # (8400, 84)
    device_label = session_device_label(sess)

    class_scores = preds[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(cls_ids)), cls_ids]
    keep = confs >= conf_thresh

    boxes: list[list[int]] = []
    kept_scores: list[float] = []
    kept_cls: list[int] = []
    for row, sc, cid in zip(preds[keep, :4], confs[keep], cls_ids[keep]):
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        bx = int(round((cx - bw * 0.5 - pad_x) / r))
        by = int(round((cy - bh * 0.5 - pad_y) / r))
        b_w = int(round(bw / r))
        b_h = int(round(bh / r))
        bx1 = max(0, bx)
        by1 = max(0, by)
        bx2 = min(w, bx + b_w)
        by2 = min(h, by + b_h)
        if bx2 > bx1 + 2 and by2 > by1 + 2:
            boxes.append([bx1, by1, bx2 - bx1, by2 - by1])
            kept_scores.append(float(sc))
            kept_cls.append(int(cid))

    detections: list[tuple[int, int, int, int, float, int]] = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, kept_scores, conf_thresh, iou_thresh)
        flat_idx = indices.flatten().tolist() if len(indices) > 0 else []
        for idx in flat_idx:
            bx, by, bw, bh = boxes[idx]
            detections.append((bx, by, bw, bh, kept_scores[idx], kept_cls[idx]))
    return detections, device_label


def op_ai_object_detect(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect 80 common object categories using YOLOv8 Nano ONNX."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    conf_thresh = float(params.get("confidence", 0.35))
    iou_thresh = float(params.get("iou_thresh", 0.45))
    display_mode = str(params.get("display_mode", "Boxes & Labels"))

    detections, device_label = _detect_yolo_objects(bgr, conf_thresh, iou_thresh)
    out = bgr.copy()
    detected_labels: list[str] = []
    if detections:
        if display_mode == "Dim Background":
            mask = np.zeros((h, w), dtype=np.uint8)
            for bx, by, bw, bh, _sc, _cid in detections:
                mask[by : by + bh, bx : bx + bw] = 1
            dimmed = (bgr.astype(np.float32) * 0.28).astype(np.uint8)
            out = np.where(mask[:, :, None] > 0, bgr, dimmed)

        palette = (
            (245, 139, 76),
            (125, 175, 76),
            (94, 168, 255),
            (180, 115, 240),
            (79, 83, 217),
        )
        for bx, by, bw, bh, sc, cid in detections:
            label = _COCO_CLASSES[cid] if cid < len(_COCO_CLASSES) else f"cls{cid}"
            detected_labels.append(f"{label} {sc:.0%}")
            if display_mode == "Highlight Objects":
                overlay = out.copy()
                col = palette[cid % len(palette)]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), col, -1)
                cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0, out)
            col = palette[cid % len(palette)]
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, 2)
            txt = f"{label} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = max(th + 6, by)
            cv2.rectangle(out, (bx, ty - th - 6), (bx + tw + 8, ty), col, -1)
            cv2.putText(out, txt, (bx + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return out, {
        "model": "YOLOv8n",
        "device": device_label,
        "objects": len(detected_labels),
        "classes": ", ".join(detected_labels[:4]) if detected_labels else "none",
    }


def op_ai_low_light(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Illumination-map low-light enhancement with adaptive shadow lift and noise control."""
    bgr = to_bgr(img)
    lift = float(params.get("lift", 0.65))
    denoise = float(params.get("denoise", 0.35))

    bgr_f = bgr.astype(np.float32) / 255.0
    illum_raw = np.max(bgr_f, axis=2)
    illum = cv2.GaussianBlur(illum_raw, (0, 0), 18.0)
    illum = np.clip(0.35 * illum_raw + 0.65 * illum, 0.04, 1.0)

    gamma = max(0.22, 1.0 - 0.68 * lift)
    lifted = np.power(illum, gamma)
    gain = np.clip(lifted / illum, 1.0, 4.5)[:, :, None]
    enhanced = np.clip(bgr_f * gain, 0.0, 1.0)
    out = np.clip(np.round(enhanced * 255.0), 0, 255).astype(np.uint8)

    if denoise > 0.05:
        sigma_color = 12.0 + 38.0 * denoise
        smooth = cv2.bilateralFilter(out, d=5, sigmaColor=sigma_color, sigmaSpace=7.0)
        mix = 0.45 * denoise
        out = cv2.addWeighted(out, 1.0 - mix, smooth, mix, 0.0)
    return out, {"engine": "LIME-Illum", "gamma": round(gamma, 2), "lift": f"{lift * 100:.0f}%"}


def op_ai_color_splash(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep segmented AI subject in vivid color while styling background monochrome or tinted."""
    bgr = to_bgr(img)
    model_choice = str(params.get("model", "RMBG-1.4 (Studio Quality)"))
    bg_style = str(params.get("bg_style", "Monochrome B&W"))
    bg_dim = float(params.get("bg_dim", 0.90))
    fg_sat = float(params.get("fg_saturation", 1.15))
    thresh = float(params.get("threshold", 0.45))

    matte = _predict_matte(bgr, model_choice)
    alpha = np.clip((matte - (thresh - 0.12)) / 0.24, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)[:, :, None]

    fg = bgr
    if abs(fg_sat - 1.0) > 1e-3:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * fg_sat, 0, 255)
        fg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if bg_style == "Muted Desaturated":
        hsv_bg = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bg[:, :, 1] *= 0.25
        bg_target = cv2.cvtColor(hsv_bg.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    elif bg_style == "Cool Slate Tint":
        bg_target = np.clip(gray_bgr * np.array([1.14, 1.00, 0.84], dtype=np.float32), 0, 255)
    elif bg_style == "Warm Sepia Tint":
        bg_target = np.clip(gray_bgr * np.array([0.78, 0.96, 1.14], dtype=np.float32), 0, 255)
    else:
        bg_target = gray_bgr

    bg_final = bgr.astype(np.float32) * (1.0 - bg_dim) + bg_target * bg_dim
    out = fg.astype(np.float32) * alpha + bg_final * (1.0 - alpha)
    _, model_tag = _matte_model_meta(model_choice)
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {"model": model_tag, "bg_style": bg_style}


def op_ai_sticker_cutout(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a die-cut sticker cutout with configurable stroke outline, shadow, and transparency."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    model_choice = str(params.get("model", "RMBG-1.4 (Studio Quality)"))
    outline_px = int(params.get("outline_px", 12))
    outline_color_name = str(params.get("outline_color", "Crisp White"))
    drop_shadow = bool(params.get("shadow", True))
    bg_mode = str(params.get("background", "Transparent (Alpha)"))
    thresh = float(params.get("threshold", 0.45))

    matte = _predict_matte(bgr, model_choice)
    fg_alpha = np.clip((matte - (thresh - 0.12)) / 0.24, 0.0, 1.0)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (3, 3), 0)

    if outline_px > 0:
        binary = (fg_alpha >= 0.5).astype(np.uint8) * 255
        k = outline_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(binary, kernel).astype(np.float32) / 255.0
        outer_alpha = cv2.GaussianBlur(dilated, (5, 5), 0)
    else:
        outer_alpha = fg_alpha.copy()

    outline_colors = {
        "Crisp White": (250, 250, 252),
        "Studio Cream": (228, 240, 248),
        "Electric Cyan": (255, 200, 64),
        "Coral Accent": (98, 122, 245),
        "Charcoal Dark": (34, 36, 42),
    }
    outline_col = np.array(outline_colors.get(outline_color_name, (250, 250, 252)), dtype=np.float32)

    sticker_rgb = bgr.astype(np.float32) * fg_alpha[:, :, None] + outline_col * (1.0 - fg_alpha[:, :, None])
    sticker_a = np.maximum(fg_alpha, outer_alpha)

    if drop_shadow:
        dx = max(4, outline_px // 2 + 4)
        dy = max(6, outline_px // 2 + 6)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(sticker_a, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        shadow_a = cv2.GaussianBlur(shifted, (21, 21), 0) * 0.45
        comb_a = sticker_a + shadow_a * (1.0 - sticker_a)
        safe_a = np.maximum(comb_a, 1e-4)
        comb_rgb = (sticker_rgb * sticker_a[:, :, None]) / safe_a[:, :, None]
        sticker_rgb, sticker_a = np.clip(comb_rgb, 0, 255), comb_a

    _, model_tag = _matte_model_meta(model_choice)
    if bg_mode == "Transparent (Alpha)":
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[:, :, :3] = np.clip(np.round(sticker_rgb), 0, 255).astype(np.uint8)
        bgra[:, :, 3] = np.clip(np.round(sticker_a * 255.0), 0, 255).astype(np.uint8)
        return bgra, {"model": model_tag, "outline": f"{outline_px}px", "channels": "RGBA"}

    if bg_mode == "Pastel Grid":
        yy, xx = np.indices((h, w))
        grid = np.where((xx % 32 < 2) | (yy % 32 < 2), 214, 236).astype(np.float32)
        bg = np.stack([grid + 8, grid + 2, grid - 2], axis=2)
    elif bg_mode == "White":
        bg = np.full((h, w, 3), 255.0, dtype=np.float32)
    else:
        bg = np.full((h, w, 3), (30, 24, 22), dtype=np.float32)

    out = sticker_rgb * sticker_a[:, :, None] + bg * (1.0 - sticker_a[:, :, None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {"model": model_tag, "outline": f"{outline_px}px"}


def op_ai_depth_fog(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Add volumetric 3D atmospheric fog/haze using Depth Anything V2 scene geometry."""
    bgr = to_bgr(img)
    depth, device_label = _predict_depth_map(bgr)
    density = float(params.get("density", 0.72))
    fog_reach = float(params.get("fog_reach", 0.78))
    tone = str(params.get("tone", "Morning Mist"))

    fog_colors = {
        "Morning Mist": (235, 228, 222),
        "Warm Golden Haze": (182, 214, 242),
        "Cool Nordic Fog": (238, 218, 198),
        "Midnight Blue": (78, 48, 32),
        "Dramatic Dark": (32, 34, 40),
    }
    fog_col = np.array(fog_colors.get(tone, (235, 228, 222)), dtype=np.float32)

    fog_mask = np.clip((fog_reach - depth) / max(0.12, fog_reach), 0.0, 1.0)
    fog_weight = (np.power(fog_mask, 1.15) * density)[:, :, None]
    out = bgr.astype(np.float32) * (1.0 - fog_weight) + fog_col * fog_weight
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {
        "model": "DepthAnythingV2-S",
        "device": device_label,
        "tone": tone,
    }


def op_ai_depth_relight(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate 3D surface normals from monocular depth and apply directional key relighting."""
    bgr = to_bgr(img)
    depth, device_label = _predict_depth_map(bgr)
    az_deg = float(params.get("angle_deg", -45.0))
    el_deg = float(params.get("elevation_deg", 35.0))
    intensity = float(params.get("intensity", 1.15))
    ambient = float(params.get("ambient", 0.55))
    light_tone = str(params.get("light_tone", "Studio Neutral"))

    depth_s = cv2.GaussianBlur(depth, (35, 35), 0)
    dx = cv2.Sobel(depth_s, cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(depth_s, cv2.CV_32F, 0, 1, ksize=5)
    relief = 2.6
    nx = -dx * relief
    ny = -dy * relief
    nz = np.ones_like(depth_s, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= norm
    ny /= norm
    nz /= norm

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    lx = float(np.cos(el) * np.cos(az))
    ly = float(np.cos(el) * np.sin(az))
    lz = float(np.sin(el))

    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    depth_falloff = 0.45 + 0.55 * depth_s
    tone_gains = {
        "Studio Neutral": (1.00, 1.00, 1.00),
        "Warm Tungsten": (0.82, 0.96, 1.14),
        "Golden Hour": (0.76, 0.94, 1.18),
        "Cool Moonlight": (1.16, 1.00, 0.84),
    }
    tint = np.array(tone_gains.get(light_tone, (1.0, 1.0, 1.0)), dtype=np.float32)
    gain = (ambient + intensity * ndotl * depth_falloff)[:, :, None] * tint
    out = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out, {
        "model": "DepthAnythingV2-S",
        "device": device_label,
        "light": f"{int(round(az_deg))}°",
    }


def op_ai_face_auto_crop(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Smart portrait cropper driven by YuNet face detection and target aspect framing."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    conf = float(params.get("confidence", 0.50))
    aspect_choice = str(params.get("aspect", "1:1 (Avatar)"))
    framing = str(params.get("framing", "Close-up Face"))

    aspect_map = {
        "1:1 (Avatar)": 1.0,
        "4:5 (Portrait)": 0.8,
        "3:4 (Editorial)": 0.75,
        "16:9 (Cinema)": 16.0 / 9.0,
        "Original Aspect": w / max(1.0, float(h)),
    }
    target_ratio = aspect_map.get(aspect_choice, 1.0)
    faces = _detect_faces(bgr, conf)

    if not faces:
        if w / max(1, h) >= target_ratio:
            ch = h
            cw = max(16, int(round(ch * target_ratio)))
        else:
            cw = w
            ch = max(16, int(round(cw / target_ratio)))
        x0 = max(0, (w - cw) // 2)
        y0 = max(0, (h - ch) // 2)
        cropped = bgr[y0 : y0 + ch, x0 : x0 + cw]
        return cropped, {"model": "YuNet-2023", "faces_found": 0, "fallback": "center_crop"}

    primary = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    fx, fy, fw, fh = float(primary[0]), float(primary[1]), float(primary[2]), float(primary[3])
    cx = fx + fw * 0.5
    cy = fy + fh * 0.44

    scale_mult = {"Close-up Face": 1.9, "Head & Shoulders": 2.9, "Upper Body": 4.2}.get(framing, 2.9)
    ch = min(float(h), max(32.0, fh * scale_mult))
    cw = ch * target_ratio
    if cw > w:
        cw = float(w)
        ch = max(32.0, cw / target_ratio)

    x0 = int(round(np.clip(cx - cw * 0.5, 0.0, max(0.0, w - cw))))
    y0 = int(round(np.clip(cy - ch * 0.42, 0.0, max(0.0, h - ch))))
    x1 = min(w, x0 + max(16, int(round(cw))))
    y1 = min(h, y0 + max(16, int(round(ch))))
    cropped = bgr[y0:y1, x0:x1]
    return cropped, {
        "model": "YuNet-2023",
        "faces_found": len(faces),
        "crop": f"{cropped.shape[1]}×{cropped.shape[0]}",
    }


def op_ai_face_enhance(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine detected faces with edge-preserving skin smoothing, local illumination lift, and eye crispness."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    conf = float(params.get("confidence", 0.50))
    smoothness = float(params.get("smoothness", 0.55))
    eye_clarity = float(params.get("eye_clarity", 0.45))
    face_light = float(params.get("face_light", 0.15))

    faces = _detect_faces(bgr, conf)
    if not faces:
        return bgr.copy(), {"model": "YuNet-2023", "faces_enhanced": 0}

    smooth_full = cv2.bilateralFilter(bgr, d=7, sigmaColor=20.0 + 45.0 * smoothness, sigmaSpace=9.0).astype(np.float32)
    detail_blur = cv2.GaussianBlur(bgr, (0, 0), 1.6)
    crisp_full = np.clip(bgr.astype(np.float32) + eye_clarity * 1.15 * (bgr.astype(np.float32) - detail_blur.astype(np.float32)), 0, 255)

    skin_mask = np.zeros((h, w), dtype=np.float32)
    eye_mask = np.zeros((h, w), dtype=np.float32)
    for f in faces:
        fx, fy, fw, fh = int(round(f[0])), int(round(f[1])), int(round(f[2])), int(round(f[3]))
        if fw < 8 or fh < 8:
            continue
        cx, cy = fx + fw // 2, fy + fh // 2
        cv2.ellipse(skin_mask, (cx, cy), (max(4, int(fw * 0.46)), max(4, int(fh * 0.52))), 0, 0, 360, 1.0, -1)
        for idx in (4, 6, 10, 12):  # left eye, right eye, mouth corners
            lx, ly = int(round(f[idx])), int(round(f[idx + 1]))
            if 0 <= lx < w and 0 <= ly < h:
                cv2.circle(eye_mask, (lx, ly), max(4, int(min(fw, fh) * 0.13)), 1.0, -1)

    skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
    eye_mask = cv2.GaussianBlur(eye_mask, (11, 11), 0)
    pure_skin = np.clip(skin_mask - eye_mask * 0.85, 0.0, 1.0)[:, :, None] * (0.75 * smoothness)
    eye_w = eye_mask[:, :, None]

    out = bgr.astype(np.float32) * (1.0 - pure_skin) + smooth_full * pure_skin
    out = out * (1.0 - eye_w) + crisp_full * eye_w
    if face_light > 1e-3:
        lift = 1.0 + face_light * 0.45 * skin_mask[:, :, None]
        out = np.clip(out * lift, 0, 255)
    return np.clip(np.round(out), 0, 255).astype(np.uint8), {
        "model": "YuNet-2023",
        "faces_enhanced": len(faces),
    }


def op_ai_object_privacy(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect and mask people, vehicles, screens, or all COCO objects with YOLOv8n for privacy."""
    bgr = to_bgr(img).copy()
    target = str(params.get("target", "People (person)"))
    mode = str(params.get("mode", "Mosaic Pixelate"))
    strength = max(4, int(params.get("strength", 18)))
    conf_thresh = float(params.get("confidence", 0.35))
    iou_thresh = float(params.get("iou_thresh", 0.45))

    target_sets = {
        "People (person)": {0},
        "Vehicles (car, truck, bus, bike)": {1, 2, 3, 5, 7},
        "Screens & Phones (tv, laptop, phone)": {62, 63, 67},
        "Books & Documents (book)": {73},
    }
    allowed = target_sets.get(target)
    detections, device_label = _detect_yolo_objects(bgr, conf_thresh, iou_thresh)
    redacted = 0
    for bx, by, bw, bh, _sc, cid in detections:
        if allowed is not None and cid not in allowed:
            continue
        roi = bgr[by : by + bh, bx : bx + bw]
        if roi.size > 0:
            bgr[by : by + bh, bx : bx + bw] = _apply_privacy_mask(roi, mode, strength)
            redacted += 1

    return bgr, {
        "model": "YOLOv8n",
        "device": device_label,
        "redacted": redacted,
        "target": target,
    }


def get_specs() -> list[OperationSpec]:
    """Return all AI Studio operation specifications."""
    return [
        OperationSpec(
            op_id="ai.upscale",
            name="AI Super-Resolution (2×-4×)",
            category=OperationCategory.AI,
            description="Neural super-resolution via ESPCN (Fast) or 4x-UltraSharp (High Quality) ONNX.",
            func=op_ai_upscale,
            keywords=("ai", "upscale", "super resolution", "ultrasharp", "espcn", "onnx"),
            params=(
                ParamSpec("model", "Model", ParamType.CHOICE, "ESPCN 3× (Fast)", choices=("ESPCN 3× (Fast)", "4x-UltraSharp (High Quality)")),
                ParamSpec("scale", "Target Scale", ParamType.CHOICE, "3×", choices=("2×", "3×", "4×")),
                ParamSpec("crispness", "Detail Crispness", ParamType.FLOAT, 0.15, 0.0, 1.0, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.bg_remove",
            name="AI Background Removal",
            category=OperationCategory.AI,
            description="Studio-grade neural foreground matting (RMBG-1.4, MODNet, U2-Net-p) with true alpha transparency.",
            func=op_ai_bg_remove,
            keywords=("ai", "background", "remove", "transparent", "alpha", "rmbg", "modnet", "u2net"),
            params=(
                ParamSpec("model", "Segmentation Model", ParamType.CHOICE, "RMBG-1.4 (Studio Quality)", choices=("RMBG-1.4 (Studio Quality)", "MODNet (Portrait)", "U²-Net-p (Fast)")),
                ParamSpec("background", "Background", ParamType.CHOICE, "Transparent (Alpha)", choices=("Transparent (Alpha)", "White", "Dark Studio", "Black", "Chroma Green", "Alpha Matte (Grayscale)")),
                ParamSpec("threshold", "Matte Threshold", ParamType.FLOAT, 0.45, 0.15, 0.85, 0.05, 2),
                ParamSpec("feather", "Edge Feather", ParamType.INT, 3, 0, 15, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.bokeh",
            name="AI Portrait Bokeh",
            category=OperationCategory.AI,
            description="Separate subject via RMBG-1.4 / MODNet matte and apply optical depth-of-field background blur.",
            func=op_ai_bokeh,
            keywords=("ai", "bokeh", "portrait", "blur", "depth", "rmbg", "modnet"),
            params=(
                ParamSpec("model", "Matting Model", ParamType.CHOICE, "RMBG-1.4 (Studio Quality)", choices=("RMBG-1.4 (Studio Quality)", "MODNet (Studio Portrait)", "U²-Net-p (Fast)")),
                ParamSpec("blur_strength", "Background Blur", ParamType.INT, 25, 5, 65, 2, odd_only=True),
                ParamSpec("threshold", "Subject Threshold", ParamType.FLOAT, 0.45, 0.15, 0.85, 0.05, 2),
                ParamSpec("edge_feather", "Edge Feather", ParamType.INT, 7, 3, 21, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="ai.color_splash",
            name="AI Color Splash (Subject Pop)",
            category=OperationCategory.AI,
            description="Keep the AI-segmented foreground in rich color while styling the background B&W or tinted.",
            func=op_ai_color_splash,
            keywords=("ai", "color splash", "selective color", "pop", "foreground", "rmbg", "modnet"),
            params=(
                ParamSpec("model", "Matting Model", ParamType.CHOICE, "RMBG-1.4 (Studio Quality)", choices=("RMBG-1.4 (Studio Quality)", "MODNet (Portrait)", "U²-Net-p (Fast)")),
                ParamSpec("bg_style", "Background Style", ParamType.CHOICE, "Monochrome B&W", choices=("Monochrome B&W", "Muted Desaturated", "Cool Slate Tint", "Warm Sepia Tint")),
                ParamSpec("bg_dim", "Background Effect Mix", ParamType.FLOAT, 0.90, 0.0, 1.0, 0.05, 2),
                ParamSpec("fg_saturation", "Subject Saturation", ParamType.FLOAT, 1.15, 0.50, 1.80, 0.05, 2),
                ParamSpec("threshold", "Subject Threshold", ParamType.FLOAT, 0.45, 0.15, 0.85, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.sticker_cutout",
            name="AI Sticker & Outline Cutout",
            category=OperationCategory.AI,
            description="Generate a die-cut sticker outline and drop shadow around the AI-segmented subject.",
            func=op_ai_sticker_cutout,
            keywords=("ai", "sticker", "outline", "cutout", "border", "alpha", "rmbg"),
            params=(
                ParamSpec("model", "Matting Model", ParamType.CHOICE, "RMBG-1.4 (Studio Quality)", choices=("RMBG-1.4 (Studio Quality)", "MODNet (Portrait)", "U²-Net-p (Fast)")),
                ParamSpec("outline_px", "Outline Thickness", ParamType.INT, 12, 0, 40, 2, unit="px"),
                ParamSpec("outline_color", "Outline Color", ParamType.CHOICE, "Crisp White", choices=("Crisp White", "Studio Cream", "Electric Cyan", "Coral Accent", "Charcoal Dark")),
                ParamSpec("shadow", "Drop Shadow", ParamType.BOOL, True),
                ParamSpec("background", "Background", ParamType.CHOICE, "Transparent (Alpha)", choices=("Transparent (Alpha)", "Dark Studio", "White", "Pastel Grid")),
                ParamSpec("threshold", "Subject Threshold", ParamType.FLOAT, 0.45, 0.15, 0.85, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.depth_map",
            name="AI 3D Depth Map",
            category=OperationCategory.AI,
            description="Estimate relative scene depth using Depth Anything V2 Small ONNX.",
            func=op_ai_depth_map,
            keywords=("ai", "depth", "3d", "map", "distance", "depth anything"),
            params=(
                ParamSpec("colormap", "Palette", ParamType.CHOICE, "Turbo", choices=("Turbo", "Inferno", "Magma", "Viridis", "Plasma", "Grayscale")),
                ParamSpec("invert", "Invert Near / Far", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="ai.depth_focus",
            name="AI 3D Depth Focus",
            category=OperationCategory.AI,
            description="Apply depth-aware optical blur around any focal distance estimated by Depth Anything V2.",
            func=op_ai_depth_focus,
            keywords=("ai", "depth", "focus", "dof", "lens blur", "portrait"),
            params=(
                ParamSpec("focus_depth", "Focal Depth (Far → Near)", ParamType.FLOAT, 0.75, 0.0, 1.0, 0.05, 2),
                ParamSpec("dof_width", "In-Focus Range", ParamType.FLOAT, 0.22, 0.05, 0.60, 0.03, 2),
                ParamSpec("blur_strength", "Blur Strength", ParamType.INT, 25, 5, 55, 2, odd_only=True),
            ),
        ),
        OperationSpec(
            op_id="ai.depth_fog",
            name="AI 3D Depth Fog & Atmosphere",
            category=OperationCategory.AI,
            description="Blend physically inspired atmospheric mist using Depth Anything V2 scene geometry.",
            func=op_ai_depth_fog,
            keywords=("ai", "depth", "fog", "haze", "mist", "atmosphere", "3d"),
            params=(
                ParamSpec("density", "Haze Density", ParamType.FLOAT, 0.72, 0.05, 1.0, 0.05, 2),
                ParamSpec("fog_reach", "Atmosphere Reach (Far → Near)", ParamType.FLOAT, 0.78, 0.15, 1.0, 0.05, 2),
                ParamSpec("tone", "Atmosphere Tone", ParamType.CHOICE, "Morning Mist", choices=("Morning Mist", "Warm Golden Haze", "Cool Nordic Fog", "Midnight Blue", "Dramatic Dark")),
            ),
        ),
        OperationSpec(
            op_id="ai.depth_relight",
            name="AI 3D Depth Relighting",
            category=OperationCategory.AI,
            description="Synthesize surface normals from 3D depth to apply controllable directional key lighting.",
            func=op_ai_depth_relight,
            keywords=("ai", "relight", "lighting", "3d", "normals", "depth"),
            params=(
                ParamSpec("angle_deg", "Light Azimuth", ParamType.FLOAT, -45.0, -180.0, 180.0, 5.0, 0, unit="°"),
                ParamSpec("elevation_deg", "Light Elevation", ParamType.FLOAT, 35.0, 5.0, 85.0, 5.0, 0, unit="°"),
                ParamSpec("intensity", "Key Light Strength", ParamType.FLOAT, 1.15, 0.0, 2.5, 0.05, 2),
                ParamSpec("ambient", "Ambient Fill", ParamType.FLOAT, 0.55, 0.10, 1.0, 0.05, 2),
                ParamSpec("light_tone", "Light Tone", ParamType.CHOICE, "Studio Neutral", choices=("Studio Neutral", "Warm Tungsten", "Golden Hour", "Cool Moonlight")),
            ),
        ),
        OperationSpec(
            op_id="ai.object_detect",
            name="AI Object Detection (YOLO)",
            category=OperationCategory.AI,
            description="Detect 80 common object categories with bounding boxes and confidence labels via YOLOv8n.",
            func=op_ai_object_detect,
            keywords=("ai", "object", "detect", "yolo", "bounding box", "coco"),
            params=(
                ParamSpec("confidence", "Confidence Threshold", ParamType.FLOAT, 0.35, 0.15, 0.90, 0.05, 2),
                ParamSpec("iou_thresh", "NMS IoU Threshold", ParamType.FLOAT, 0.45, 0.20, 0.80, 0.05, 2),
                ParamSpec("display_mode", "Overlay Style", ParamType.CHOICE, "Boxes & Labels", choices=("Boxes & Labels", "Highlight Objects", "Dim Background")),
            ),
        ),
        OperationSpec(
            op_id="ai.object_privacy",
            name="AI Object Privacy Redactor (YOLO)",
            category=OperationCategory.AI,
            description="Detect and pixelate, blur, or mask people, vehicles, screens, or books via YOLOv8n.",
            func=op_ai_object_privacy,
            keywords=("ai", "privacy", "redact", "yolo", "vehicle", "person", "screen", "blur"),
            params=(
                ParamSpec("target", "Target Category", ParamType.CHOICE, "People (person)", choices=("People (person)", "Vehicles (car, truck, bus, bike)", "Screens & Phones (tv, laptop, phone)", "Books & Documents (book)", "All Detected Objects")),
                ParamSpec("mode", "Mask Mode", ParamType.CHOICE, "Mosaic Pixelate", choices=("Mosaic Pixelate", "Gaussian Blur", "Solid Charcoal")),
                ParamSpec("strength", "Block / Blur Strength", ParamType.INT, 18, 4, 48, 2),
                ParamSpec("confidence", "Confidence Threshold", ParamType.FLOAT, 0.35, 0.15, 0.90, 0.05, 2),
                ParamSpec("iou_thresh", "NMS IoU Threshold", ParamType.FLOAT, 0.45, 0.20, 0.80, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.anime_style",
            name="AI Anime / Illustration",
            category=OperationCategory.AI,
            description="Stylize photos into hand-drawn AnimeGANv2 (Hayao) illustration artwork.",
            func=op_ai_anime_style,
            keywords=("ai", "anime", "cartoon", "illustration", "animegan", "ghibli"),
            params=(
                ParamSpec("strength", "Illustration Mix", ParamType.FLOAT, 0.90, 0.10, 1.0, 0.05, 2),
                ParamSpec("preserve_color", "Preserve Original Colors", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="ai.face_detect",
            name="AI Face & Landmark Detection",
            category=OperationCategory.AI,
            description="Detect faces and 5 facial keypoints in real time using YuNet ONNX.",
            func=op_ai_face_detect,
            keywords=("ai", "face", "detect", "yunet", "landmarks", "onnx"),
            params=(
                ParamSpec("confidence", "Confidence Threshold", ParamType.FLOAT, 0.60, 0.25, 0.95, 0.05, 2),
                ParamSpec("landmarks", "Draw 5 Facial Landmarks", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="ai.face_privacy",
            name="AI Face Privacy Anonymizer",
            category=OperationCategory.AI,
            description="Automatically detect and pixelate, blur, or mask faces for privacy.",
            func=op_ai_face_privacy,
            keywords=("ai", "privacy", "blur face", "pixelate", "anonymize", "yunet"),
            params=(
                ParamSpec("mode", "Mask Mode", ParamType.CHOICE, "Mosaic Pixelate", choices=("Mosaic Pixelate", "Gaussian Blur", "Solid Charcoal")),
                ParamSpec("strength", "Block / Blur Strength", ParamType.INT, 18, 4, 48, 2),
                ParamSpec("confidence", "Detection Confidence", ParamType.FLOAT, 0.55, 0.25, 0.95, 0.05, 2),
                ParamSpec("padding_pct", "Box Padding", ParamType.FLOAT, 10.0, 0.0, 40.0, 2.0, 0, unit="%"),
            ),
        ),
        OperationSpec(
            op_id="ai.face_auto_crop",
            name="AI Smart Portrait Auto-Crop",
            category=OperationCategory.AI,
            description="Automatically center and frame subjects around detected YuNet faces with preset aspect ratios.",
            func=op_ai_face_auto_crop,
            keywords=("ai", "face", "crop", "portrait", "avatar", "yunet", "auto crop"),
            params=(
                ParamSpec("aspect", "Target Aspect", ParamType.CHOICE, "1:1 (Avatar)", choices=("1:1 (Avatar)", "4:5 (Portrait)", "3:4 (Editorial)", "16:9 (Cinema)", "Original Aspect")),
                ParamSpec("framing", "Framing Scale", ParamType.CHOICE, "Close-up Face", choices=("Close-up Face", "Head & Shoulders", "Upper Body")),
                ParamSpec("confidence", "Detection Confidence", ParamType.FLOAT, 0.50, 0.20, 0.95, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.face_enhance",
            name="AI Portrait Retouch & Clarity",
            category=OperationCategory.AI,
            description="Apply targeted skin smoothing and eye/feature crispness using YuNet facial landmarks.",
            func=op_ai_face_enhance,
            keywords=("ai", "face", "retouch", "portrait", "skin", "clarity", "yunet"),
            params=(
                ParamSpec("smoothness", "Skin Smoothing", ParamType.FLOAT, 0.55, 0.0, 1.0, 0.05, 2),
                ParamSpec("eye_clarity", "Eye & Detail Clarity", ParamType.FLOAT, 0.45, 0.0, 1.0, 0.05, 2),
                ParamSpec("face_light", "Face Illumination Lift", ParamType.FLOAT, 0.15, 0.0, 0.60, 0.05, 2),
                ParamSpec("confidence", "Detection Confidence", ParamType.FLOAT, 0.50, 0.20, 0.95, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.style_transfer",
            name="AI Neural Style Transfer",
            category=OperationCategory.AI,
            description="Apply Fast Neural Style networks (Mosaic or Candy ONNX) with optional color lock.",
            func=op_ai_style_transfer,
            keywords=("ai", "style", "neural", "mosaic", "candy", "art"),
            params=(
                ParamSpec("style", "Style Model", ParamType.CHOICE, "Mosaic", choices=("Mosaic", "Candy")),
                ParamSpec("strength", "Style Mix", ParamType.FLOAT, 0.85, 0.1, 1.0, 0.05, 2),
                ParamSpec("preserve_color", "Preserve Original Colors", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="ai.low_light",
            name="AI Low-Light Night Enhance",
            category=OperationCategory.AI,
            description="Adaptive illumination-map shadow recovery with noise-controlled tone lifting.",
            func=op_ai_low_light,
            keywords=("ai", "night", "low light", "dark", "shadow", "exposure", "lime"),
            params=(
                ParamSpec("lift", "Shadow Recovery", ParamType.FLOAT, 0.65, 0.15, 1.0, 0.05, 2),
                ParamSpec("denoise", "Chroma Noise Control", ParamType.FLOAT, 0.35, 0.0, 1.0, 0.05, 2),
            ),
        ),
        OperationSpec(
            op_id="ai.smart_enhance",
            name="AI Smart Auto-Enhance",
            category=OperationCategory.AI,
            description="Perceptual Multi-Scale Retinex color and dynamic-range restoration.",
            func=op_ai_smart_enhance,
            keywords=("ai", "auto", "enhance", "retinex", "lighting", "smart"),
            params=(
                ParamSpec("strength", "Enhancement Mix", ParamType.FLOAT, 0.65, 0.1, 1.0, 0.05, 2),
            ),
        ),
    ]
