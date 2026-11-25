"""Thresholding & Mathematical Morphology (Chapters 4 & 5)."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType

_SE_SHAPES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross (+)": cv2.MORPH_CROSS,
}


def build_structuring_element(shape_name: str, w: int, h: int) -> np.ndarray:
    """Create a structuring element with sanitized positive odd/integer dimensions."""
    kw = max(1, int(w))
    kh = max(1, int(h))
    shape = _SE_SHAPES.get(shape_name, cv2.MORPH_RECT)
    return cv2.getStructuringElement(shape, (kw, kh))


def _maybe_binarize(img: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "Off (Gray/Color)":
        return img, {}
    gray = to_gray(img)
    if mode == "Otsu Auto-Binary":
        t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw, {"pre_otsu_t": round(float(t), 1)}
    if mode == "Otsu Inverted":
        t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bw, {"pre_otsu_t": round(float(t), 1)}
    # Manual T=128
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return bw, {"pre_threshold": 128}


def _se_from_params(params: dict[str, Any]) -> tuple[np.ndarray, str, int, int, int]:
    shape_name = str(params.get("se_shape", "Rectangle"))
    kw = max(1, int(params.get("se_w", 5)))
    kh = kw if bool(params.get("se_square", True)) else max(1, int(params.get("se_h", 5)))
    iters = max(1, int(params.get("iterations", 1)))
    se = build_structuring_element(shape_name, kw, kh)
    return se, shape_name, kw, kh, iters


def op_global_threshold(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed / Global thresholding with selectable OpenCV threshold type."""
    gray = to_gray(img)
    thresh = int(params.get("threshold", 128))
    mode_name = str(params.get("mode", "Binary"))
    flags = {
        "Binary": cv2.THRESH_BINARY,
        "Binary Inverted": cv2.THRESH_BINARY_INV,
        "Truncate": cv2.THRESH_TRUNC,
        "To Zero": cv2.THRESH_TOZERO,
        "To Zero Inverted": cv2.THRESH_TOZERO_INV,
    }.get(mode_name, cv2.THRESH_BINARY)
    t, out = cv2.threshold(gray, thresh, 255, flags)
    return out, {"threshold": float(t), "mode": mode_name}


def op_otsu_threshold(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Otsu optimal global thresholding minimizing intra-class variance."""
    gray = to_gray(img)
    pre_blur = int(params.get("pre_blur", 0))
    if pre_blur >= 3:
        k = pre_blur if pre_blur % 2 == 1 else pre_blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    invert = bool(params.get("invert", False))
    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    t, out = cv2.threshold(gray, 0, 255, flag)
    fg_pct = round(float(np.count_nonzero(out)) / max(out.size, 1) * 100.0, 1)
    return out, {"otsu_threshold": float(t), "foreground_pct": f"{fg_pct}%"}


def op_adaptive_threshold(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Local adaptive thresholding for uneven illumination (e.g., sudoku.png, rice.png)."""
    gray = to_gray(img)
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if params.get("method", "Gaussian") == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    block = max(3, int(params.get("block_size", 15)))
    if block % 2 == 0:
        block += 1
    c_val = float(params.get("c_offset", 5.0))
    invert = bool(params.get("invert", False))
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    out = cv2.adaptiveThreshold(gray, 255, method, ttype, block, c_val)
    return out, {"block_size": block, "C": c_val}


def _morph_op(img: np.ndarray, params: dict[str, Any], cv_op: int, op_label: str) -> tuple[np.ndarray, dict[str, Any]]:
    img = ensure_uint8(img)
    pre_mode = str(params.get("binarize", "Off (Gray/Color)"))
    src, meta = _maybe_binarize(img, pre_mode)
    se, shape_name, kw, kh, iters = _se_from_params(params)

    if cv_op == cv2.MORPH_ERODE:
        out = cv2.erode(src, se, iterations=iters)
    elif cv_op == cv2.MORPH_DILATE:
        out = cv2.dilate(src, se, iterations=iters)
    else:
        out = cv2.morphologyEx(src, cv_op, se, iterations=iters)

    meta.update({"op": op_label, "se": f"{shape_name} {kw}×{kh}", "iterations": iters})
    return out, meta


def op_erode(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _morph_op(img, params, cv2.MORPH_ERODE, "Erode")


def op_dilate(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _morph_op(img, params, cv2.MORPH_DILATE, "Dilate")


def op_open(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _morph_op(img, params, cv2.MORPH_OPEN, "Open (Erode → Dilate)")


def op_close(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _morph_op(img, params, cv2.MORPH_CLOSE, "Close (Dilate → Erode)")


def op_morph_gradient(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    return _morph_op(img, params, cv2.MORPH_GRADIENT, "Morphological Gradient")


def op_tophat(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-Hat (Original − Opening) isolates bright features smaller than SE (uneven illumination)."""
    return _morph_op(img, params, cv2.MORPH_TOPHAT, "Top-Hat (White)")


def op_blackhat(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Black-Hat (Closing − Original) isolates dark features smaller than SE."""
    return _morph_op(img, params, cv2.MORPH_BLACKHAT, "Black-Hat")


def op_boundary_extract(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Morphological boundary extraction β(A) = A − (A ⊖ B) (Chapter 4)."""
    img = ensure_uint8(img)
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    src, meta = _maybe_binarize(img, pre_mode)
    se, shape_name, kw, kh, iters = _se_from_params(params)
    eroded = cv2.erode(src, se, iterations=iters)
    out = cv2.subtract(src, eroded)
    meta.update({"formula": "A - (A ⊖ B)", "se": f"{shape_name} {kw}×{kh}"})
    return out, meta


def op_skeleton(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Iterative morphological thinning / skeletonization (Chapter 4)."""
    gray = to_gray(img)
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    if pre_mode == "Off (Gray/Color)":
        pre_mode = "Otsu Auto-Binary"
    bw, meta = _maybe_binarize(gray, pre_mode)
    se_shape = str(params.get("se_shape", "Cross (+)"))
    se = build_structuring_element(se_shape, 3, 3)

    skel = np.zeros_like(bw)
    working = bw.copy()
    max_iters = int(params.get("max_steps", 64))
    steps = 0
    while steps < max_iters:
        eroded = cv2.erode(working, se)
        opened = cv2.dilate(eroded, se)
        temp = cv2.subtract(working, opened)
        skel = cv2.bitwise_or(skel, temp)
        working = eroded
        steps += 1
        if cv2.countNonZero(working) == 0:
            break
    meta.update({"iterations": steps, "se": se_shape})
    return skel, meta


def op_multi_threshold(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-level thresholding (T1, T2) or iterative automatic global thresholding (Ch. 5)."""
    gray = to_gray(img)
    mode = str(params.get("mode", "Iterative Auto T*"))
    if mode == "3-Level (T1, T2)":
        t1 = int(params.get("t1", 85))
        t2 = int(params.get("t2", 170))
        if t1 > t2:
            t1, t2 = t2, t1
        out = np.zeros_like(gray)
        out[(gray > t1) & (gray <= t2)] = 128
        out[gray > t2] = 255
        return out, {"mode": "3-Level", "T1": t1, "T2": t2}

    t = float(np.mean(gray))
    iters = 0
    for iters in range(1, 64):
        g1 = gray[gray > t]
        g2 = gray[gray <= t]
        if g1.size == 0 or g2.size == 0:
            break
        t_new = 0.5 * (float(g1.mean()) + float(g2.mean()))
        if abs(t_new - t) < 0.5:
            t = t_new
            break
        t = t_new
    _, out = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return out, {"mode": "Iterative T*", "T*": round(t, 1), "iterations": iters}


def op_niblack_sauvola(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Niblack and Sauvola local mean/std adaptive document thresholding (Chapter 5)."""
    gray = to_gray(img).astype(np.float32)
    method = str(params.get("method", "Sauvola"))
    block = max(3, int(params.get("block_size", 25)) | 1)
    k = float(params.get("k_val", 0.25 if method == "Sauvola" else -0.20))
    r_dyn = max(16.0, float(params.get("sauvola_r", 128.0)))
    invert = bool(params.get("invert", False))

    mean = cv2.boxFilter(gray, cv2.CV_32F, (block, block), borderType=cv2.BORDER_REFLECT101)
    sq_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (block, block), borderType=cv2.BORDER_REFLECT101)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))

    if method == "Niblack":
        thresh = mean + k * std
    else:
        thresh = mean * (1.0 + k * (std / r_dyn - 1.0))

    fg = gray <= thresh if invert else gray > thresh
    out = np.where(fg, 255, 0).astype(np.uint8)
    return out, {"method": method, "window": f"{block}×{block}", "k": round(k, 2)}


def op_hole_fill(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill enclosed interior background holes via border-connected flood fill (Ch. 4)."""
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    if pre_mode == "Off (Gray/Color)":
        pre_mode = "Otsu Auto-Binary"
    bw, meta = _maybe_binarize(to_gray(img), pre_mode)
    padded = cv2.copyMakeBorder(bw, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    flood = padded.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_cropped = flood[1:-1, 1:-1]
    holes = cv2.bitwise_not(flood_cropped)
    out = cv2.bitwise_or(bw, holes)
    filled_px = int(np.count_nonzero(cv2.subtract(out, bw)))
    meta["filled_px"] = filled_px
    return out, meta


def op_convex_hull(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute morphological/geometric Convex Hull of foreground components (Ch. 4)."""
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    if pre_mode == "Off (Gray/Color)":
        pre_mode = "Otsu Auto-Binary"
    min_area = int(params.get("min_area", 40))
    display = str(params.get("display", "Hull Over Original"))
    bw, meta = _maybe_binarize(to_gray(img), pre_mode)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(c) for c in contours if cv2.contourArea(c) >= min_area]

    hull_mask = np.zeros_like(bw)
    cv2.drawContours(hull_mask, hulls, -1, 255, -1)
    meta["hulls"] = len(hulls)

    if display == "Filled Binary Hull":
        return hull_mask, meta
    if display == "Defects (Hull - Object)":
        return cv2.subtract(hull_mask, bw), meta

    vis = cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2BGR) if img.ndim == 2 else ensure_uint8(img)[:, :, :3].copy()
    cv2.drawContours(vis, hulls, -1, (94, 168, 255), 2, lineType=cv2.LINE_AA)
    return vis, meta


_HMT_KERNELS: dict[str, list[np.ndarray]] = {
    "All 4 Corners (Union)": [
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, -1]], dtype=np.int8),
        np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.int8),
        np.array([[0, 1, -1], [0, 1, 1], [0, 0, 0]], dtype=np.int8),
        np.array([[-1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int8),
    ],
    "Top-Left Corner": [np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 0]], dtype=np.int8)],
    "Isolated Pixels": [np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8)],
    "Line Endpoints": [
        np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]], dtype=np.int8),
        np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8),
        np.array([[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]], dtype=np.int8),
        np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]], dtype=np.int8),
    ],
}


def op_hit_or_miss(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Hit-or-Miss transform A ⊛ B for binary template and corner/endpoint detection (Ch. 4)."""
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    if pre_mode == "Off (Gray/Color)":
        pre_mode = "Otsu Auto-Binary"
    pattern = str(params.get("pattern", "All 4 Corners (Union)"))
    enlarge = bool(params.get("visible_markers", True))
    bw, meta = _maybe_binarize(to_gray(img), pre_mode)

    hits = np.zeros_like(bw)
    for k in _HMT_KERNELS.get(pattern, _HMT_KERNELS["All 4 Corners (Union)"]):
        hits = cv2.bitwise_or(hits, cv2.morphologyEx(bw, cv2.MORPH_HITMISS, k))

    hit_count = int(np.count_nonzero(hits))
    if enlarge and hit_count > 0:
        hits = cv2.dilate(hits, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    meta.update({"pattern": pattern, "hits": hit_count})
    return hits, meta


def op_thinning_pruning(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Iterative Zhang-Suen morphological thinning with optional spur pruning (Chapter 4)."""
    pre_mode = str(params.get("binarize", "Otsu Auto-Binary"))
    if pre_mode == "Off (Gray/Color)":
        pre_mode = "Otsu Auto-Binary"
    max_iters = int(params.get("max_iters", 40))
    prune_steps = int(params.get("prune_steps", 2))
    bw, meta = _maybe_binarize(to_gray(img), pre_mode)

    # Vectorized Zhang-Suen thinning on {0, 1} grid
    grid = (bw > 0).astype(np.uint8)
    iters = 0
    for iters in range(1, max_iters + 1):
        changed = False
        for step_idx in (0, 1):
            p2 = np.roll(grid, 1, axis=0)
            p3 = np.roll(p2, -1, axis=1)
            p4 = np.roll(grid, -1, axis=1)
            p5 = np.roll(p4, -1, axis=0)
            p6 = np.roll(grid, -1, axis=0)
            p7 = np.roll(p6, 1, axis=1)
            p8 = np.roll(grid, 1, axis=1)
            p9 = np.roll(p2, 1, axis=1)
            nbrs = [p2, p3, p4, p5, p6, p7, p8, p9]
            b_sum = sum(nbrs)
            trans = np.zeros_like(grid)
            for idx in range(8):
                trans += ((nbrs[idx] == 0) & (nbrs[(idx + 1) % 8] == 1)).astype(np.uint8)
            cond0 = (grid == 1) & (b_sum >= 2) & (b_sum <= 6) & (trans == 1)
            if step_idx == 0:
                cond1 = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond1 = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rm = cond0 & cond1
            rm[0, :] = rm[-1, :] = rm[:, 0] = rm[:, -1] = False
            if np.any(rm):
                grid[rm] = 0
                changed = True
        if not changed:
            break

    out = grid * 255
    for _ in range(prune_steps):
        endpoints = np.zeros_like(out)
        for k in _HMT_KERNELS["Line Endpoints"]:
            endpoints = cv2.bitwise_or(endpoints, cv2.morphologyEx(out, cv2.MORPH_HITMISS, k))
        if np.count_nonzero(endpoints) == 0:
            break
        out = cv2.subtract(out, endpoints)

    meta.update({"iterations": iters, "prune_steps": prune_steps})
    return out, meta


def _morph_params(default_binarize: str = "Off (Gray/Color)") -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            "binarize",
            "Pre-Binarize",
            ParamType.CHOICE,
            default_binarize,
            choices=("Off (Gray/Color)", "Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)"),
            tooltip="Apply binary thresholding before morphology (or operate directly on grayscale/binary input)",
        ),
        ParamSpec("se_shape", "Structuring Element", ParamType.CHOICE, "Rectangle", choices=tuple(_SE_SHAPES.keys())),
        ParamSpec("se_square", "Square SE (W = H)", ParamType.BOOL, True),
        ParamSpec("se_w", "Kernel Width", ParamType.INT, 5, 1, 35, 2, odd_only=True),
        ParamSpec("se_h", "Kernel Height", ParamType.INT, 5, 1, 35, 2, odd_only=True),
        ParamSpec("iterations", "Iterations", ParamType.INT, 1, 1, 15, 1),
    )


def get_specs() -> list[OperationSpec]:
    """Return all Thresholding and Morphological operation specifications."""
    return [
        OperationSpec(
            op_id="morph.threshold",
            name="Global Threshold",
            category=OperationCategory.MORPHOLOGY,
            description="Segment image into foreground/background at a fixed threshold T.",
            func=op_global_threshold,
            keywords=("threshold", "binary", "global", "segment"),
            params=(
                ParamSpec("threshold", "Threshold (T)", ParamType.INT, 128, 0, 255, 1),
                ParamSpec(
                    "mode",
                    "Threshold Mode",
                    ParamType.CHOICE,
                    "Binary",
                    choices=("Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"),
                ),
            ),
        ),
        OperationSpec(
            op_id="morph.otsu",
            name="Otsu Threshold (Auto)",
            category=OperationCategory.MORPHOLOGY,
            description="Automatically select optimal threshold T* maximizing between-class variance.",
            func=op_otsu_threshold,
            keywords=("otsu", "binary", "threshold", "automatic"),
            params=(
                ParamSpec("invert", "Invert Output (Dark Foreground)", ParamType.BOOL, False),
                ParamSpec("pre_blur", "Gaussian Pre-Blur (0 = Off)", ParamType.INT, 0, 0, 15, 1),
            ),
        ),
        OperationSpec(
            op_id="morph.adaptive",
            name="Adaptive Threshold",
            category=OperationCategory.MORPHOLOGY,
            description="Compute spatially varying local threshold for uneven illumination.",
            func=op_adaptive_threshold,
            keywords=("adaptive", "local", "threshold", "illumination"),
            params=(
                ParamSpec("method", "Method", ParamType.CHOICE, "Gaussian", choices=("Gaussian", "Mean")),
                ParamSpec("block_size", "Block Size", ParamType.INT, 15, 3, 99, 2, odd_only=True),
                ParamSpec("c_offset", "Constant C", ParamType.FLOAT, 5.0, -30.0, 30.0, 1.0, 0),
                ParamSpec("invert", "Invert Output", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="morph.multi_threshold",
            name="Multi-Level / Iterative T*",
            category=OperationCategory.MORPHOLOGY,
            description="3-level thresholding (T1, T2) or iterative optimal global threshold convergence T*.",
            func=op_multi_threshold,
            keywords=("multi", "iterative", "threshold", "3-level", "chapter 5"),
            params=(
                ParamSpec("mode", "Mode", ParamType.CHOICE, "Iterative Auto T*", choices=("Iterative Auto T*", "3-Level (T1, T2)")),
                ParamSpec("t1", "Low Threshold (T1)", ParamType.INT, 85, 0, 254, 1),
                ParamSpec("t2", "High Threshold (T2)", ParamType.INT, 170, 1, 255, 1),
            ),
        ),
        OperationSpec(
            op_id="morph.niblack_sauvola",
            name="Niblack / Sauvola Threshold",
            category=OperationCategory.MORPHOLOGY,
            description="Local variance-aware thresholding ideal for shadowed documents and handwriting.",
            func=op_niblack_sauvola,
            keywords=("niblack", "sauvola", "document", "adaptive", "threshold", "variance"),
            params=(
                ParamSpec("method", "Formula", ParamType.CHOICE, "Sauvola", choices=("Sauvola", "Niblack")),
                ParamSpec("block_size", "Window Size", ParamType.INT, 25, 3, 99, 2, odd_only=True),
                ParamSpec("k_val", "Sensitivity (k)", ParamType.FLOAT, 0.25, -0.80, 0.80, 0.05, 2),
                ParamSpec("sauvola_r", "Dynamic Range (R)", ParamType.FLOAT, 128.0, 32.0, 255.0, 8.0, 0),
                ParamSpec("invert", "Invert Foreground", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="morph.erode",
            name="Erode",
            category=OperationCategory.MORPHOLOGY,
            description="Shrink bright regions and remove small foreground specks (A ⊖ B).",
            func=op_erode,
            preview_widget="se",
            keywords=("erode", "erosion", "morphology", "shrink"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.dilate",
            name="Dilate",
            category=OperationCategory.MORPHOLOGY,
            description="Expand bright regions and bridge small gaps/breaks (A ⊕ B).",
            func=op_dilate,
            preview_widget="se",
            keywords=("dilate", "dilation", "morphology", "expand"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.open",
            name="Morphological Open",
            category=OperationCategory.MORPHOLOGY,
            description="Erosion followed by Dilation (A ∘ B): removes noise protrusions while preserving size.",
            func=op_open,
            preview_widget="se",
            keywords=("open", "opening", "noise", "morphology"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.close",
            name="Morphological Close",
            category=OperationCategory.MORPHOLOGY,
            description="Dilation followed by Erosion (A • B): fills internal pinholes and dark gaps.",
            func=op_close,
            preview_widget="se",
            keywords=("close", "closing", "holes", "morphology"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.gradient",
            name="Morphological Gradient",
            category=OperationCategory.MORPHOLOGY,
            description="Difference between Dilation and Erosion ((A ⊕ B) − (A ⊖ B)) highlighting boundaries.",
            func=op_morph_gradient,
            preview_widget="se",
            keywords=("gradient", "outline", "morphology"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.tophat",
            name="Top-Hat (Background Correct)",
            category=OperationCategory.MORPHOLOGY,
            description="Original minus Opening (f − (f ∘ b)) to extract bright objects on uneven backgrounds.",
            func=op_tophat,
            preview_widget="se",
            keywords=("tophat", "top-hat", "illumination", "rice", "morphology"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.blackhat",
            name="Black-Hat",
            category=OperationCategory.MORPHOLOGY,
            description="Closing minus Original ((f • b) − f) to extract dark features on bright backgrounds.",
            func=op_blackhat,
            preview_widget="se",
            keywords=("blackhat", "black-hat", "morphology"),
            params=_morph_params(),
        ),
        OperationSpec(
            op_id="morph.boundary",
            name="Boundary Extraction",
            category=OperationCategory.MORPHOLOGY,
            description="Extract clean 1-pixel or multi-pixel binary boundaries via A − (A ⊖ B).",
            func=op_boundary_extract,
            preview_widget="se",
            keywords=("boundary", "perimeter", "morphology"),
            params=_morph_params(default_binarize="Otsu Auto-Binary"),
        ),
        OperationSpec(
            op_id="morph.skeleton",
            name="Morphological Skeleton",
            category=OperationCategory.MORPHOLOGY,
            description="Iterative topological thinning to compute the medial skeleton of binary shapes.",
            func=op_skeleton,
            preview_widget="se",
            keywords=("skeleton", "thinning", "medial", "morphology"),
            params=(
                ParamSpec(
                    "binarize",
                    "Pre-Binarize",
                    ParamType.CHOICE,
                    "Otsu Auto-Binary",
                    choices=("Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)"),
                ),
                ParamSpec("se_shape", "Structuring Element", ParamType.CHOICE, "Cross (+)", choices=tuple(_SE_SHAPES.keys())),
                ParamSpec("max_steps", "Max Iterations", ParamType.INT, 64, 5, 200, 5),
            ),
        ),
        OperationSpec(
            op_id="morph.hole_fill",
            name="Morphological Hole Filling",
            category=OperationCategory.MORPHOLOGY,
            description="Flood-fill enclosed interior holes inside binary foreground regions.",
            func=op_hole_fill,
            keywords=("hole", "fill", "floodfill", "morphology"),
            params=(
                ParamSpec("binarize", "Pre-Binarize", ParamType.CHOICE, "Otsu Auto-Binary", choices=("Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)")),
            ),
        ),
        OperationSpec(
            op_id="morph.convex_hull",
            name="Morphological Convex Hull",
            category=OperationCategory.MORPHOLOGY,
            description="Construct the outer convex envelope of each foreground object or show convexity defects.",
            func=op_convex_hull,
            keywords=("convex hull", "envelope", "defects", "morphology"),
            params=(
                ParamSpec("binarize", "Pre-Binarize", ParamType.CHOICE, "Otsu Auto-Binary", choices=("Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)")),
                ParamSpec("display", "Display Mode", ParamType.CHOICE, "Hull Over Original", choices=("Hull Over Original", "Filled Binary Hull", "Defects (Hull - Object)")),
                ParamSpec("min_area", "Min Component Area", ParamType.INT, 40, 1, 2000, 10, unit="px"),
            ),
        ),
        OperationSpec(
            op_id="morph.hit_or_miss",
            name="Hit-or-Miss Transform",
            category=OperationCategory.MORPHOLOGY,
            description="Detect exact binary configurations (corners, isolated pixels, endpoints) via A ⊛ B.",
            func=op_hit_or_miss,
            keywords=("hit or miss", "hmt", "corners", "endpoints", "morphology"),
            params=(
                ParamSpec("binarize", "Pre-Binarize", ParamType.CHOICE, "Otsu Auto-Binary", choices=("Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)")),
                ParamSpec("pattern", "Structuring Pattern", ParamType.CHOICE, "All 4 Corners (Union)", choices=tuple(_HMT_KERNELS.keys())),
                ParamSpec("visible_markers", "Dilate Hits for Visibility", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="morph.thinning",
            name="Thinning & Spur Pruning",
            category=OperationCategory.MORPHOLOGY,
            description="Vectorized Zhang-Suen topological thinning with endpoint spur pruning.",
            func=op_thinning_pruning,
            keywords=("thinning", "zhang-suen", "pruning", "skeleton", "morphology"),
            params=(
                ParamSpec("binarize", "Pre-Binarize", ParamType.CHOICE, "Otsu Auto-Binary", choices=("Otsu Auto-Binary", "Otsu Inverted", "Manual (T=128)")),
                ParamSpec("max_iters", "Max Thinning Passes", ParamType.INT, 40, 1, 120, 2),
                ParamSpec("prune_steps", "Spur Pruning Passes", ParamType.INT, 2, 0, 25, 1),
            ),
        ),
    ]
