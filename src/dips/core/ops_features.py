"""Segmentation & Feature Extraction (Chapters 5 & 7): Connected Components, Watershed, K-Means, Corners, Contours, Hough."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .io_utils import ensure_uint8, to_bgr, to_gray
from .types import OperationCategory, OperationSpec, ParamSpec, ParamType


def _label_palette(num_labels: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 245, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors


def op_connected_components(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Label connected regions using Otsu thresholding and 4/8 connectivity."""
    gray = to_gray(img)
    invert = bool(params.get("invert", False))
    min_area = int(params.get("min_area", 25))
    conn = 4 if str(params.get("connectivity", "8-Connected")) == "4-Connected" else 8

    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, bw = cv2.threshold(gray, 0, 255, flag)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=conn)

    palette = _label_palette(num_labels)
    vis = np.zeros((*gray.shape, 3), dtype=np.uint8)
    kept = 0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            vis[labels == idx] = palette[idx]
            cx, cy = centroids[idx]
            cv2.circle(vis, (int(cx), int(cy)), 3, (255, 255, 255), -1)
            kept += 1

    return vis, {"components": kept, "total_raw": max(0, num_labels - 1), "connectivity": conn}


def op_watershed(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Distance-transform marker-based Watershed segmentation for touching objects (Ch. 5)."""
    bgr = to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    invert = bool(params.get("invert", False))
    dist_ratio = float(params.get("dist_ratio", 0.45))

    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, bw = cv2.threshold(gray, 0, 255, flag)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_d = float(dist.max()) if dist.size else 0.0
    _, sure_fg = cv2.threshold(dist, dist_ratio * max_d, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)

    overlay = bgr.copy()
    overlay[markers == -1] = [60, 120, 255]  # crisp accent boundary
    return overlay, {"regions": max(0, num_markers - 1), "dist_ratio": dist_ratio}


def op_kmeans(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """K-Means color/intensity clustering segmentation (Chapter 5)."""
    img = ensure_uint8(img)
    k = max(2, min(16, int(params.get("k", 4))))
    max_iter = int(params.get("attempts", 3))

    if img.ndim == 2:
        data = img.reshape(-1, 1).astype(np.float32)
    else:
        data = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, max_iter, cv2.KMEANS_PP_CENTERS)
    centers_u8 = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    quantized = centers_u8[labels.flatten()].reshape(img.shape)
    return quantized, {"clusters_k": k}


def op_corners(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect corners/keypoints using Harris, Shi-Tomasi, or FAST (Chapter 7)."""
    bgr = to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detector = str(params.get("detector", "Shi-Tomasi"))
    count = 0

    if detector == "Harris":
        block = max(2, int(params.get("block_size", 2)))
        ksize = max(3, int(params.get("ksize", 3)) | 1)
        harris_k = float(params.get("harris_k", 0.04))
        thresh_pct = float(params.get("threshold_pct", 0.02))
        dst = cv2.cornerHarris(gray.astype(np.float32), block, ksize, harris_k)
        dst = cv2.dilate(dst, None)
        mask = dst > thresh_pct * float(dst.max() if dst.size else 0.0)
        ys, xs = np.where(mask)
        count = int(len(xs))
        # Draw crisp circles on sampled peaks if dense
        step = max(1, count // 1500)
        for x, y in zip(xs[::step], ys[::step]):
            cv2.circle(bgr, (int(x), int(y)), 2, (245, 139, 76), -1)
    elif detector == "FAST":
        thresh = int(params.get("fast_threshold", 25))
        nms = bool(params.get("nonmax_suppression", True))
        fast = cv2.FastFeatureDetector_create(threshold=thresh, nonmaxSuppression=nms)
        kps = fast.detect(gray, None)
        count = len(kps)
        for kp in kps[:2000]:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv2.circle(bgr, (x, y), 3, (90, 210, 140), 1, lineType=cv2.LINE_AA)
    else:  # Shi-Tomasi
        max_corners = int(params.get("max_corners", 150))
        quality = max(0.001, float(params.get("quality_level", 0.02)))
        min_dist = max(1, int(params.get("min_distance", 10)))
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist)
        if corners is not None:
            pts = corners.reshape(-1, 2)
            count = len(pts)
            for px, py in pts:
                x, y = int(round(float(px))), int(round(float(py)))
                cv2.circle(bgr, (x, y), 4, (245, 139, 76), 1, lineType=cv2.LINE_AA)
                cv2.circle(bgr, (x, y), 1, (255, 255, 255), -1)

    return bgr, {"detector": detector, "keypoints": count}


def op_contours_hu(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract contours, bounding boxes, and compute Hu Moment invariants for largest shape (Ch. 7)."""
    bgr = to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    invert = bool(params.get("invert", False))
    min_area = int(params.get("min_area", 50))
    draw_boxes = bool(params.get("draw_boxes", True))

    flag = (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU
    _, bw = cv2.threshold(gray, 0, 255, flag)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    cv2.drawContours(bgr, valid, -1, (245, 139, 76), 2, lineType=cv2.LINE_AA)
    if draw_boxes:
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (100, 210, 150), 1)

    meta: dict[str, Any] = {"contours": len(valid)}
    if valid:
        largest = max(valid, key=cv2.contourArea)
        m = cv2.moments(largest)
        hu = cv2.HuMoments(m).ravel()
        meta["largest_area_px"] = int(round(cv2.contourArea(largest)))
        meta["hu1_hu2"] = f"{hu[0]:.4f}, {hu[1]:.4e}"
    return bgr, meta


def op_hough(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect geometric lines (Probabilistic Hough) or circles (Hough Gradient) (Ch. 5)."""
    bgr = to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mode = str(params.get("mode", "Lines (HoughLinesP)"))

    if mode == "Circles (HoughCircles)":
        blurred = cv2.medianBlur(gray, 5)
        dp = float(params.get("dp", 1.2))
        min_dist = max(5, int(params.get("min_dist", 30)))
        param2 = max(10, int(params.get("accumulator_thresh", 45)))
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=120,
            param2=param2,
            minRadius=5,
            maxRadius=0,
        )
        count = 0
        if circles is not None:
            rounded = np.uint16(np.around(circles.reshape(-1, 3)))
            count = len(rounded)
            for cx, cy, r in rounded:
                cv2.circle(bgr, (int(cx), int(cy)), int(r), (245, 139, 76), 2, lineType=cv2.LINE_AA)
                cv2.circle(bgr, (int(cx), int(cy)), 2, (255, 255, 255), -1)
        return bgr, {"mode": "Circles", "detected": count}

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    thresh = max(10, int(params.get("accumulator_thresh", 60)))
    min_len = max(5, int(params.get("min_line_len", 40)))
    max_gap = max(1, int(params.get("max_line_gap", 10)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=thresh, minLineLength=min_len, maxLineGap=max_gap)
    count = 0
    if lines is not None:
        reshaped = lines.reshape(-1, 4)
        count = len(reshaped)
        for x1, y1, x2, y2 in reshaped:
            cv2.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (245, 139, 76), 2, lineType=cv2.LINE_AA)
    return bgr, {"mode": "Lines", "detected": count}


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def op_document_scan(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect dominant 4-corner document page, dewarp perspective, and apply clean scan enhancement."""
    bgr = to_bgr(img)
    h, w = bgr.shape[:2]
    gray = to_gray(bgr)
    look = str(params.get("look", "Clean Scan (Contrast)"))
    clean_strength = float(params.get("clean_strength", 0.65))
    auto_dewarp = bool(params.get("auto_dewarp", True))

    warped = bgr
    quad_found = False
    if auto_dewarp:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_quad_area = 0.12 * float(h * w)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < min_quad_area:
                break
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
                tl, tr, br, bl = rect
                dw = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 24)
                dh = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 24)
                dst = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(bgr, M, (dw, dh), flags=cv2.INTER_CUBIC)
                quad_found = True
                break

    if look == "Original Rectified":
        return warped, {"quad_detected": quad_found, "look": look}

    scan_gray = to_gray(warped)
    if look == "B&W Ink (Adaptive)":
        block = max(11, (min(scan_gray.shape[:2]) // 18) | 1)
        bw = cv2.adaptiveThreshold(
            cv2.GaussianBlur(scan_gray, (3, 3), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            max(4, int(round(4 + 12 * clean_strength))),
        )
        return bw, {"quad_detected": quad_found, "look": look}

    # Clean Scan (Contrast): divide out slow paper illumination variation
    bg_sigma = max(12.0, min(scan_gray.shape[:2]) / 8.0)
    paper_illum = cv2.GaussianBlur(scan_gray.astype(np.float32), (0, 0), bg_sigma)
    normalized = np.clip((scan_gray.astype(np.float32) / np.maximum(paper_illum, 12.0)) * 238.0, 0, 255)
    clahe = cv2.createCLAHE(clipLimit=1.5 + 1.5 * clean_strength, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(np.clip(np.round(normalized), 0, 255).astype(np.uint8))
    out = cv2.addWeighted(to_gray(warped), 1.0 - clean_strength, enhanced_gray, clean_strength, 0.0)
    return out, {"quad_detected": quad_found, "look": look}


def _quadtree_split(gray_f: np.ndarray, out: np.ndarray, y0: int, y1: int, x0: int, x1: int, std_thresh: float, min_blk: int, draw_grid: bool, leaves: list[int]) -> None:
    sub = gray_f[y0:y1, x0:x1]
    h, w = sub.shape
    if h <= min_blk or w <= min_blk or float(np.std(sub)) <= std_thresh:
        mean_v = float(sub.mean())
        out[y0:y1, x0:x1] = mean_v
        if draw_grid and h >= 2 and w >= 2:
            out[y0, x0:x1] = 255
            out[y0:y1, x0] = 255
        leaves[0] += 1
        return
    ym = (y0 + y1) // 2
    xm = (x0 + x1) // 2
    _quadtree_split(gray_f, out, y0, ym, x0, xm, std_thresh, min_blk, draw_grid, leaves)
    _quadtree_split(gray_f, out, y0, ym, xm, x1, std_thresh, min_blk, draw_grid, leaves)
    _quadtree_split(gray_f, out, ym, y1, x0, xm, std_thresh, min_blk, draw_grid, leaves)
    _quadtree_split(gray_f, out, ym, y1, xm, x1, std_thresh, min_blk, draw_grid, leaves)


def op_region_segment(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Region-based segmentation via Seeded Region Growing or Quadtree Split & Merge (Ch. 5)."""
    bgr = to_bgr(img)
    gray = to_gray(bgr)
    mode = str(params.get("mode", "Quadtree Split & Merge"))
    if mode == "Seeded Region Growing":
        tol = max(1, int(params.get("tolerance", 18)))
        h, w = gray.shape
        seed_x = max(0, min(w - 1, int(round(w * float(params.get("seed_x_pct", 50.0)) / 100.0))))
        seed_y = max(0, min(h - 1, int(round(h * float(params.get("seed_y_pct", 50.0)) / 100.0))))
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(gray, mask, (seed_x, seed_y), 255, tol, tol, flags)
        region = mask[1:-1, 1:-1] > 0
        vis = bgr.copy()
        tint = vis.copy()
        tint[region] = (94, 168, 255)
        cv2.addWeighted(tint, 0.50, vis, 0.50, 0.0, vis)
        cv2.circle(vis, (seed_x, seed_y), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (seed_x, seed_y), 5, (79, 83, 217), 1, lineType=cv2.LINE_AA)
        pct = round(float(np.count_nonzero(region)) / max(region.size, 1) * 100.0, 1)
        return vis, {"mode": "Region Growing", "area_pct": f"{pct}%"}

    std_thresh = float(params.get("std_thresh", 16.0))
    min_blk = max(4, int(params.get("min_block", 8)))
    draw_grid = bool(params.get("draw_grid", True))
    out_f = np.zeros_like(gray, dtype=np.float32)
    leaves = [0]
    _quadtree_split(gray.astype(np.float32), out_f, 0, gray.shape[0], 0, gray.shape[1], std_thresh, min_blk, draw_grid, leaves)
    return np.clip(np.round(out_f), 0, 255).astype(np.uint8), {"mode": "Quadtree", "leaves": leaves[0]}


def op_blobs(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale-space blob detection via LoG, DoG, or Determinant of Hessian (Chapter 7)."""
    bgr = to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    detector = str(params.get("detector", "LoG (Laplacian of Gaussian)"))
    sigma = max(0.8, float(params.get("sigma", 2.5)))
    thresh_pct = float(params.get("threshold_pct", 0.30))
    show_map = bool(params.get("show_response", False))

    if detector.startswith("DoG"):
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
        resp = np.abs(g2 - g1)
    elif detector.startswith("DoH"):
        sm = cv2.GaussianBlur(gray, (0, 0), sigma)
        lxx = cv2.Sobel(sm, cv2.CV_64F, 2, 0, ksize=3)
        lyy = cv2.Sobel(sm, cv2.CV_64F, 0, 2, ksize=3)
        lxy = cv2.Sobel(sm, cv2.CV_64F, 1, 1, ksize=3)
        resp = np.abs((sigma ** 4) * (lxx * lyy - lxy * lxy))
    else:
        sm = cv2.GaussianBlur(gray, (0, 0), sigma)
        resp = np.abs((sigma ** 2) * cv2.Laplacian(sm, cv2.CV_64F, ksize=3))

    if show_map:
        norm = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(np.clip(norm, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), {"detector": detector.split(" ")[0]}

    dil = cv2.dilate(resp, np.ones((7, 7), dtype=np.uint8))
    mx = float(resp.max()) if resp.size else 0.0
    peaks = (resp == dil) & (resp >= thresh_pct * mx) & (mx > 1e-8)
    ys, xs = np.where(peaks)
    radius = max(3, int(round(sigma * 1.414)))
    for x, y in zip(xs[:1200], ys[:1200]):
        cv2.circle(bgr, (int(x), int(y)), radius, (245, 139, 76), 1, lineType=cv2.LINE_AA)
    return bgr, {"detector": detector.split(" ")[0], "blobs": int(len(xs)), "sigma": sigma}


def op_texture(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Texture analysis: Local Binary Patterns (LBP), Gabor filter, or GLCM Haralick metrics (Ch. 7)."""
    gray = to_gray(img)
    method = str(params.get("method", "LBP (Local Binary Pattern)"))

    if method.startswith("Gabor"):
        theta = np.deg2rad(float(params.get("angle_deg", 45.0)))
        lambd = max(2.0, float(params.get("wavelength", 8.0)))
        sigma = max(1.0, float(params.get("sigma", 3.5)))
        ksize = max(5, int(round(sigma * 4.0)) | 1)
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, 0.5, 0.0, ktype=cv2.CV_32F)
        resp = np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel))
        norm = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(np.round(norm), 0, 255).astype(np.uint8), {"method": "Gabor", "angle": f"{int(params.get('angle_deg', 45))}°"}

    if method.startswith("GLCM"):
        levels = 32
        q = (gray.astype(np.int32) * (levels - 1)) // 255
        dist = max(1, int(params.get("glcm_dist", 2)))
        a = q[:, :-dist].ravel()
        b = q[:, dist:].ravel()
        glcm = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(glcm, (a, b), 1.0)
        glcm = glcm + glcm.T
        tot = float(glcm.sum())
        if tot > 0:
            glcm /= tot
        ii, jj = np.indices((levels, levels))
        contrast = float(np.sum(((ii - jj) ** 2) * glcm))
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))
        energy = float(np.sqrt(np.sum(glcm ** 2)))
        diff_img = np.abs(gray[:, dist:].astype(np.float32) - gray[:, :-dist].astype(np.float32))
        padded = np.pad(diff_img, ((0, 0), (0, dist)), mode="edge")
        vis = cv2.applyColorMap(np.clip(cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return vis, {
            "method": "GLCM-0°",
            "contrast": round(contrast, 2),
            "homogeneity": round(homogeneity, 3),
            "energy": round(energy, 3),
        }

    # Vectorized 8-neighbor LBP
    g = gray.astype(np.int16)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)
        lbp |= ((shifted >= g).astype(np.uint8) << bit)
    return lbp, {"method": "LBP-8", "unique_codes": int(len(np.unique(lbp)))}


def op_hog_vis(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute Histogram of Oriented Gradients (HOG) and render cell orientation stars (Ch. 7)."""
    gray = to_gray(img).astype(np.float32)
    h, w = gray.shape
    cell = max(8, min(32, int(params.get("cell_size", 16))))
    overlay = bool(params.get("overlay", True))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0

    canvas = (cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR) * 0.35).astype(np.uint8) if overlay else np.zeros((h, w, 3), dtype=np.uint8)
    n_bins = 9
    bin_angles = np.deg2rad(np.arange(n_bins) * (180.0 / n_bins) + 90.0)
    max_r = cell * 0.45

    cells_y = max(1, h // cell)
    cells_x = max(1, w // cell)
    for cy in range(cells_y):
        for cx in range(cells_x):
            y0, x0 = cy * cell, cx * cell
            m_cell = mag[y0 : y0 + cell, x0 : x0 + cell]
            a_cell = ang[y0 : y0 + cell, x0 : x0 + cell]
            hist, _ = np.histogram(a_cell, bins=n_bins, range=(0.0, 180.0), weights=m_cell)
            h_mx = float(hist.max())
            if h_mx <= 1e-3:
                continue
            hist = hist / h_mx
            center_x = x0 + cell // 2
            center_y = y0 + cell // 2
            for b_idx in range(n_bins):
                strength = float(hist[b_idx])
                if strength < 0.25:
                    continue
                rad = max_r * strength
                dx = int(round(rad * float(np.cos(bin_angles[b_idx]))))
                dy = int(round(rad * float(np.sin(bin_angles[b_idx]))))
                cv2.line(canvas, (center_x - dx, center_y - dy), (center_x + dx, center_y + dy), (94, 168, 255), 1, lineType=cv2.LINE_AA)
    return canvas, {"cell_px": f"{cell}×{cell}", "bins": n_bins, "cells": f"{cells_x}×{cells_y}"}


def op_local_descriptors(img: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect scale/rotation-invariant SIFT or ORB local keypoints & descriptors (Ch. 7)."""
    bgr = to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    algo = str(params.get("algorithm", "ORB (Fast Oriented BRIEF)"))
    max_kp = int(params.get("max_keypoints", 250))

    if algo.startswith("SIFT") and hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=max_kp)
        kps, desc = det.detectAndCompute(gray, None)
        label = "SIFT-128D"
    else:
        det = cv2.ORB_create(nfeatures=max_kp)
        kps, desc = det.detectAndCompute(gray, None)
        label = "ORB-256bit"

    out = cv2.drawKeypoints(
        bgr,
        kps[:max_kp],
        None,
        color=(94, 168, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    dim = int(desc.shape[1]) if desc is not None and desc.ndim == 2 else 0
    return out, {"detector": label, "keypoints": len(kps), "descriptor_dim": dim}


def get_specs() -> list[OperationSpec]:
    """Return all Segmentation & Feature operation specifications."""
    return [
        OperationSpec(
            op_id="feat.components",
            name="Connected Components",
            category=OperationCategory.FEATURES,
            description="Label distinct foreground blobs with 4- or 8-connectivity and centroid markers.",
            func=op_connected_components,
            keywords=("connected", "components", "labeling", "blobs"),
            params=(
                ParamSpec("connectivity", "Connectivity", ParamType.CHOICE, "8-Connected", choices=("8-Connected", "4-Connected")),
                ParamSpec("min_area", "Min Component Area", ParamType.INT, 25, 1, 5000, 10, unit="px"),
                ParamSpec("invert", "Invert Threshold (Dark Objects)", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="feat.watershed",
            name="Watershed Segmentation",
            category=OperationCategory.FEATURES,
            description="Distance-transform marker Watershed to separate touching objects.",
            func=op_watershed,
            keywords=("watershed", "segmentation", "touching", "coins"),
            params=(
                ParamSpec("dist_ratio", "Seed Distance Ratio", ParamType.FLOAT, 0.45, 0.10, 0.90, 0.05, 2),
                ParamSpec("invert", "Dark Objects on Light BG", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="feat.kmeans",
            name="K-Means Color Quantization",
            category=OperationCategory.FEATURES,
            description="Cluster pixels into K representative intensity/color regions.",
            func=op_kmeans,
            keywords=("kmeans", "clustering", "quantize", "segmentation"),
            params=(
                ParamSpec("k", "Clusters (K)", ParamType.INT, 4, 2, 16, 1),
                ParamSpec("attempts", "Init Attempts", ParamType.INT, 3, 1, 10, 1),
            ),
        ),
        OperationSpec(
            op_id="feat.corners",
            name="Corner & Keypoint Detector",
            category=OperationCategory.FEATURES,
            description="Detect Harris, Shi-Tomasi, or FAST keypoints overlaid on the image.",
            func=op_corners,
            keywords=("harris", "shi-tomasi", "fast", "corners", "keypoints"),
            params=(
                ParamSpec("detector", "Detector", ParamType.CHOICE, "Shi-Tomasi", choices=("Shi-Tomasi", "Harris", "FAST")),
                ParamSpec("max_corners", "Max Corners (Shi-Tomasi)", ParamType.INT, 150, 10, 1000, 10),
                ParamSpec("quality_level", "Quality Level", ParamType.FLOAT, 0.02, 0.002, 0.20, 0.005, 3),
                ParamSpec("min_distance", "Min Distance", ParamType.INT, 10, 1, 50, 1, unit="px"),
                ParamSpec("threshold_pct", "Harris Peak Ratio", ParamType.FLOAT, 0.02, 0.002, 0.25, 0.005, 3),
                ParamSpec("fast_threshold", "FAST Threshold", ParamType.INT, 25, 5, 100, 1),
            ),
        ),
        OperationSpec(
            op_id="feat.contours",
            name="Contours & Hu Moments",
            category=OperationCategory.FEATURES,
            description="Trace object boundaries, bounding boxes, and compute Hu invariant moments.",
            func=op_contours_hu,
            keywords=("contours", "hu moments", "shape", "bounding box"),
            params=(
                ParamSpec("min_area", "Min Contour Area", ParamType.INT, 50, 1, 5000, 10, unit="px"),
                ParamSpec("draw_boxes", "Draw Bounding Boxes", ParamType.BOOL, True),
                ParamSpec("invert", "Dark Objects on Light BG", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="feat.hough",
            name="Hough Lines / Circles",
            category=OperationCategory.FEATURES,
            description="Detect parametric straight lines or circles via Hough transform.",
            func=op_hough,
            keywords=("hough", "lines", "circles", "geometry"),
            params=(
                ParamSpec("mode", "Primitive", ParamType.CHOICE, "Lines (HoughLinesP)", choices=("Lines (HoughLinesP)", "Circles (HoughCircles)")),
                ParamSpec("accumulator_thresh", "Accumulator Threshold", ParamType.INT, 60, 10, 250, 5),
                ParamSpec("min_line_len", "Min Line Length", ParamType.INT, 40, 5, 300, 5, unit="px"),
                ParamSpec("max_line_gap", "Max Line Gap", ParamType.INT, 10, 1, 100, 1, unit="px"),
                ParamSpec("min_dist", "Min Circle Center Dist", ParamType.INT, 30, 5, 200, 5, unit="px"),
            ),
        ),
        OperationSpec(
            op_id="feat.document_scan",
            name="Document Scanner & Dewarp",
            category=OperationCategory.FEATURES,
            description="Auto-detect 4-corner document borders, rectify perspective, and enhance page contrast.",
            func=op_document_scan,
            keywords=("document", "scanner", "scan", "dewarp", "perspective", "page", "whiteboard"),
            params=(
                ParamSpec("look", "Scan Finish", ParamType.CHOICE, "Clean Scan (Contrast)", choices=("Clean Scan (Contrast)", "B&W Ink (Adaptive)", "Original Rectified")),
                ParamSpec("clean_strength", "Clean / Contrast Mix", ParamType.FLOAT, 0.65, 0.0, 1.0, 0.05, 2),
                ParamSpec("auto_dewarp", "Auto-Detect 4-Corner Quad", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="feat.region_segment",
            name="Region Growing & Split-Merge",
            category=OperationCategory.FEATURES,
            description="Seeded Flood-Fill Region Growing or Quadtree homogeneity Split-and-Merge (Ch. 5).",
            func=op_region_segment,
            keywords=("region growing", "quadtree", "split", "merge", "segmentation", "seed"),
            params=(
                ParamSpec("mode", "Method", ParamType.CHOICE, "Quadtree Split & Merge", choices=("Quadtree Split & Merge", "Seeded Region Growing")),
                ParamSpec("std_thresh", "Quadtree Std Threshold", ParamType.FLOAT, 16.0, 2.0, 60.0, 1.0, 1),
                ParamSpec("min_block", "Min Block Size", ParamType.INT, 8, 4, 64, 2, unit="px"),
                ParamSpec("draw_grid", "Show Quadtree Grid", ParamType.BOOL, True),
                ParamSpec("seed_x_pct", "Seed X", ParamType.FLOAT, 50.0, 5.0, 95.0, 2.0, 0, unit="%"),
                ParamSpec("seed_y_pct", "Seed Y", ParamType.FLOAT, 50.0, 5.0, 95.0, 2.0, 0, unit="%"),
                ParamSpec("tolerance", "Intensity Tolerance", ParamType.INT, 18, 1, 100, 1),
            ),
        ),
        OperationSpec(
            op_id="feat.blobs",
            name="Blob Detector (LoG / DoG / DoH)",
            category=OperationCategory.FEATURES,
            description="Detect multi-scale intensity blobs using LoG, DoG, or Determinant of Hessian (Ch. 7).",
            func=op_blobs,
            keywords=("blob", "log", "dog", "hessian", "scale space", "laplacian of gaussian"),
            params=(
                ParamSpec("detector", "Operator", ParamType.CHOICE, "LoG (Laplacian of Gaussian)", choices=("LoG (Laplacian of Gaussian)", "DoG (Difference of Gaussians)", "DoH (Determinant of Hessian)")),
                ParamSpec("sigma", "Scale (σ)", ParamType.FLOAT, 2.5, 0.8, 10.0, 0.2, 1),
                ParamSpec("threshold_pct", "Peak Ratio", ParamType.FLOAT, 0.30, 0.05, 0.90, 0.05, 2),
                ParamSpec("show_response", "Show Response Heatmap", ParamType.BOOL, False),
            ),
        ),
        OperationSpec(
            op_id="feat.texture",
            name="Texture (LBP / Gabor / GLCM)",
            category=OperationCategory.FEATURES,
            description="Analyze micro-texture via Local Binary Patterns, Gabor Filters, or GLCM Haralick stats (Ch. 7).",
            func=op_texture,
            keywords=("texture", "lbp", "gabor", "glcm", "haralick", "local binary"),
            params=(
                ParamSpec("method", "Method", ParamType.CHOICE, "LBP (Local Binary Pattern)", choices=("LBP (Local Binary Pattern)", "Gabor Filter", "GLCM Contrast & Stats")),
                ParamSpec("angle_deg", "Gabor Orientation", ParamType.FLOAT, 45.0, 0.0, 180.0, 5.0, 0, unit="°"),
                ParamSpec("wavelength", "Gabor Wavelength (λ)", ParamType.FLOAT, 8.0, 2.0, 32.0, 1.0, 1),
                ParamSpec("sigma", "Gabor Sigma (σ)", ParamType.FLOAT, 3.5, 1.0, 12.0, 0.5, 1),
                ParamSpec("glcm_dist", "GLCM Offset (d)", ParamType.INT, 2, 1, 16, 1, unit="px"),
            ),
        ),
        OperationSpec(
            op_id="feat.hog",
            name="HOG (Oriented Gradients)",
            category=OperationCategory.FEATURES,
            description="Compute Histogram of Oriented Gradients and visualize directional cell histograms (Ch. 7).",
            func=op_hog_vis,
            keywords=("hog", "oriented gradients", "descriptor", "cells", "bins"),
            params=(
                ParamSpec("cell_size", "Cell Size", ParamType.INT, 16, 8, 32, 2, unit="px"),
                ParamSpec("overlay", "Blend Over Image", ParamType.BOOL, True),
            ),
        ),
        OperationSpec(
            op_id="feat.local_descriptors",
            name="Local Features (SIFT / ORB)",
            category=OperationCategory.FEATURES,
            description="Detect scale- and rotation-invariant SIFT (128D) or ORB keypoints with orientation rings (Ch. 7).",
            func=op_local_descriptors,
            keywords=("sift", "orb", "keypoints", "descriptor", "local features", "scale invariant"),
            params=(
                ParamSpec("algorithm", "Descriptor", ParamType.CHOICE, "ORB (Fast Oriented BRIEF)", choices=("ORB (Fast Oriented BRIEF)", "SIFT (Scale-Invariant)")),
                ParamSpec("max_keypoints", "Max Keypoints", ParamType.INT, 250, 20, 1000, 25),
            ),
        ),
    ]
