"""Built-in one-click workflow presets demonstrating multi-step non-destructive pipelines."""

from __future__ import annotations

from typing import Any


BUILTIN_PRESETS: dict[str, list[dict[str, Any]]] = {
    "Document Scan Cleanup": [
        {"op_id": "basics.grayscale", "params": {"method": "Rec.601 (OpenCV)", "keep_3ch": False}},
        {"op_id": "morph.adaptive", "params": {"method": "Gaussian", "block_size": 19, "c_offset": 7.0, "invert": False}},
        {"op_id": "morph.open", "params": {"binarize": "Off (Gray/Color)", "se_shape": "Rectangle", "se_square": True, "se_w": 3, "se_h": 3, "iterations": 1}},
    ],
    "Portrait Clarity & Tone": [
        {"op_id": "hist.clahe", "params": {"clip_limit": 2.0, "tile_grid": 8, "color_space": "LAB (Lightness L)"}},
        {"op_id": "art.clarity", "params": {"sigma_s": 12.0, "sigma_r": 0.14}},
        {"op_id": "art.vignette", "params": {"strength": 0.45, "radius": 0.85}},
    ],
    "Salt & Pepper Denoise Demo": [
        {"op_id": "restore.noise", "params": {"noise_type": "Salt & Pepper", "amount": 0.08, "sigma": 22.0, "seed": 42}},
        {"op_id": "spatial.median", "params": {"ksize": 5}},
    ],
    "Edge Blueprint": [
        {"op_id": "spatial.gaussian", "params": {"ksize": 3, "sigma": 0.8}},
        {"op_id": "spatial.canny", "params": {"auto_thresholds": True, "threshold1": 60, "threshold2": 160, "blur_ksize": 3, "l2_gradient": True}},
        {"op_id": "color.pseudocolor", "params": {"colormap": "Turbo"}},
    ],
    "Medical Contrast Boost": [
        {"op_id": "hist.clahe", "params": {"clip_limit": 3.2, "tile_grid": 8, "color_space": "LAB (Lightness L)"}},
        {"op_id": "intensity.gamma", "params": {"gamma": 0.82, "gain": 1.0}},
        {"op_id": "spatial.unsharp", "params": {"amount": 1.1, "sigma": 1.2, "ksize": 5}},
    ],
    "Vintage Film Look": [
        {"op_id": "intensity.brightness_contrast", "params": {"brightness": 4, "contrast": 1.12}},
        {"op_id": "art.toning", "params": {"preset": "Warm Film (Kodak)", "strength": 0.75}},
        {"op_id": "art.vignette", "params": {"strength": 0.55, "radius": 0.75}},
    ],
}
