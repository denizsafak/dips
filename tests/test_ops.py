"""Comprehensive unit tests for every registered DIP Studio operation on grayscale and BGR images."""

from __future__ import annotations

import numpy as np
import pytest

from dips.core import (
    BUILTIN_PRESETS,
    ImagePipeline,
    compute_quality_metrics,
    discover_sample_images,
    get_registry,
    load_image,
    save_image,
)
from dips.core.ops_histogram import compute_histogram, compute_stats
from dips.core.ops_intensity import compute_lut
from dips.ui.update_checker import is_newer_version, parse_version, select_platform_asset


@pytest.fixture
def sample_bgr() -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.zeros((96, 128, 3), dtype=np.uint8)
    img[16:80, 24:104] = [60, 140, 210]
    noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_gray() -> np.ndarray:
    img = np.zeros((96, 128), dtype=np.uint8)
    img[20:76, 30:98] = 190
    img[35:60, 50:78] = 75
    return img


def test_every_registered_op_runs_cleanly_on_bgr_and_gray(sample_bgr: np.ndarray, sample_gray: np.ndarray) -> None:
    registry = get_registry()
    specs = registry.all_specs()
    assert len(specs) >= 110

    for spec in specs:
        params = spec.default_params()
        out_bgr, meta_bgr = spec.func(sample_bgr, params)
        assert out_bgr.dtype == np.uint8
        assert out_bgr.ndim in (2, 3)
        assert isinstance(meta_bgr, dict)

        out_gray, meta_gray = spec.func(sample_gray, params)
        assert out_gray.dtype == np.uint8
        assert out_gray.ndim in (2, 3)
        assert isinstance(meta_gray, dict)

    assert registry.get("ai.face_auto_crop").default_params()["framing"] == "Close-up Face"
    sticker_out, _ = registry.get("ai.sticker_cutout").func(sample_bgr, {"background": "Transparent (Alpha)", "outline_px": 8})
    assert sticker_out.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 4)

    _, glcm_meta = registry.get("feat.texture").func(sample_gray, {"method": "GLCM Contrast & Stats"})
    assert "contrast" in glcm_meta and "homogeneity" in glcm_meta
    _, sift_meta = registry.get("feat.local_descriptors").func(sample_bgr, {"algorithm": "SIFT (Scale-Invariant)"})
    assert "keypoints" in sift_meta


def test_negative_gamma_and_luts(sample_gray: np.ndarray) -> None:
    reg = get_registry()
    neg, _ = reg.get("intensity.negative").func(sample_gray, {})
    assert np.array_equal(neg, 255 - sample_gray)

    lut = compute_lut("intensity.gamma", {"gamma": 0.5, "gain": 1.0})
    assert lut is not None and lut.shape == (256,)
    assert lut[0] == 0 and lut[255] == 255 and lut[64] > 64


def test_otsu_returns_threshold_and_binary_output(sample_gray: np.ndarray) -> None:
    reg = get_registry()
    out, meta = reg.get("morph.otsu").func(sample_gray, {"invert": False, "pre_blur": 0})
    assert "otsu_threshold" in meta
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_affine_matrix_and_expansion(sample_bgr: np.ndarray) -> None:
    reg = get_registry()
    out, meta = reg.get("affine.rotate").func(sample_bgr, {"angle": 45.0, "expand": True})
    assert "affine_matrix" in meta
    assert out.shape[0] > sample_bgr.shape[0]
    assert out.shape[1] > sample_bgr.shape[1]
    assert out.shape[2] == 4
    assert int(out[0, 0, 3]) == 0


def test_unicode_io_roundtrip(tmp_path, sample_bgr: np.ndarray) -> None:
    path = tmp_path / "şafak_test_image.png"
    save_image(sample_bgr, path)
    loaded = load_image(path)
    assert loaded.shape == sample_bgr.shape
    assert np.array_equal(loaded, sample_bgr)


def test_histogram_and_stats(sample_bgr: np.ndarray) -> None:
    h = compute_histogram(sample_bgr)
    assert set(h.keys()) == {"lum", "r", "g", "b"}
    assert h["lum"].sum() == sample_bgr.shape[0] * sample_bgr.shape[1]
    st = compute_stats(sample_bgr)
    assert st.width == 128 and st.height == 96 and st.channels == 3
    assert st.entropy > 0.0


def test_quality_metrics_samples_presets_and_versioning(sample_bgr: np.ndarray) -> None:
    identical = compute_quality_metrics(sample_bgr, sample_bgr)
    assert identical is not None
    assert identical["psnr"] >= 99.0
    assert abs(identical["ssim"] - 1.0) < 1e-4

    shifted = np.clip(sample_bgr.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    degraded = compute_quality_metrics(sample_bgr, shifted)
    assert degraded is not None
    assert degraded["psnr"] < 30.0
    assert degraded["ssim"] < 1.0

    samples = discover_sample_images()
    assert len(samples) >= 16

    assert parse_version("v1.2.3") == (1, 2, 3)
    assert is_newer_version("v1.0.1", "1.0.0") is True
    assert is_newer_version("v1.0.0", "1.0.0") is False
    assert is_newer_version("v0.9.9", "1.0.0") is False

    assets = [
        {"name": "DIPS-windows-amd64.zip", "browser_download_url": "https://example/win.zip"},
        {"name": "DIPS-macos-arm64.zip", "browser_download_url": "https://example/mac-arm.zip"},
        {"name": "DIPS-macos-amd64.zip", "browser_download_url": "https://example/mac-x64.zip"},
        {"name": "DIPS-linux-amd64.tar.gz", "browser_download_url": "https://example/lin.tar.gz"},
    ]
    assert select_platform_asset(assets) is not None

    from pathlib import Path
    import tomllib
    from dips import __version__
    from dips.core.ops_ai import get_preferred_ort_providers

    py_ver = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert __version__ == py_ver
    providers = get_preferred_ort_providers()
    assert providers[-1] == "CPUExecutionProvider"

    reg = get_registry()
    up_spec = reg.get("ai.upscale")
    lena = load_image(next(p for p in samples if p.name == "lena.png"))
    espcn_out, espcn_meta = up_spec.func(lena, {"model": "ESPCN 3× (Ultra Fast)", "scale": "3×", "crispness": 0.1})
    assert espcn_out.shape[:2] == (lena.shape[0] * 3, lena.shape[1] * 3)
    assert espcn_meta["model"] == "ESPCN-ONNX"

    bg_spec = reg.get("ai.bg_remove")
    bg_out, bg_meta = bg_spec.func(sample_bgr, {"model": "RMBG-1.4 (Studio Quality)", "background": "Transparent (Alpha)"})
    assert bg_out.shape == (sample_bgr.shape[0], sample_bgr.shape[1], 4)
    assert bg_meta["model"] == "RMBG-1.4"
    assert bg_meta["channels"] == "RGBA"

    pipe = ImagePipeline()
    for name, steps in BUILTIN_PRESETS.items():
        loaded = pipe.load_preset_steps(steps)
        assert loaded == len(steps), f"Preset {name} failed to load all steps"
        out, _, _ = pipe.evaluate(sample_bgr)
        assert out.dtype == np.uint8
