"""Tests for ImagePipeline non-destructive evaluation, undo/redo, preset JSON, and batching."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from dips.core import ImagePipeline, load_image, save_image


def test_pipeline_undo_redo_and_toggle() -> None:
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    pipe = ImagePipeline()

    pipe.add_step("basics.grayscale", {"keep_3ch": False})
    pipe.add_step("intensity.negative", {})
    assert len(pipe.steps) == 2

    res, results, total_ms = pipe.evaluate(img)
    assert res.ndim == 2
    assert int(res[0, 0]) == 155
    assert len(results) == 2
    assert total_ms >= 0.0

    # Toggle negative off
    pipe.toggle_step(1)
    res_toggled, _, _ = pipe.evaluate(img)
    assert int(res_toggled[0, 0]) == 100

    # Undo toggle -> back to negative on
    assert pipe.undo() is True
    res_undo, _, _ = pipe.evaluate(img)
    assert int(res_undo[0, 0]) == 155

    # Redo toggle -> back to off
    assert pipe.redo() is True
    res_redo, _, _ = pipe.evaluate(img)
    assert int(res_redo[0, 0]) == 100


def test_pipeline_draft_commit_and_edit() -> None:
    img = np.full((48, 48), 50, dtype=np.uint8)
    pipe = ImagePipeline()

    pipe.set_draft("intensity.negative")
    res_preview, _, _ = pipe.evaluate(img, include_draft=True)
    assert int(res_preview[0, 0]) == 205
    assert len(pipe.steps) == 0

    idx = pipe.commit_draft()
    assert idx == 0
    assert len(pipe.steps) == 1

    # Edit step 0 in place
    pipe.set_draft("intensity.brightness_contrast", {"brightness": 25, "contrast": 1.0}, editing_index=0)
    res_edit, _, _ = pipe.evaluate(img, include_draft=True)
    assert int(res_edit[0, 0]) == 75
    pipe.commit_draft()
    assert len(pipe.steps) == 1
    assert pipe.steps[0].op_id == "intensity.brightness_contrast"


def test_preset_save_load_and_batch(tmp_path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    for name in ("a.png", "b.png"):
        save_image(np.full((32, 32, 3), 40, dtype=np.uint8), in_dir / name)

    pipe = ImagePipeline()
    pipe.add_step("basics.grayscale", {})
    pipe.add_step("intensity.negative", {})
    preset_path = tmp_path / "test.dips.json"
    pipe.save_preset(preset_path)

    pipe2 = ImagePipeline()
    loaded_count = pipe2.load_preset(preset_path)
    assert loaded_count == 2

    outputs = pipe2.batch_process(in_dir, out_dir, out_ext=".png")
    assert len(outputs) == 2
    out_img = load_image(outputs[0])
    assert int(out_img[0, 0]) == 215


def test_pipeline_prefix_cache_skips_unchanged_steps(monkeypatch) -> None:
    img = np.full((32, 32, 3), 80, dtype=np.uint8)
    pipe = ImagePipeline()
    spec_step0 = pipe.registry.get("intensity.negative")
    calls = {"count": 0}
    orig_func = spec_step0.func

    def counted_negative(arr, params):
        calls["count"] += 1
        return orig_func(arr, params)

    monkeypatch.setitem(pipe.registry._specs, "intensity.negative", replace(spec_step0, func=counted_negative))

    pipe.add_step("intensity.negative", {})
    pipe.set_draft("intensity.brightness_contrast", {"brightness": 10, "contrast": 1.0})
    res1, _, _ = pipe.evaluate(img, include_draft=True, image_token=1)
    assert calls["count"] == 1
    assert int(res1[0, 0, 0]) == 185

    # Changing draft step reuses cached committed step 0 without re-running it
    pipe.set_draft("intensity.gamma", {"gamma": 1.0})
    pipe.evaluate(img, include_draft=True, image_token=1)
    assert calls["count"] == 1

    # Changing image_token invalidates prefix cache
    pipe.evaluate(img, include_draft=True, image_token=2)
    assert calls["count"] == 2


def test_pipeline_preserves_alpha_across_ops(tmp_path) -> None:
    from dips.core import discover_sample_images

    samples = {p.name: p for p in discover_sample_images()}
    face = load_image(samples["face.jpg"])
    pipe = ImagePipeline()
    pipe.add_step("ai.bg_remove", {"model": "RMBG-1.4 (Studio Quality)", "background": "Transparent (Alpha)"})

    bg_only, _, _ = pipe.evaluate(face, include_draft=False, image_token=10)
    assert bg_only.ndim == 3 and bg_only.shape[2] == 4
    alpha = bg_only[:, :, 3]
    assert int(alpha.min()) < 30 and int(alpha.max()) > 220

    # Non-spatial (Grayscale, Bilateral, CLAHE, Style) and spatial (Crop, Rotate, Upscale) keep RGBA alpha
    for op_id, params in (
        ("basics.grayscale", {"keep_3ch": False}),
        ("spatial.bilateral", {}),
        ("hist.clahe", {}),
        ("ai.style_transfer", {"style": "Mosaic", "strength": 0.5}),
        ("geom.crop", {"x_pct": 5.0, "y_pct": 5.0, "w_pct": 80.0, "h_pct": 80.0}),
        ("affine.rotate", {"angle": 15.0, "expand": True}),
        ("ai.upscale", {"model": "ESPCN 3\u00d7 (Fast)", "scale": "2\u00d7"}),
    ):
        pipe.set_draft(op_id, params)
        out, _, _ = pipe.evaluate(face, include_draft=True, image_token=10)
        assert out.ndim == 3 and out.shape[2] == 4, f"{op_id} dropped alpha channel"
        assert int(out[:, :, 3].min()) < 30 and int(out[:, :, 3].max()) > 220, f"{op_id} lost transparency mask"

    # Saving BGRA to JPEG/BMP composites cleanly onto white without failing
    out_jpg = tmp_path / "cutout.jpg"
    save_image(bg_only, out_jpg)
    assert load_image(out_jpg).shape == (face.shape[0], face.shape[1], 3)
