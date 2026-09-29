"""Focused validation for the continuous colour / multi-scale fitter."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiscale_fitter import (
    MultiScaleFitConfig,
    benchmark_image,
    fit_multiscale_image,
    synthetic_shaded_scene,
)
from evaluator import GridEvaluator
from run_all import _try_multiscale_procedural_path


def _fast_colour_config(**overrides):
    values = dict(
        pyramid_long_sides=(24, 48),
        layers_per_scale=1,
        max_layers=2,
        max_radials=1,
        refine_steps=12,
        acceptance_margin_bits=1.0,
        min_validation_data_gain_bits=0.1,
        min_validation_psnr_gain=0.001,
        min_validation_payback=0.0,
        max_source_length=1200,
        device="cpu",
    )
    values.update(overrides)
    return MultiScaleFitConfig(**values)


def test_multiscale_fitter_improves_a_shaded_multicolour_continuous_scene():
    target = synthetic_shaded_scene(48)
    bilinear_only = fit_multiscale_image(
        target,
        _fast_colour_config(max_layers=0, max_radials=0, refine_steps=1),
    )
    result = fit_multiscale_image(target, _fast_colour_config())

    assert result.program.is_coordinate_continuous
    assert result.reconstruction.shape == target.shape
    assert result.psnr >= bilinear_only.psnr + 2.0
    assert result.ssim >= 0.70
    assert result.accepted_layers
    assert result.source_length < 1200
    assert all(item.validation_data_gain_bits > 0.0 for item in result.accepted_layers)

    # The same scalar program is valid at a different sampling resolution;
    # no target-sized texture/grid is embedded in its source.
    enlarged = result.program.render(73, 91, device="cpu")
    source = result.program.serialize().lower()
    assert enlarged.shape == (73, 91, 3)
    assert torch.isfinite(enlarged).all()
    assert "texture" not in source and "dct" not in source and "pixel" not in source

    evaluator_source = result.program.to_current_evaluator_expression(48, 48)
    evaluator_image = GridEvaluator().evaluate(
        evaluator_source, 48, 48, differentiable=True, render_mode="linear"
    ).float() / 255.0
    assert len(evaluator_source) < 6000
    assert torch.allclose(evaluator_image, result.reconstruction, atol=2.5e-3)


def test_local_complex_image_crop_returns_an_auditable_not_lossless_result():
    image_path = Path(__file__).resolve().parents[1] / "Image.jpg"
    if not image_path.exists():
        pytest.skip("no local complex image fixture is available")
    pil_image = pytest.importorskip("PIL.Image")
    image = pil_image.open(image_path).convert("RGB")
    width, height = image.size
    crop = image.crop((width // 2 - 80, height // 2 - 80, width // 2 + 80, height // 2 + 80))
    target = np.array(crop.resize((40, 40)), copy=True)

    report = benchmark_image(
        target,
        _fast_colour_config(
            pyramid_long_sides=(24, 40), max_layers=1, refine_steps=6,
        ),
    )

    assert report["continuous_program"] is True
    assert report["stop_reason"] in {
        "pyramid_complete", "no_rate_distortion_gain", "layer_budget", "budget",
    }
    assert report["source_length"] < 1200
    assert report["model_bits"] > 0.0
    assert math.isfinite(report["psnr"])
    assert math.isfinite(report["ssim"])
    # Natural-detail quality is reported rather than declared lossless.
    assert "lossless" not in report["equation"].lower()


def test_pipeline_accepts_a_high_fidelity_continuous_colour_field():
    height, width = 48, 64
    y, x = np.mgrid[0:height, 0:width]
    target = np.stack(
        (
            30.0 + 150.0 * x / (width - 1),
            20.0 + 120.0 * y / (height - 1),
            50.0 + 80.0 * (x / (width - 1)) * (y / (height - 1)),
        ),
        axis=-1,
    ).round().astype(np.uint8)

    accepted = _try_multiscale_procedural_path(target, GridEvaluator())

    assert accepted is not None
    assert accepted["psnr"] >= 28.0
    assert accepted["ssim"] >= 0.90
    assert accepted["source_length"] < 6000
    assert "dct" not in accepted["equation"].lower()
