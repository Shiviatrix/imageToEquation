import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import GridEvaluator, Lexer, Parser
from run_all import _try_symbolic_path
from multiscale_fitter import psnr
from symbolic_fitter import (
    FitConfig,
    Primitive,
    SymbolicProgram,
    benchmark_synthetic_fish,
    fit_binary_mask,
    synthetic_fish_mask,
)


def _fast_config() -> FitConfig:
    # Fixed targeted-validation budget, not a parameter sweep.
    return FitConfig(max_shapes=4, refine_steps=32, device="cpu")


def test_greedy_mdl_fits_compact_synthetic_fish():
    report = benchmark_synthetic_fish(80, _fast_config())

    assert report["iou"] >= 0.80
    assert 0 < report["source_length"] < 1000
    assert report["primitive_kinds"]
    assert all(improvement > 0.0 for improvement in report["accepted_improvements"])
    assert report["mdl_bits"] == report["data_bits"] + report["model_bits"]


def test_flat_colour_input_and_renderer_match_target_shape():
    mask = synthetic_fish_mask(72).cpu()
    # A non-black flat foreground exercises the documented colour-mask path.
    image = torch.zeros((72, 72, 3), dtype=torch.uint8)
    image[mask.bool()] = torch.tensor([35, 180, 240], dtype=torch.uint8)
    result = fit_binary_mask(image, _fast_config())

    assert result.iou >= 0.78
    assert result.program.render(72, 72, device="cpu").shape == (72, 72)


def test_supported_implicit_field_parses_in_current_evaluator_grammar():
    program = SymbolicProgram(
        [
            Primitive.disk(0.1, 0.0, 0.3),
            Primitive.superellipse(-0.15, 0.05, 0.22, 0.1, 2.0, 0.1),
            Primitive.wedge(-0.5, 0.0, 0.3, 0.4, 3.14159),
        ]
    )
    field = program.to_current_evaluator_field()

    assert field is not None
    Parser(Lexer(field).tokens).parse()
    assert GridEvaluator().evaluate(field, 24, 24).shape == (24, 24, 3)
    executable = program.to_current_evaluator_expression(24, 24, colour=(0.2, 0.5, 0.8))
    assert executable is not None
    image = GridEvaluator().evaluate(executable, 24, 24, render_mode="linear")
    assert image.shape == (24, 24, 3)
    assert image.dtype == torch.uint8
    rounded_box = SymbolicProgram([Primitive("rounded_box", {
        "cx": 0.0, "cy": 0.0, "theta": 0.0, "hx": 0.2, "hy": 0.1, "radius": 0.05
    })])
    rounded_box_source = rounded_box.to_current_evaluator_expression(24, 24)
    assert rounded_box_source is not None
    assert GridEvaluator().evaluate(rounded_box_source, 24, 24, render_mode="linear").shape == (24, 24, 3)


def test_pipeline_accepts_verified_compact_flat_colour_fish():
    mask = synthetic_fish_mask(80).cpu().numpy().astype(bool)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[mask] = np.array([38, 153, 230], dtype=np.uint8)

    accepted = _try_symbolic_path(image, GridEvaluator())

    assert accepted is not None
    assert accepted["iou"] >= 0.80
    assert accepted["source_length"] < 1000
    assert accepted["primitive_kinds"]
    decoded = torch.as_tensor(accepted["rendered"], dtype=torch.float32) / 255.0
    assert psnr(decoded, torch.as_tensor(image, dtype=torch.float32) / 255.0) >= 20.0
