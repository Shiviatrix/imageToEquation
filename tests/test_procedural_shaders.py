"""Focused validation for the continuous procedural colour AST layer."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import GridEvaluator


def test_rgb_affine_and_bilinear_gradients_are_continuous_and_resolution_independent():
    renderer = GridEvaluator()
    source = (
        "rgb(affine(x,y,0,1,0),affine(x,y,0,0,1),"
        "bilinear(x,y,0,1,1,0))"
    )
    coarse = renderer.evaluate(source, 33, 33, differentiable=True, render_mode="linear")
    fine = renderer.evaluate(source, 65, 65, differentiable=True, render_mode="linear")

    assert coarse.dtype == torch.float32
    assert coarse.shape == (33, 33, 3)
    # At (0, 0), all three scalar fields are zero.  At (1, 1), R and G
    # are one while the selected bilinear corner is zero.
    assert torch.allclose(coarse[0, 0], torch.tensor([0.0, 0.0, 0.0], device=coarse.device))
    assert torch.allclose(coarse[-1, -1], torch.tensor([255.0, 255.0, 0.0], device=coarse.device))
    # Both canvases sample the same mathematical point (0.5, 0.5).
    assert torch.allclose(coarse[16, 16], fine[32, 32], atol=1.0e-5)
    assert coarse[16, 16, 0] > coarse[16, 15, 0]


def test_fbm_is_finite_and_backpropagates_to_shader_parameters():
    renderer = GridEvaluator()
    params = {
        "frequency": torch.tensor(3.0, device=renderer.device, requires_grad=True),
        "gain": torch.tensor(0.55, device=renderer.device, requires_grad=True),
        "seed": torch.tensor(0.3, device=renderer.device, requires_grad=True),
    }
    # Octaves are a literal grammar bound, hence source length represents the
    # program structure and cannot conceal an unbounded runtime loop.
    source = (
        "rgb(fbm(x,y,frequency,gain,seed,4),"
        "fbm(x,y,frequency,gain,seed,4),"
        "fbm(x,y,frequency,gain,seed,4))"
    )
    image = renderer.evaluate(
        source,
        41,
        29,
        differentiable=True,
        params=params,
        render_mode="linear",
    )
    loss = image.square().mean()
    loss.backward()

    assert torch.isfinite(image).all()
    for parameter in params.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert parameter.grad.abs() > 1.0e-6


def test_sdf_masked_shader_over_composites_foreground_and_background_colours():
    renderer = GridEvaluator()
    source = (
        "over(rgb(0,0,1),rgb(1,0,0),"
        "sdfmask(disk(x,y,0.5,0.5,0.25),0.02))"
    )
    image = renderer.evaluate(source, 33, 33, differentiable=True, render_mode="linear") / 255.0

    # Centre is inside the disk and therefore red; corner is outside and blue.
    centre, corner = image[16, 16], image[0, 0]
    assert centre[0] > 0.99 and centre[2] < 0.01
    assert corner[2] > 0.99 and corner[0] < 0.01
    # Also validate a uniform RGB shader, whose colour is spatially constant
    # but still expands over arbitrary resolution without a texture lookup.
    uniform = renderer.evaluate("rgb(0.2,0.4,0.6)", 17, 11, differentiable=True, render_mode="linear")
    assert torch.allclose(uniform[0, 0], uniform[-1, -1])
    solid_over = renderer.evaluate(
        "over(rgb(0,0,1),rgb(1,0,0),1)", 17, 11, differentiable=True, render_mode="linear"
    )
    assert solid_over[0, 0, 0] > 254.0 and solid_over[0, 0, 2] == 0.0


def test_sdfcolour_preserves_smooth_colour_union_without_repeating_fields():
    renderer = GridEvaluator()
    source = (
        "sdfcolour(rgb(0,0,1),30,0.02,"
        "disk(x,y,0.38,0.5,0.25),rgb(1,0,0),"
        "disk(x,y,0.62,0.5,0.25),rgb(0,1,0))"
    )
    image = renderer.evaluate(source, 51, 51, differentiable=True, render_mode="linear") / 255.0
    assert image[25, 0, 2] > 0.99
    assert image[25, 19, 0] > image[25, 19, 1]
    assert image[25, 31, 1] > image[25, 31, 0]
    assert torch.isfinite(image).all()
