import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_deep_ast import deep_csg_fbm_expression
from evaluator import GridEvaluator


def test_deep_csg_fbm_source_cache_preserves_output_and_reuses_ast():
    equation = deep_csg_fbm_expression(primitives=20, octaves=6)
    evaluator = GridEvaluator(expression_cache_size=2)

    first = evaluator.evaluate(equation, 48, 40, render_mode="linear")
    second = evaluator.evaluate(equation, 48, 40, render_mode="linear")

    assert torch.equal(first, second)
    assert evaluator.cache_info() == {
        "expression_entries": 1,
        "expression_capacity": 2,
        "expression_hits": 1,
        "expression_misses": 1,
        "grid_entries": 1,
    }


def test_cached_expression_keeps_parameter_gradients_live():
    evaluator = GridEvaluator()
    gain = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    equation = "0.5+gain*sum(s=1..5,0.5^s*cos(2^s*x+s*y))"

    rendered = evaluator.evaluate(
        equation,
        32,
        32,
        differentiable=True,
        params={"gain": gain},
        render_mode="linear",
    )
    rendered.mean().backward()

    assert gain.grad is not None
    assert torch.isfinite(gain.grad)

    with torch.no_grad():
        gain.add_(0.2)
    updated = evaluator.evaluate(
        equation,
        32,
        32,
        differentiable=True,
        params={"gain": gain},
        render_mode="linear",
    )
    assert not torch.equal(rendered.detach(), updated.detach())
    assert evaluator.cache_info()["expression_hits"] == 1
