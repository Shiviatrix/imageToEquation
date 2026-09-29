"""Stream a supervised, evidence-led corpus for neural symbolic regression.

This intentionally does not download or copy an artist's images.  Each record
contains a canonical compact AST and an antialiased render of it.  The strict
curriculum uses only sin/cos, low-degree/even powers, sums/products, and bounded
normalised coordinates.  The evaluator's soft SDF wrapper is a rasterisation
device; it is not presented to the model as a claimed Yeganeh construction.

Examples
--------
Stream samples directly into a PyTorch training loop::

    dataset = SyntheticYeganehDataset(count=1_000_000, image_size=64)

Materialise a resumable set of shard files (large; roughly 12 GB raw per one
million 64px RGB samples)::

    python synthetic_yeganeh_dataset.py --count 1000000 --output data/yeganeh
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset

from image_to_equation.evaluator import GridEvaluator
from image_to_equation.neural_symbolic import YeganehVocabulary, program_to_source


def _prefix(*items: str | list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        result.extend(item if isinstance(item, list) else [item])
    return result


def _constant(vocabulary: YeganehVocabulary, value: float) -> list[str]:
    return [vocabulary.quantize(value)]


def _sub(left: list[str], right: list[str]) -> list[str]:
    return _prefix("sub", left, right)


def _add(left: list[str], right: list[str]) -> list[str]:
    return _prefix("add", left, right)


def _mul(left: list[str], right: list[str]) -> list[str]:
    return _prefix("mul", left, right)


def _sin(argument: list[str]) -> list[str]:
    return _prefix("sin", argument)


def _cos(argument: list[str]) -> list[str]:
    return _prefix("cos", argument)


def _pow2(value: list[str]) -> list[str]:
    return _prefix("pow2", value)


def _smin(left: list[str], right: list[str], vocabulary: YeganehVocabulary, sharpness: float = 1.5) -> list[str]:
    return _prefix("smin", left, right, _constant(vocabulary, sharpness))


def curriculum_phase(step: int) -> int:
    if step < 5_000:
        return 1
    if step < 15_000:
        return 2
    if step < 30_000:
        return 3
    if step < 45_000:
        return 4
    if step < 60_000:
        return 5
    if step < 75_000:
        return 6
    return 7


def _circle_field(vocabulary: YeganehVocabulary, rng: random.Random, central: bool) -> list[str]:
    """A high-contrast single circle; the first learnable inverse-render task."""
    centre_spread = 0.10 if central else 0.30
    cx, cy = 0.5 + rng.uniform(-centre_spread, centre_spread), 0.5 + rng.uniform(-centre_spread, centre_spread)
    radius = rng.uniform(0.16, 0.28) if central else rng.uniform(0.08, 0.30)
    return _sub(_add(
        _pow2(_sub(["x"], _constant(vocabulary, cx))),
        _pow2(_sub(["y"], _constant(vocabulary, cy))),
    ), _constant(vocabulary, radius * radius))


def _line_field(vocabulary: YeganehVocabulary, rng: random.Random, central: bool) -> list[str]:
    centre_spread = 0.10 if central else 0.32
    offset = 0.5 + rng.uniform(-centre_spread, centre_spread)
    thickness = rng.uniform(0.05, 0.12) if central else rng.uniform(0.025, 0.11)
    return _sub(_pow2(_sub(["y"], _constant(vocabulary, offset))), _constant(vocabulary, thickness * thickness))


def _ellipse_field(vocabulary: YeganehVocabulary, rng: random.Random) -> list[str]:
    """A rotated ellipse, lowered from complex-plane rotation to real algebra."""
    cx, cy = rng.uniform(0.30, 0.70), rng.uniform(0.30, 0.70)
    rx, ry = rng.uniform(0.12, 0.32), rng.uniform(0.08, 0.26)
    angle = rng.uniform(-math.pi, math.pi)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = _sub(["x"], _constant(vocabulary, cx))
    dy = _sub(["y"], _constant(vocabulary, cy))
    u = _add(_mul(_constant(vocabulary, cosine), dx), _mul(_constant(vocabulary, sine), dy))
    v = _sub(_mul(_constant(vocabulary, cosine), dy), _mul(_constant(vocabulary, sine), dx))
    return _sub(_add(
        _pow2(_mul(_constant(vocabulary, 1.0 / rx), u)),
        _pow2(_mul(_constant(vocabulary, 1.0 / ry), v)),
    ), _constant(vocabulary, 1.0))


def _trig_band_field(vocabulary: YeganehVocabulary, rng: random.Random) -> list[str]:
    """A continuous stroke-like family based on bounded sine/cosine paths."""
    base = rng.uniform(0.25, 0.75)
    amplitude = rng.uniform(0.06, 0.18)
    frequency = rng.choice((1, 2, 3, 4, 5, 6, 8))
    phase = rng.uniform(-1.2, 1.2)
    path = _add(
        _constant(vocabulary, base),
        _mul(_constant(vocabulary, amplitude), _cos(_add(
            _mul(_constant(vocabulary, frequency), ["x"]), _constant(vocabulary, phase)
        ))),
    )
    # Squared distance is algebraic and domain-total; its threshold becomes a
    # stroke only in the renderer, not in the learned motif vocabulary.
    return _sub(_pow2(_sub(["y"], path)), _constant(vocabulary, rng.uniform(0.002, 0.012)))


def _trignoise_field(vocabulary: YeganehVocabulary, rng: random.Random) -> list[str]:
    base_freq = rng.uniform(2.0, 25.0)
    lacunarity = rng.uniform(1.2, 2.5)
    octaves = rng.choice((1, 2, 3, 4, 5))
    return _prefix(
        "trignoise",
        ["x"],
        ["y"],
        _constant(vocabulary, base_freq),
        _constant(vocabulary, lacunarity),
        _constant(vocabulary, float(octaves)),
    )


def _polar_field(vocabulary: YeganehVocabulary, rng: random.Random) -> list[str]:
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    dx = _sub(["x"], _constant(vocabulary, cx))
    dy = _sub(["y"], _constant(vocabulary, cy))
    angle = _prefix("atan2", dy, dx)
    freq = _constant(vocabulary, rng.choice((1, 2, 3, 4, 5)))
    scale = _constant(vocabulary, rng.uniform(5.0, 20.0))
    radius_sq = _add(_pow2(dx), _pow2(dy))
    return _sin(_add(_mul(freq, angle), _mul(scale, radius_sq)))

def _groundtruth_carrier(vocabulary: YeganehVocabulary, rng: random.Random, s: int) -> list[str]:
    """An exact Yeganeh rotated trigonometric carrier wave: cos(scale * (cos(a)x + sin(a)y) + phase)."""
    s_f = float(s)
    scale = _constant(vocabulary, min(2.0, (11.0 / 10.0)**s_f))
    angle = rng.uniform(-math.pi, math.pi)
    phase = _constant(vocabulary, rng.uniform(-2.0, 2.0))
    # u = cos(a)x + sin(a)y
    u = _add(
        _mul(_constant(vocabulary, math.cos(angle)), ["x"]),
        _mul(_constant(vocabulary, math.sin(angle)), ["y"])
    )
    return _cos(_add(_mul(scale, u), phase))

def _groundtruth_texture(vocabulary: YeganehVocabulary, rng: random.Random, depth: int) -> list[str]:
    """Exact hierarchical summation E(x,y) = sum w_s T_s(x,y)."""
    if depth <= 1:
        return _groundtruth_carrier(vocabulary, rng, 1)
    weight = _constant(vocabulary, rng.uniform(0.6, 0.9))
    return _add(
        _groundtruth_carrier(vocabulary, rng, depth),
        _mul(weight, _groundtruth_texture(vocabulary, rng, depth - 1))
    )


def _high_contrast_palette(rng: random.Random) -> tuple[list[float], list[float]]:
    """Never emit black-on-black or near-flat scenes, a collapse trigger."""
    background = [rng.uniform(0.10, 0.85) for _ in range(3)]
    foreground = [min(0.95, max(0.10, 1.0 - channel + rng.uniform(-0.12, 0.12))) for channel in background]
    if math.sqrt(sum((a - b) ** 2 for a, b in zip(background, foreground))) < 0.55:
        foreground = [1.0 - channel for channel in background]
    return background, foreground


def sample_program(vocabulary: YeganehVocabulary, rng: random.Random, step: int = 0) -> list[str]:
    """Sample a staged target: primitive, smooth composition, then trig sweeps."""
    phase = curriculum_phase(step)
    background, foreground = _high_contrast_palette(rng)
    edge = rng.uniform(0.015, 0.08)
    if phase == 1:
        field = _circle_field(vocabulary, rng, central=True) if rng.random() < 0.65 else _line_field(vocabulary, rng, central=True)
    elif phase == 2:
        first = _circle_field(vocabulary, rng, central=False) if rng.random() < 0.55 else _line_field(vocabulary, rng, central=False)
        second = _ellipse_field(vocabulary, rng) if rng.random() < 0.55 else _trig_band_field(vocabulary, rng)
        field = _smin(first, second, vocabulary)
        if rng.random() < 0.35:
            field = _smin(field, _circle_field(vocabulary, rng, central=False), vocabulary)
    elif phase == 3:
        # The band is the rasterised form of a ruled trigonometric sweep; the
        # ellipse applies the documented complex-plane rotation concept.
        field = _smin(_trig_band_field(vocabulary, rng), _ellipse_field(vocabulary, rng), vocabulary)
        if rng.random() < 0.65:
            field = _smin(field, _trig_band_field(vocabulary, rng), vocabulary)
    elif phase == 4:
        # Textured shapes
        shape = _circle_field(vocabulary, rng, central=False) if rng.random() < 0.5 else _ellipse_field(vocabulary, rng)
        texture = _trignoise_field(vocabulary, rng)
        field = _add(shape, _mul(_constant(vocabulary, rng.uniform(0.01, 0.1)), texture))
        if rng.random() < 0.5:
            field = _prefix("dexp", field)
    elif phase == 5:
        # Layered compositions with dexp mask
        first = _prefix("dexp", _circle_field(vocabulary, rng, central=False))
        second = _prefix("dexp", _ellipse_field(vocabulary, rng))
        field = _smin(first, second, vocabulary)
    elif phase == 6:
        # Polar constructions & true hierarchical texture
        polar_tex = _polar_field(vocabulary, rng)
        texture = _groundtruth_texture(vocabulary, rng, depth=rng.choice((2, 3, 4)))
        shape = _circle_field(vocabulary, rng, central=True)
        field = _add(shape, _mul(_constant(vocabulary, rng.uniform(0.05, 0.2)), texture))
        if rng.random() < 0.5:
            field = _prefix("dexp", field)
    else:
        # Let bindings with exact texture and nested masking
        tex = _groundtruth_texture(vocabulary, rng, depth=3)
        c1 = _add(_circle_field(vocabulary, rng, central=True), _mul(_constant(vocabulary, 0.1), ["t0"]))
        c2 = _add(_circle_field(vocabulary, rng, central=False), _mul(_constant(vocabulary, 0.1), ["t0"]))
        field = _prefix("let", tex, _smin(c1, c2, vocabulary))
    return _prefix(
        "SCENE",
        *[_constant(vocabulary, value) for value in background + foreground + [edge]],
        field,
        "<eos>",
    )


class SyntheticYeganehDataset(Dataset):
    """Deterministic, storage-free synthetic `(image, canonical AST)` corpus."""

    def __init__(self, count: int, image_size: int = 64, seed: int = 20260722, vocabulary: YeganehVocabulary | None = None, curriculum_step: int = 0, max_source_length: int | None = None) -> None:
        if count <= 0 or image_size < 16:
            raise ValueError("count must be positive and image_size must be at least 16")
        self.count = count
        self.image_size = image_size
        self.seed = seed
        self.vocabulary = vocabulary or YeganehVocabulary()
        self.curriculum_step = curriculum_step
        self.max_source_length = max_source_length
        self._evaluator: GridEvaluator | None = None

    def __len__(self) -> int:
        return self.count

    def set_curriculum_step(self, step: int) -> None:
        self.curriculum_step = max(0, int(step))

    def sample(self, index: int, step: int | None = None) -> dict[str, torch.Tensor | str]:
        active_step = self.curriculum_step if step is None else step
        rng = random.Random(self.seed + int(index) * 1_000_003)
        # An optional budget remains available for compression experiments;
        # quality-first training leaves it disabled so richer targets survive.
        for _ in range(32):
            program = sample_program(self.vocabulary, rng, active_step)
            source = program_to_source(program, self.vocabulary)
            if self.max_source_length is None or len(source) <= self.max_source_length:
                break
        else:
            raise RuntimeError("unable to sample a program within source budget")
        if self._evaluator is None:
            self._evaluator = GridEvaluator(expression_cache_size=0)
        image = self._evaluator.evaluate(source, self.image_size, self.image_size, render_mode="linear")
        return {
            "image": (image.float() / 255.0).permute(2, 0, 1).cpu(),
            "tokens": torch.tensor(self.vocabulary.encode(program), dtype=torch.long),
            "source": source,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.sample(index)


def collate_programs(samples: Sequence[dict[str, torch.Tensor | str]], vocabulary: YeganehVocabulary) -> dict[str, torch.Tensor]:
    """Pad variable-length AST targets for teacher forcing."""
    images = torch.stack([sample["image"] for sample in samples])  # type: ignore[arg-type]
    sequences = [torch.cat((torch.tensor([vocabulary.bos_id]), sample["tokens"])) for sample in samples]  # type: ignore[arg-type]
    length = max(sequence.numel() for sequence in sequences)
    tokens = torch.full((len(sequences), length), vocabulary.pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, :sequence.numel()] = sequence
    return {"image": images, "decoder_input": tokens[:, :-1], "target": tokens[:, 1:]}


def materialise_shards(dataset: SyntheticYeganehDataset, output: Path, shard_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for first in range(0, len(dataset), shard_size):
        samples = [dataset[index] for index in range(first, min(first + shard_size, len(dataset)))]
        path = output / f"yeganeh_{first:09d}.pt"
        torch.save(samples, path)
        print(f"wrote {path} ({len(samples)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000, help="set 1_000_000 for a million-sample corpus")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True, help="directory for .pt shards")
    parser.add_argument("--shard-size", type=int, default=1_000)
    args = parser.parse_args()
    dataset = SyntheticYeganehDataset(args.count, args.image_size, args.seed)
    materialise_shards(dataset, args.output, args.shard_size)


if __name__ == "__main__":
    main()
