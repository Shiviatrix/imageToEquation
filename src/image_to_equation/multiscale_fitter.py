"""Continuous multi-scale RGB fitting for compact symbolic scene programs.

This module is a deliberately bounded next step after :mod:`symbolic_fitter`.
It fits a *continuous* colour field--a bilinear background, optional radial
lighting terms, and softly blended coloured SDF atoms--to sampled RGB images.
No target samples, tiles, DCT coefficients, or pixel lookup tables are ever
stored in the returned program.

The search is deterministic.  A Gaussian/Laplacian-pyramid schedule locates
large residual structures first, then offers smaller disk/superellipse layers
at later scales.  Each candidate must improve both an MDL-like training code
and a held-out spatial subset before it can be accepted.  That rate-distortion
gate is intentional: it makes failure on high-entropy photographs explicit,
rather than pretending a growing collection of near-pixel primitives is a
universal lossless procedural codec.

Public API
==========
``fit_multiscale_image(image, config=None)``
    Return a :class:`MultiScaleFitResult` for an RGB image.
``benchmark_image(image, config=None)``
    Return compact-program length/bits plus PSNR, SSIM, and edge proxy.
``synthetic_shaded_scene(size=80)``
    A deterministic continuous multi-colour test scene.

The result's ``program.serialize()`` is a compact scene DSL.  The same program
can emit a parser-compatible expression through
``MultiScaleProgram.to_current_evaluator_expression()``; callers must still
independently round-trip render it before accepting a fitted scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from image_to_equation.symbolic_fitter import Primitive, normalized_grid


_EPS = 1e-6


def _number(value: float, digits: int = 4) -> str:
    """Format one scalar compactly without locale-dependent output."""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _rgb_tuple(values: Sequence[float], name: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly three RGB values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _scalar_bits(value: float, delta: float) -> float:
    """Small operational universal code for one quantized real parameter."""
    return 1.0 + 0.5 * math.log2(1.0 + (float(value) / delta) ** 2)


def _rgb_bits(values: Sequence[float]) -> float:
    return sum(_scalar_bits(float(value), 1.0 / 255.0) for value in values)


@dataclass(frozen=True)
class BilinearRGB:
    """A continuous RGB field specified by its four canvas-corner colours."""

    c00: Tuple[float, float, float]
    c10: Tuple[float, float, float]
    c01: Tuple[float, float, float]
    c11: Tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("c00", "c10", "c01", "c11"):
            object.__setattr__(self, name, _rgb_tuple(getattr(self, name), name))

    @classmethod
    def constant(cls, rgb: Sequence[float]) -> "BilinearRGB":
        value = _rgb_tuple(rgb, "rgb")
        return cls(value, value, value, value)

    def render(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = ((x + 1.0) * 0.5).unsqueeze(-1)
        v = ((y + 1.0) * 0.5).unsqueeze(-1)
        device, dtype = x.device, x.dtype
        c00 = torch.tensor(self.c00, device=device, dtype=dtype)
        c10 = torch.tensor(self.c10, device=device, dtype=dtype)
        c01 = torch.tensor(self.c01, device=device, dtype=dtype)
        c11 = torch.tensor(self.c11, device=device, dtype=dtype)
        return (
            (1.0 - u) * (1.0 - v) * c00
            + u * (1.0 - v) * c10
            + (1.0 - u) * v * c01
            + u * v * c11
        )

    def model_bits(self) -> float:
        return 7.0 + _rgb_bits((*self.c00, *self.c10, *self.c01, *self.c11))

    def to_dsl(self) -> str:
        values = (*self.c00, *self.c10, *self.c01, *self.c11)
        return "bilinear_rgb(" + ",".join(_number(value) for value in values) + ")"


@dataclass(frozen=True)
class RadialRGB:
    """An additive, continuous Gaussian lighting field."""

    cx: float
    cy: float
    sigma: float
    amplitude: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("radial sigma must be positive")
        object.__setattr__(self, "amplitude", _rgb_tuple(self.amplitude, "amplitude"))

    def render(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r2 = (x - self.cx).square() + (y - self.cy).square()
        envelope = torch.exp(-r2 / max(2.0 * self.sigma * self.sigma, _EPS)).unsqueeze(-1)
        amplitude = torch.tensor(self.amplitude, device=x.device, dtype=x.dtype)
        return envelope * amplitude

    def model_bits(self) -> float:
        return (
            6.0
            + _scalar_bits(self.cx, 1.0 / 128.0)
            + _scalar_bits(self.cy, 1.0 / 128.0)
            + _scalar_bits(self.sigma, 1.0 / 128.0)
            + _rgb_bits(self.amplitude)
        )

    def to_dsl(self) -> str:
        values = (self.cx, self.cy, self.sigma, *self.amplitude)
        return "radial_rgb(" + ",".join(_number(value) for value in values) + ")"


@dataclass(frozen=True)
class AffineRGB:
    """Layer-local RGB value plus a continuous two-dimensional colour gradient."""

    value: Tuple[float, float, float]
    gradient_x: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gradient_y: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _rgb_tuple(self.value, "value"))
        object.__setattr__(self, "gradient_x", _rgb_tuple(self.gradient_x, "gradient_x"))
        object.__setattr__(self, "gradient_y", _rgb_tuple(self.gradient_y, "gradient_y"))

    def render(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cx: float | torch.Tensor,
        cy: float | torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        value = torch.tensor(self.value, device=device, dtype=dtype)
        gradient_x = torch.tensor(self.gradient_x, device=device, dtype=dtype)
        gradient_y = torch.tensor(self.gradient_y, device=device, dtype=dtype)
        return value + (x - cx).unsqueeze(-1) * gradient_x + (y - cy).unsqueeze(-1) * gradient_y

    def model_bits(self) -> float:
        return 4.0 + _rgb_bits((*self.value, *self.gradient_x, *self.gradient_y))

    def to_dsl(self) -> str:
        values = (*self.value, *self.gradient_x, *self.gradient_y)
        return "affine_rgb(" + ",".join(_number(value) for value in values) + ")"


@dataclass(frozen=True)
class ColourSDFLayer:
    """A typed SDF atom with a layer-local continuous RGB affine field."""

    primitive: Primitive
    colour: AffineRGB

    def model_bits(self) -> float:
        return self.primitive.model_bits() + self.colour.model_bits() + 2.0

    def to_dsl(self) -> str:
        return f"layer({self.primitive.to_dsl()},{self.colour.to_dsl()})"


@dataclass
class MultiScaleProgram:
    """A coordinate-continuous RGB scene program with smooth SDF blending."""

    background: BilinearRGB
    radials: List[RadialRGB] = field(default_factory=list)
    layers: List[ColourSDFLayer] = field(default_factory=list)
    edge_pixels: float = 1.5
    smin_pixels: float = 3.0

    def _render_background(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        colour = self.background.render(x, y)
        for radial in self.radials:
            colour = colour + radial.render(x, y)
        return colour

    def render(
        self,
        height: int,
        width: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Render ``[H,W,3]`` RGB in ``[0,1]`` using no raster payload."""
        x, y = normalized_grid(height, width, device)
        return _render_program(self, x, y)

    def model_bits(self) -> float:
        # Blend temperatures are global scalar grammar parameters, not repeated
        # on every layer.
        return (
            self.background.model_bits()
            + sum(radial.model_bits() for radial in self.radials)
            + sum(layer.model_bits() for layer in self.layers)
            + 5.0
            + _scalar_bits(self.edge_pixels, 0.25)
            + _scalar_bits(self.smin_pixels, 0.25)
        )

    def serialize(self) -> str:
        """Emit a continuous scene DSL; it contains no target-grid data."""
        parts = [self.background.to_dsl(), *(radial.to_dsl() for radial in self.radials)]
        if not self.layers:
            return "p=(2*x-1,2*y-1);rgb=" + "+".join(parts)
        fields = ",".join(layer.primitive.to_dsl() for layer in self.layers)
        colours = ",".join(layer.colour.to_dsl() for layer in self.layers)
        return (
            "p=(2*x-1,2*y-1);base="
            + "+".join(parts)
            + ";d=smin("
            + fields
            + ",k="
            + _number(self.smin_pixels)
            + ");rgb=sdf_colour(base,d,["
            + colours
            + "],tau="
            + _number(self.edge_pixels)
            + ")"
        )

    def to_current_evaluator_expression(self, height: int, width: int) -> str:
        """Emit an executable expression for the upgraded ``GridEvaluator``.

        The expression preserves this IR's RGB bilinear background, additive
        Gaussian lighting, soft-SDF union, softmax colour weights, and smooth
        coverage.  It uses the evaluator's typed SDF and shader calls rather
        than flattening any sampled image data.  Source references normalized
        geometry as ``2*x-1, 2*y-1`` because evaluator coordinates are [0, 1].
        """
        if height < 2 or width < 2:
            raise ValueError("an evaluator expression needs a canvas of at least 2x2")
        px, py = "(2*x-1)", "(2*y-1)"
        background = _evaluator_bilinear_rgb(self.background)
        for radial in self.radials:
            background = f"({background}+{_evaluator_radial_rgb(radial, px, py)})"
        if not self.layers:
            return background

        fields = [_evaluator_primitive(layer.primitive, px, py) for layer in self.layers]
        blend = _normalised_softness(self.smin_pixels, torch.empty((height, width)))
        edge = _normalised_softness(self.edge_pixels, torch.empty((height, width)))
        sharpness = 1.0 / blend
        pairs = []
        for field, layer in zip(fields, self.layers):
            colour = "rgb(" + ",".join(
                _evaluator_affine_channel(layer, channel, px, py) for channel in range(3)
            ) + ")"
            pairs.extend((field, colour))
        # Keeping distances and RGB fields as typed children of one AST node
        # gives the compiled source the same sharing as the program IR.
        return "sdfcolour(" + ",".join((
            background, _number(sharpness), _number(edge), *pairs,
        )) + ")"

    @property
    def source_length(self) -> int:
        return len(self.serialize())

    @property
    def is_coordinate_continuous(self) -> bool:
        """A guard for callers: this IR contains scalar fields, never a texture."""
        return True


@dataclass(frozen=True)
class MultiScaleFitConfig:
    """Bounded, deterministic fitting and rate-distortion policy.

    ``pyramid_long_sides`` selects *analysis* sample resolutions.  It is not
    encoded by the final program.  ``max_layers`` and the bit/source budgets
    prevent a photo from being approximated by a stealth per-pixel program.
    """

    pyramid_long_sides: Tuple[int, ...] = (32, 48, 72)
    layers_per_scale: int = 2
    max_layers: int = 6
    max_radials: int = 1
    refine_steps: int = 28
    learning_rate: float = 0.045
    edge_pixels: float = 1.5
    smin_pixels: float = 3.0
    noise_sigma: float = 0.045
    model_weight: float = 1.0
    acceptance_margin_bits: float = 8.0
    min_validation_data_gain_bits: float = 3.0
    min_validation_psnr_gain: float = 0.025
    min_validation_payback: float = 0.08
    max_model_bits: float = 3000.0
    max_source_length: int = 6000
    device: Optional[str] = None


@dataclass(frozen=True)
class AcceptedLayer:
    """Audit record for one accepted rate-distortion decision."""

    scale: Tuple[int, int]
    kind: str
    train_objective_gain_bits: float
    validation_data_gain_bits: float
    validation_psnr_gain: float
    added_model_bits: float


@dataclass
class MultiScaleFitResult:
    """Program, sampled verification render, and honest stopping metadata."""

    program: MultiScaleProgram
    reconstruction: torch.Tensor
    target: torch.Tensor
    train_data_bits: float
    validation_data_bits: float
    model_bits: float
    accepted_layers: List[AcceptedLayer]
    stop_reason: str

    @property
    def source_length(self) -> int:
        return self.program.source_length

    @property
    def psnr(self) -> float:
        return psnr(self.reconstruction, self.target)

    @property
    def ssim(self) -> float:
        return ssim(self.reconstruction, self.target)

    @property
    def perceptual_proxy(self) -> float:
        return multiscale_edge_error(self.reconstruction, self.target)


def _to_rgb_image(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Validate/normalize an RGB sample image to float32 ``[H,W,3]``."""
    if isinstance(image, np.ndarray) and not image.flags.writeable:
        # Pillow exposes some arrays as read-only views.  The model never writes
        # source samples, but PyTorch rightfully warns that sharing such storage
        # would make any accidental write undefined.
        image = image.copy()
    value = torch.as_tensor(image).detach()
    if value.ndim != 3:
        raise ValueError("image must be [H,W,3] or [3,H,W]")
    if value.shape[-1] not in (3, 4) and value.shape[0] in (3, 4):
        value = value.permute(1, 2, 0)
    if value.shape[-1] not in (3, 4):
        raise ValueError("image must have three RGB channels (or four RGBA channels)")
    value = value[..., :3].to(device=device, dtype=torch.float32)
    if value.numel() == 0:
        raise ValueError("image may not be empty")
    if float(value.max().detach().cpu()) > 1.0:
        value = value / 255.0
    return value.clamp(0.0, 1.0)


def _pyramid(image: torch.Tensor, long_sides: Sequence[int]) -> List[torch.Tensor]:
    """Antialiased low-pass analysis levels, ordered coarse to fine."""
    height, width, _ = image.shape
    longest = max(height, width)
    requested = sorted({max(8, min(int(size), longest)) for size in long_sides})
    if longest not in requested:
        requested.append(longest)
    levels: List[torch.Tensor] = []
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    for side in requested:
        scale = side / longest
        out_h = max(8, int(round(height * scale)))
        out_w = max(8, int(round(width * scale)))
        if (out_h, out_w) == (height, width):
            levels.append(image)
        else:
            sampled = F.interpolate(
                nchw, size=(out_h, out_w), mode="bilinear", align_corners=True, antialias=True
            )
            levels.append(sampled[0].permute(1, 2, 0))
    return levels


def _bilinear_least_squares(target: torch.Tensor) -> BilinearRGB:
    height, width, _ = target.shape
    x, y = normalized_grid(height, width, target.device)
    u, v = (x + 1.0) * 0.5, (y + 1.0) * 0.5
    features = torch.stack((torch.ones_like(u), u, v, u * v), dim=-1).reshape(-1, 4)
    coefficients = torch.linalg.lstsq(features, target.reshape(-1, 3)).solution
    c00 = coefficients[0]
    c10 = coefficients[0] + coefficients[1]
    c01 = coefficients[0] + coefficients[2]
    c11 = coefficients[0] + coefficients[1] + coefficients[2] + coefficients[3]
    return BilinearRGB(
        tuple(float(value.detach().cpu()) for value in c00),
        tuple(float(value.detach().cpu()) for value in c10),
        tuple(float(value.detach().cpu()) for value in c01),
        tuple(float(value.detach().cpu()) for value in c11),
    )


def _normalised_softness(pixels: float, x: torch.Tensor) -> float:
    height, width = x.shape
    return max(float(pixels) * 2.0 / max(height - 1, width - 1, 1), 1e-4)


def _evaluator_bilinear_rgb(field: BilinearRGB) -> str:
    """Build the RGB evaluator form of a continuous bilinear colour field."""
    channels = []
    for channel in range(3):
        corners = (field.c00[channel], field.c10[channel], field.c01[channel], field.c11[channel])
        channels.append("bilinear(x,y," + ",".join(_number(value) for value in corners) + ")")
    return "rgb(" + ",".join(channels) + ")"


def _evaluator_radial_rgb(field: RadialRGB, px: str, py: str) -> str:
    """Build one additive normalized-coordinate Gaussian RGB lighting term."""
    sigma2 = max(2.0 * field.sigma * field.sigma, _EPS)
    envelope = (
        "exp(-((({px})-({cx}))^2+(({py})-({cy}))^2)/({sigma2}))"
    ).format(px=px, py=py, cx=_number(field.cx), cy=_number(field.cy), sigma2=_number(sigma2))
    return "rgb(" + ",".join(
        f"({_number(amplitude)})*({envelope})" for amplitude in field.amplitude
    ) + ")"


def _evaluator_primitive(primitive: Primitive, px: str, py: str) -> str:
    """Serialize the restricted multi-scale geometry dictionary compactly."""
    values = primitive.values
    if primitive.kind == "disk":
        return "disk(" + ",".join((
            px, py, _number(values["cx"]), _number(values["cy"]), _number(values["radius"]),
        )) + ")"
    if primitive.kind == "superellipse":
        return "superellipse(" + ",".join((
            px, py, _number(values["cx"]), _number(values["cy"]), _number(values["rx"]),
            _number(values["ry"]), _number(values["power"]), _number(values["theta"]),
        )) + ")"
    raise ValueError(f"multi-scale evaluator serialization does not support {primitive.kind!r}")


def _evaluator_affine_channel(layer: ColourSDFLayer, channel: int, px: str, py: str) -> str:
    """Serialize ``value + gx*(x-cx) + gy*(y-cy)`` for one local RGB channel."""
    primitive, colour = layer.primitive, layer.colour
    gx, gy = colour.gradient_x[channel], colour.gradient_y[channel]
    bias = colour.value[channel] - gx * primitive.values["cx"] - gy * primitive.values["cy"]
    return "affine(" + ",".join((
        px, py, _number(bias), _number(gx), _number(gy),
    )) + ")"


def _render_program(program: MultiScaleProgram, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    background = program._render_background(x, y)
    if not program.layers:
        return background.clamp(0.0, 1.0)
    fields = torch.stack([layer.primitive.sdf(x, y) for layer in program.layers], dim=0)
    colour_fields = torch.stack(
        [
            layer.colour.render(
                x, y, layer.primitive.values["cx"], layer.primitive.values["cy"]
            )
            for layer in program.layers
        ],
        dim=0,
    )
    blend = _normalised_softness(program.smin_pixels, x)
    edge = _normalised_softness(program.edge_pixels, x)
    weights = torch.softmax(-fields / blend, dim=0)
    union_distance = -blend * torch.logsumexp(-fields / blend, dim=0)
    alpha = torch.sigmoid(-union_distance / edge).unsqueeze(-1)
    layer_colour = (weights.unsqueeze(-1) * colour_fields).sum(dim=0)
    return ((1.0 - alpha) * background + alpha * layer_colour).clamp(0.0, 1.0)


def _sdf(kind: str, values: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable subset of the geometry vocabulary used in this stage."""
    cx, cy = values["cx"], values["cy"]
    dx, dy = x - cx, y - cy
    theta = values.get("theta", torch.zeros((), device=x.device, dtype=x.dtype))
    cosine, sine = torch.cos(theta), torch.sin(theta)
    qx, qy = cosine * dx + sine * dy, -sine * dx + cosine * dy
    if kind == "disk":
        return torch.sqrt(qx.square() + qy.square() + _EPS) - values["radius"].clamp_min(_EPS)
    if kind == "superellipse":
        rx = values["rx"].clamp_min(_EPS)
        ry = values["ry"].clamp_min(_EPS)
        power = values["power"].clamp(1.05, 8.0)
        minimum = torch.minimum(rx, ry)
        normal = (qx.abs() / rx).pow(power) + (qy.abs() / ry).pow(power)
        return normal.pow(1.0 / power) * minimum - minimum
    raise ValueError(f"multi-scale stage does not support candidate kind {kind!r}")


def _candidate_colour(
    target: torch.Tensor,
    alpha: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cx: float,
    cy: float,
) -> AffineRGB:
    """Weighted affine RGB initialization under a candidate's soft coverage."""
    design = torch.stack((torch.ones_like(x), x - cx, y - cy), dim=-1).reshape(-1, 3)
    weights = alpha.reshape(-1).clamp_min(1e-4).sqrt().unsqueeze(-1)
    lhs = design * weights
    rhs = target.reshape(-1, 3) * weights
    coefficients = torch.linalg.lstsq(lhs, rhs).solution
    return AffineRGB(
        tuple(float(value.detach().cpu()) for value in coefficients[0]),
        tuple(float(value.detach().cpu()) for value in coefficients[1]),
        tuple(float(value.detach().cpu()) for value in coefficients[2]),
    )


def _colour_from_tensors(
    values: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    base = values["colour"].reshape(1, 1, 3)
    gx = values["gradient_x"].reshape(1, 1, 3)
    gy = values["gradient_y"].reshape(1, 1, 3)
    return base + (x - values["cx"]).unsqueeze(-1) * gx + (y - values["cy"]).unsqueeze(-1) * gy


def _render_with_candidate(
    program: MultiScaleProgram,
    kind: str,
    values: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    background = program._render_background(x, y)
    fields = [layer.primitive.sdf(x, y) for layer in program.layers]
    colours = [
        layer.colour.render(x, y, layer.primitive.values["cx"], layer.primitive.values["cy"])
        for layer in program.layers
    ]
    fields.append(_sdf(kind, values, x, y))
    colours.append(_colour_from_tensors(values, x, y))
    field_stack = torch.stack(fields, dim=0)
    colour_stack = torch.stack(colours, dim=0)
    blend = _normalised_softness(program.smin_pixels, x)
    edge = _normalised_softness(program.edge_pixels, x)
    weights = torch.softmax(-field_stack / blend, dim=0)
    union_distance = -blend * torch.logsumexp(-field_stack / blend, dim=0)
    alpha = torch.sigmoid(-union_distance / edge).unsqueeze(-1)
    colour = (weights.unsqueeze(-1) * colour_stack).sum(dim=0)
    return ((1.0 - alpha) * background + alpha * colour).clamp(0.0, 1.0)


def _spatial_split(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """A fixed held-out spatial subset; it is validation only, never output."""
    rows = torch.arange(height, device=device).view(-1, 1)
    cols = torch.arange(width, device=device).view(1, -1)
    validation = ((17 * rows + 31 * cols) % 7) == 0
    return ~validation, validation


def _data_bits(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sigma: float) -> float:
    diff = (render - target).square()
    squared_error = float((diff * mask.unsqueeze(-1)).sum().detach().cpu())
    observations = int(mask.sum().detach().cpu()) * 3
    if observations == 0:
        return 0.0
    variance = max(float(sigma) ** 2, 1e-8)
    return observations * 0.5 * math.log2(2.0 * math.pi * variance) + squared_error / (2.0 * variance * math.log(2.0))


def _psnr_mask(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    denom = float(mask.sum().detach().cpu()) * 3.0
    if denom <= 0:
        return float("inf")
    mse = float((((render - target).square() * mask.unsqueeze(-1)).sum() / denom).detach().cpu())
    return -10.0 * math.log10(max(mse, 1e-12))


def psnr(render: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR for benchmark samples in the normalized [0,1] RGB domain."""
    if render.shape != target.shape:
        raise ValueError("PSNR inputs must share shape")
    mask = torch.ones(render.shape[:2], dtype=torch.bool, device=render.device)
    return _psnr_mask(render, target.to(render.device), mask)


def ssim(render: torch.Tensor, target: torch.Tensor) -> float:
    """Small dependency-free RGB SSIM implementation for benchmark reporting."""
    if render.shape != target.shape:
        raise ValueError("SSIM inputs must share shape")
    first = render.permute(2, 0, 1).unsqueeze(0)
    second = target.to(render.device).permute(2, 0, 1).unsqueeze(0)
    kernel = min(11, render.shape[0] if render.shape[0] % 2 else render.shape[0] - 1,
                 render.shape[1] if render.shape[1] % 2 else render.shape[1] - 1)
    kernel = max(kernel, 1)
    padding = kernel // 2
    mu_first = F.avg_pool2d(first, kernel, stride=1, padding=padding, count_include_pad=False)
    mu_second = F.avg_pool2d(second, kernel, stride=1, padding=padding, count_include_pad=False)
    variance_first = F.avg_pool2d(first.square(), kernel, stride=1, padding=padding, count_include_pad=False) - mu_first.square()
    variance_second = F.avg_pool2d(second.square(), kernel, stride=1, padding=padding, count_include_pad=False) - mu_second.square()
    covariance = F.avg_pool2d(first * second, kernel, stride=1, padding=padding, count_include_pad=False) - mu_first * mu_second
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_first * mu_second + c1) * (2.0 * covariance + c2)) / (
        (mu_first.square() + mu_second.square() + c1) * (variance_first + variance_second + c2) + _EPS
    )
    return float(score.mean().detach().cpu())


def multiscale_edge_error(render: torch.Tensor, target: torch.Tensor) -> float:
    """A differentiable perceptual proxy: average multi-scale RGB edge L1."""
    if render.shape != target.shape:
        raise ValueError("edge-error inputs must share shape")
    first = render.permute(2, 0, 1).unsqueeze(0)
    second = target.to(render.device).permute(2, 0, 1).unsqueeze(0)
    errors: List[torch.Tensor] = []
    for _ in range(3):
        dx_first, dy_first = first[..., :, 1:] - first[..., :, :-1], first[..., 1:, :] - first[..., :-1, :]
        dx_second, dy_second = second[..., :, 1:] - second[..., :, :-1], second[..., 1:, :] - second[..., :-1, :]
        errors.append((dx_first - dx_second).abs().mean() + (dy_first - dy_second).abs().mean())
        if min(first.shape[-2:]) < 8:
            break
        first = F.avg_pool2d(first, kernel_size=2, stride=2)
        second = F.avg_pool2d(second, kernel_size=2, stride=2)
    return float(torch.stack(errors).mean().detach().cpu())


def _background_radial_proposal(program: MultiScaleProgram, target: torch.Tensor) -> RadialRGB:
    """Analytically propose one broad residual lighting field."""
    height, width, _ = target.shape
    x, y = normalized_grid(height, width, target.device)
    residual = target - _render_program(program, x, y)
    energy = residual.square().mean(dim=-1).clamp_min(1e-8)
    total = energy.sum()
    cx = (energy * x).sum() / total
    cy = (energy * y).sum() / total
    variance = (energy * ((x - cx).square() + (y - cy).square())).sum() / total
    sigma = torch.sqrt(variance.clamp_min(0.03 ** 2)).clamp(0.12, 1.5)
    envelope = torch.exp(-((x - cx).square() + (y - cy).square()) / (2.0 * sigma.square())).unsqueeze(-1)
    amplitude = (envelope * residual).sum(dim=(0, 1)) / envelope.square().sum().clamp_min(_EPS)
    return RadialRGB(
        float(cx.detach().cpu()),
        float(cy.detach().cpu()),
        float(sigma.detach().cpu()),
        tuple(float(value.detach().cpu()) for value in amplitude.clamp(-0.85, 0.85)),
    )


def _candidate_values(primitive: Primitive, colour: AffineRGB, device: torch.device) -> Dict[str, torch.Tensor]:
    values: Dict[str, torch.Tensor] = {
        key: torch.tensor(float(value), device=device, dtype=torch.float32, requires_grad=True)
        for key, value in primitive.values.items()
    }
    values["colour"] = torch.tensor(colour.value, device=device, dtype=torch.float32, requires_grad=True)
    values["gradient_x"] = torch.tensor(colour.gradient_x, device=device, dtype=torch.float32, requires_grad=True)
    values["gradient_y"] = torch.tensor(colour.gradient_y, device=device, dtype=torch.float32, requires_grad=True)
    return values


def _project_candidate(values: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in values.items():
            if name in {"cx", "cy"}:
                value.clamp_(-1.25, 1.25)
            elif name in {"radius", "rx", "ry"}:
                value.clamp_(0.015, 2.2)
            elif name == "power":
                value.clamp_(1.05, 7.0)
            elif name == "theta":
                value.remainder_(2.0 * math.pi)
            elif name in {"colour", "gradient_x", "gradient_y"}:
                value.clamp_(-1.5, 1.5)


def _layer_from_tensors(kind: str, values: Mapping[str, torch.Tensor]) -> ColourSDFLayer:
    geometry_keys = ("cx", "cy", "theta", "radius") if kind == "disk" else (
        "cx", "cy", "theta", "rx", "ry", "power"
    )
    geometry = {key: float(values[key].detach().cpu()) for key in geometry_keys}
    colour = AffineRGB(
        tuple(float(value) for value in values["colour"].detach().cpu().tolist()),
        tuple(float(value) for value in values["gradient_x"].detach().cpu().tolist()),
        tuple(float(value) for value in values["gradient_y"].detach().cpu().tolist()),
    )
    return ColourSDFLayer(Primitive(kind, geometry), colour)


def _refine_layer(
    program: MultiScaleProgram,
    proposal: Primitive,
    target: torch.Tensor,
    train_mask: torch.Tensor,
    config: MultiScaleFitConfig,
) -> Tuple[ColourSDFLayer, torch.Tensor]:
    """Jointly refine geometry and local RGB affine field through soft SDFs."""
    height, width, _ = target.shape
    x, y = normalized_grid(height, width, target.device)
    edge = _normalised_softness(program.edge_pixels, x)
    seed_alpha = torch.sigmoid(-proposal.sdf(x, y) / edge)
    seed_colour = _candidate_colour(
        target, seed_alpha, x, y, float(proposal.values["cx"]), float(proposal.values["cy"])
    )
    values = _candidate_values(proposal, seed_colour, target.device)
    optimizer = torch.optim.Adam(values.values(), lr=config.learning_rate)
    train_weight = train_mask.unsqueeze(-1).to(target.dtype)
    best_loss = float("inf")
    best_values: Optional[Dict[str, torch.Tensor]] = None
    for _ in range(config.refine_steps):
        optimizer.zero_grad(set_to_none=True)
        rendered = _render_with_candidate(program, proposal.kind, values, x, y)
        loss = ((rendered - target).square() * train_weight).sum() / train_weight.sum().clamp_min(1.0) / 3.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(values.values()), max_norm=5.0)
        optimizer.step()
        _project_candidate(values)
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_values = {name: value.detach().clone() for name, value in values.items()}
    assert best_values is not None
    layer = _layer_from_tensors(proposal.kind, best_values)
    with torch.no_grad():
        rendered = _render_with_candidate(program, proposal.kind, best_values, x, y)
    return layer, rendered.detach()


def _proposals(
    program: MultiScaleProgram,
    target: torch.Tensor,
    scale_index: int,
    scale_count: int,
) -> List[Primitive]:
    """Derive a small deterministic broad-then-local residual dictionary."""
    height, width, _ = target.shape
    x, y = normalized_grid(height, width, target.device)
    with torch.no_grad():
        residual = (target - _render_program(program, x, y)).square().mean(dim=-1)
        # Modest spatial smoothing avoids anchoring a macro atom to one noisy
        # pixel while preserving the continuous-field output invariant.
        smooth = F.avg_pool2d(residual[None, None], kernel_size=5, stride=1, padding=2)[0, 0]
        index = int(smooth.reshape(-1).argmax().detach().cpu())
        row, col = divmod(index, width)
        cx, cy = x[row, col], y[row, col]
        # Coarse stages see a wider physical support; later stages focus on the
        # remaining local Laplacian residual.
        local_sigma = 0.62 * (0.62 ** scale_index)
        local_sigma = max(local_sigma, 0.10)
        window = torch.exp(-((x - cx).square() + (y - cy).square()) / (2.0 * local_sigma ** 2))
        weights = (residual * window).clamp_min(1e-10)
        total = weights.sum()
        mx, my = (weights * x).sum() / total, (weights * y).sum() / total
        dx, dy = x - mx, y - my
        covariance = torch.stack(
            (
                torch.stack(((weights * dx.square()).sum() / total, (weights * dx * dy).sum() / total)),
                torch.stack(((weights * dx * dy).sum() / total, (weights * dy.square()).sum() / total)),
            )
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        axis = eigenvectors[:, 1]
        theta = torch.atan2(axis[1], axis[0])
        long_radius = (2.45 * torch.sqrt(eigenvalues[1].clamp_min(0.02 ** 2))).clamp(0.045, 1.25)
        short_radius = (2.45 * torch.sqrt(eigenvalues[0].clamp_min(0.02 ** 2))).clamp(0.04, 1.20)
        area_radius = torch.sqrt(long_radius * short_radius).clamp(0.04, 1.2)
        values = [float(value.detach().cpu()) for value in (mx, my, long_radius, short_radius, theta, area_radius)]
    centre_x, centre_y, rx, ry, angle, radius = values
    return [
        Primitive.superellipse(centre_x, centre_y, rx, ry, power=2.0, theta=angle),
        Primitive.superellipse(centre_x, centre_y, rx, ry, power=4.0, theta=angle),
        Primitive.disk(centre_x, centre_y, radius),
    ]


def _score_program(
    program: MultiScaleProgram,
    target: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    config: MultiScaleFitConfig,
) -> Tuple[torch.Tensor, float, float, float, float]:
    height, width, _ = target.shape
    rendered = program.render(height, width, target.device)
    train_data = _data_bits(rendered, target, train_mask, config.noise_sigma)
    validation_data = _data_bits(rendered, target, validation_mask, config.noise_sigma)
    model = program.model_bits()
    return rendered, train_data, validation_data, train_data + config.model_weight * model, _psnr_mask(rendered, target, validation_mask)


def _accept(
    current_train_objective: float,
    current_validation_data: float,
    current_validation_psnr: float,
    candidate_program: MultiScaleProgram,
    candidate_train_data: float,
    candidate_validation_data: float,
    candidate_validation_psnr: float,
    config: MultiScaleFitConfig,
) -> Tuple[bool, float, float, float, float]:
    candidate_model = candidate_program.model_bits()
    candidate_objective = candidate_train_data + config.model_weight * candidate_model
    train_gain = current_train_objective - candidate_objective
    validation_gain = current_validation_data - candidate_validation_data
    psnr_gain = candidate_validation_psnr - current_validation_psnr
    return (
        train_gain > config.acceptance_margin_bits
        and validation_gain > config.min_validation_data_gain_bits
        and psnr_gain > config.min_validation_psnr_gain,
        train_gain,
        validation_gain,
        psnr_gain,
        candidate_objective,
    )


def fit_multiscale_image(
    image: np.ndarray | torch.Tensor,
    config: Optional[MultiScaleFitConfig] = None,
) -> MultiScaleFitResult:
    """Fit a bounded continuous-colour program to sampled RGB image data.

    The returned model is an approximation.  ``stop_reason`` identifies whether
    a layer budget, source/model budget, or held-out rate-distortion gate stopped
    the search.  Callers must use that signal and final benchmark metrics when
    choosing between a symbolic result and another codec.
    """
    cfg = config or MultiScaleFitConfig()
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_full = _to_rgb_image(image, device)
    pyramid = _pyramid(target_full, cfg.pyramid_long_sides)
    background = _bilinear_least_squares(pyramid[0])
    program = MultiScaleProgram(background, edge_pixels=cfg.edge_pixels, smin_pixels=cfg.smin_pixels)
    accepted: List[AcceptedLayer] = []
    stop_reason = "pyramid_complete"

    # A radial field has its own MDL/validation acceptance rather than being a
    # free colour correction.  It is deliberately broad and bounded in count.
    first_level = pyramid[0]
    train_mask, validation_mask = _spatial_split(first_level.shape[0], first_level.shape[1], device)
    current_render, train_data, validation_data, current_objective, validation_psnr = _score_program(
        program, first_level, train_mask, validation_mask, cfg
    )
    del current_render
    for _ in range(cfg.max_radials):
        radial = _background_radial_proposal(program, first_level)
        candidate = MultiScaleProgram(
            program.background, [*program.radials, radial], list(program.layers), cfg.edge_pixels, cfg.smin_pixels
        )
        if candidate.model_bits() > cfg.max_model_bits or candidate.source_length > cfg.max_source_length:
            stop_reason = "budget"
            break
        _, candidate_train, candidate_validation, _, candidate_psnr = _score_program(
            candidate, first_level, train_mask, validation_mask, cfg
        )
        accepted_radial, train_gain, validation_gain, psnr_gain, candidate_objective = _accept(
            current_objective, validation_data, validation_psnr, candidate,
            candidate_train, candidate_validation, candidate_psnr, cfg,
        )
        added_bits = candidate.model_bits() - program.model_bits()
        if not accepted_radial or validation_gain < cfg.min_validation_payback * added_bits:
            break
        program = candidate
        train_data, validation_data, current_objective, validation_psnr = (
            candidate_train, candidate_validation, candidate_objective, candidate_psnr
        )

    terminated = stop_reason == "budget"
    for level_index, level in enumerate(pyramid):
        if terminated:
            break
        train_mask, validation_mask = _spatial_split(level.shape[0], level.shape[1], device)
        _, train_data, validation_data, current_objective, validation_psnr = _score_program(
            program, level, train_mask, validation_mask, cfg
        )
        for _ in range(cfg.layers_per_scale):
            if len(program.layers) >= cfg.max_layers:
                stop_reason = "layer_budget"
                terminated = True
                break
            proposals = _proposals(program, level, level_index, len(pyramid))
            ranked: List[Tuple[ColourSDFLayer, float, float, float, float, float]] = []
            # All proposals are optimised from deterministic analytic moments.
            for proposal in proposals:
                layer, rendered = _refine_layer(program, proposal, level, train_mask, cfg)
                candidate = MultiScaleProgram(
                    program.background, list(program.radials), [*program.layers, layer],
                    cfg.edge_pixels, cfg.smin_pixels,
                )
                if candidate.model_bits() > cfg.max_model_bits or candidate.source_length > cfg.max_source_length:
                    continue
                candidate_train = _data_bits(rendered, level, train_mask, cfg.noise_sigma)
                candidate_validation = _data_bits(rendered, level, validation_mask, cfg.noise_sigma)
                candidate_psnr = _psnr_mask(rendered, level, validation_mask)
                candidate_objective = candidate_train + cfg.model_weight * candidate.model_bits()
                del rendered
                ranked.append((
                    layer, candidate_objective, candidate_train, candidate_validation,
                    candidate_psnr, candidate.model_bits(),
                ))
            if not ranked:
                stop_reason = "budget"
                terminated = True
                break
            selected: Optional[Tuple[ColourSDFLayer, float, float, float, float, float, float, float, float]] = None
            for layer, candidate_objective, candidate_train, candidate_validation, candidate_psnr, candidate_bits in sorted(
                ranked, key=lambda item: item[1]
            ):
                candidate_program = MultiScaleProgram(
                    program.background, list(program.radials), [*program.layers, layer], cfg.edge_pixels, cfg.smin_pixels
                )
                added_bits = candidate_bits - program.model_bits()
                accepted_layer, train_gain, validation_gain, psnr_gain, _ = _accept(
                    current_objective, validation_data, validation_psnr, candidate_program,
                    candidate_train, candidate_validation, candidate_psnr, cfg,
                )
                if accepted_layer and validation_gain >= cfg.min_validation_payback * added_bits:
                    selected = (
                        layer, candidate_train, candidate_validation, candidate_psnr,
                        candidate_bits, added_bits, train_gain, validation_gain, psnr_gain,
                    )
                    break
            if selected is None:
                stop_reason = "no_rate_distortion_gain"
                terminated = True
                break
            (
                layer, candidate_train, candidate_validation, candidate_psnr,
                candidate_bits, added_bits, train_gain, validation_gain, psnr_gain,
            ) = selected
            candidate_program = MultiScaleProgram(
                program.background, list(program.radials), [*program.layers, layer], cfg.edge_pixels, cfg.smin_pixels
            )
            program = candidate_program
            accepted.append(
                AcceptedLayer(
                    scale=(level.shape[0], level.shape[1]),
                    kind=layer.primitive.kind,
                    train_objective_gain_bits=float(train_gain),
                    validation_data_gain_bits=float(validation_gain),
                    validation_psnr_gain=float(psnr_gain),
                    added_model_bits=float(added_bits),
                )
            )
            train_data, validation_data, current_objective, validation_psnr = (
                candidate_train, candidate_validation, candidate_objective, candidate_psnr
            )
        # A rejected candidate is an explicit RD stop, not a reason to spend
        # later-scale atoms on the same unprofitable residual.
        if terminated:
            break

    reconstruction = program.render(target_full.shape[0], target_full.shape[1], device).detach()
    train_mask, validation_mask = _spatial_split(target_full.shape[0], target_full.shape[1], device)
    train_data = _data_bits(reconstruction, target_full, train_mask, cfg.noise_sigma)
    validation_data = _data_bits(reconstruction, target_full, validation_mask, cfg.noise_sigma)
    return MultiScaleFitResult(
        program=program,
        reconstruction=reconstruction,
        target=target_full.detach(),
        train_data_bits=train_data,
        validation_data_bits=validation_data,
        model_bits=program.model_bits(),
        accepted_layers=accepted,
        stop_reason=stop_reason,
    )


def benchmark_image(
    image: np.ndarray | torch.Tensor,
    config: Optional[MultiScaleFitConfig] = None,
) -> Dict[str, Any]:
    """Fit an RGB image and return auditable rate/distortion benchmark values."""
    started = time.perf_counter()
    result = fit_multiscale_image(image, config)
    if result.reconstruction.is_cuda:
        torch.cuda.synchronize(result.reconstruction.device)
    elapsed = time.perf_counter() - started
    return {
        "equation": result.program.serialize(),
        "evaluator_equation": result.program.to_current_evaluator_expression(
            result.target.shape[0], result.target.shape[1]
        ),
        "source_length": result.source_length,
        "evaluator_source_length": len(
            result.program.to_current_evaluator_expression(result.target.shape[0], result.target.shape[1])
        ),
        "model_bits": result.model_bits,
        "train_data_bits": result.train_data_bits,
        "validation_data_bits": result.validation_data_bits,
        "psnr": result.psnr,
        "ssim": result.ssim,
        "perceptual_proxy": result.perceptual_proxy,
        "runtime_seconds": elapsed,
        "stop_reason": result.stop_reason,
        "accepted_layers": result.accepted_layers,
        "continuous_program": result.program.is_coordinate_continuous,
        "program": result.program,
    }


def synthetic_shaded_scene(size: int = 80, device: torch.device | str = "cpu") -> torch.Tensor:
    """A continuous RGB scene with bilinear lighting, radial glow, and two forms."""
    background = BilinearRGB(
        (0.06, 0.13, 0.25), (0.13, 0.25, 0.42), (0.03, 0.08, 0.16), (0.22, 0.38, 0.50)
    )
    radial = RadialRGB(-0.55, -0.48, 0.72, (0.18, 0.10, -0.02))
    warm = ColourSDFLayer(
        Primitive.superellipse(-0.20, 0.12, 0.55, 0.34, power=2.4, theta=-0.22),
        AffineRGB((0.88, 0.30, 0.14), (0.08, 0.12, 0.02), (-0.18, 0.04, 0.10)),
    )
    cool = ColourSDFLayer(
        Primitive.disk(0.42, -0.22, 0.22),
        AffineRGB((0.12, 0.80, 0.76), (-0.10, 0.04, 0.12), (0.03, -0.16, 0.02)),
    )
    program = MultiScaleProgram(background, [radial], [warm, cool], edge_pixels=1.5, smin_pixels=3.0)
    return program.render(size, size, device)


__all__ = [
    "AcceptedLayer",
    "AffineRGB",
    "BilinearRGB",
    "ColourSDFLayer",
    "MultiScaleFitConfig",
    "MultiScaleFitResult",
    "MultiScaleProgram",
    "RadialRGB",
    "benchmark_image",
    "fit_multiscale_image",
    "multiscale_edge_error",
    "psnr",
    "ssim",
    "synthetic_shaded_scene",
]
