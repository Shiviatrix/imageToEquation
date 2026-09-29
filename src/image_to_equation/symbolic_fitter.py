"""Deterministic, MDL-scored fitting of concise 2-D SDF programs.

This module is deliberately independent from the legacy string evaluator and
genetic algorithm.  Its internal representation is a small typed geometry DSL:
``disk``, transformed ``superellipse`` (an ellipse is ``p=2``), ``capsule``,
``wedge`` (an isosceles triangle), and ``rounded_box``.  A program is an ordered
alpha-union of primitives, which is both a useful opaque-layer model and a
compact CSG-union approximation.

Public API
==========
``fit_binary_mask(mask, config=None)``
    Fit a :class:`SymbolicProgram` to a binary mask or a flat-colour image.
``benchmark_mask(mask, config=None)``
    Fit a mask and return source length, IoU, MDL score, and elapsed time.
``synthetic_fish_mask(size=96)`` / ``benchmark_synthetic_fish(...)``
    A deterministic smoke benchmark for a low-description-length silhouette.

The search is not a GA.  At every round it extracts connected components from
the unexplained foreground, makes analytic geometry proposals from their bbox,
area, and second moments, locally refines each proposal with differentiable SDF
rasterisation, and accepts one only when a weighted Bernoulli data code plus an
explicit model code is improved.  There is no random seed or population.

Coordinates are normalized to [-1, 1]^2.  The serialized result is a readable
geometry DSL such as ``alpha=sigmoid(-sdf(union(...))/tau)``.  The fitting
representation stays independent of the string evaluator, but selected
programs can emit a complete compact expression for its SDF helpers through
``SymbolicProgram.to_current_evaluator_expression()``.  Evaluate that result
with ``GridEvaluator(..., render_mode="linear")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


_EPS = 1e-6
_KINDS = {"disk", "superellipse", "capsule", "wedge", "rounded_box"}
_GRAMMAR_BITS = {
    "disk": 5.0,
    "superellipse": 7.0,
    "capsule": 8.0,
    "wedge": 8.5,
    "rounded_box": 8.5,
}


def _number(value: float, digits: int = 4) -> str:
    """Compact, locale-independent scalar formatting for program source."""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def normalized_grid(
    height: int, width: int, device: torch.device | str | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``x, y`` grids spanning ``[-1, 1]`` with shape ``[H, W]``."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    xs = torch.linspace(-1.0, 1.0, width, device=target_device)
    ys = torch.linspace(-1.0, 1.0, height, device=target_device)
    return torch.meshgrid(xs, ys, indexing="xy")


def _tensor_value(
    values: Mapping[str, Any], name: str, reference: torch.Tensor
) -> torch.Tensor:
    value = values[name]
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(float(value), device=reference.device, dtype=reference.dtype)


def _local_coordinates(
    values: Mapping[str, Any], x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cx = _tensor_value(values, "cx", x)
    cy = _tensor_value(values, "cy", x)
    theta = _tensor_value(values, "theta", x)
    dx, dy = x - cx, y - cy
    cosine, sine = torch.cos(theta), torch.sin(theta)
    # R(-theta) (p-c): the long primitive axis is local x.
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _primitive_sdf(
    kind: str, values: Mapping[str, Any], x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Differentiable negative-inside field for one primitive."""
    if kind not in _KINDS:
        raise ValueError(f"Unsupported primitive kind {kind!r}")
    qx, qy = _local_coordinates(values, x, y)

    if kind == "disk":
        radius = _tensor_value(values, "radius", x).clamp_min(_EPS)
        return torch.sqrt(qx.square() + qy.square() + _EPS) - radius

    if kind == "superellipse":
        rx = _tensor_value(values, "rx", x).clamp_min(_EPS)
        ry = _tensor_value(values, "ry", x).clamp_min(_EPS)
        power = _tensor_value(values, "power", x).clamp(1.05, 8.0)
        normalized = (qx.abs() / rx).pow(power) + (qy.abs() / ry).pow(power)
        # Multiplying by the smaller radius makes this an SDF-like field while
        # retaining the exact p=2 ellipse boundary.
        return normalized.pow(1.0 / power) * torch.minimum(rx, ry) - torch.minimum(rx, ry)

    if kind == "capsule":
        half_length = _tensor_value(values, "half_length", x).clamp_min(0.0)
        radius = _tensor_value(values, "radius", x).clamp_min(_EPS)
        outside_x = (qx.abs() - half_length).clamp_min(0.0)
        return torch.sqrt(outside_x.square() + qy.square() + _EPS) - radius

    if kind == "wedge":
        length = _tensor_value(values, "length", x).clamp_min(_EPS)
        width = _tensor_value(values, "width", x).clamp_min(_EPS)
        # Triangle vertices in local coordinates are
        # (-L/2,-W/2), (-L/2,W/2), (L/2,0).  The max of its inward
        # half-plane fields is negative exactly inside the convex wedge.
        slope = width / (2.0 * length)
        diagonal_scale = torch.sqrt(1.0 + slope.square())
        planes = (
            -qx - length / 2.0,
            qx - length / 2.0,
            (qy - width / 4.0 + slope * qx) / diagonal_scale,
            (-qy - width / 4.0 + slope * qx) / diagonal_scale,
        )
        result = planes[0]
        for plane in planes[1:]:
            result = torch.maximum(result, plane)
        return result

    # Standard rounded-rectangle SDF. ``hx`` and ``hy`` are the half-lengths
    # of the unrounded central box; the outside radius is added around it.
    hx = _tensor_value(values, "hx", x).clamp_min(_EPS)
    hy = _tensor_value(values, "hy", x).clamp_min(_EPS)
    radius = _tensor_value(values, "radius", x).clamp_min(0.0)
    dx, dy = qx.abs() - hx, qy.abs() - hy
    outside = torch.sqrt(dx.clamp_min(0.0).square() + dy.clamp_min(0.0).square() + _EPS)
    inside = torch.minimum(torch.maximum(dx, dy), torch.zeros_like(dx))
    return outside + inside - radius


@dataclass(frozen=True)
class Primitive:
    """A scalar-parameterized SDF atom in the compact geometry dictionary."""

    kind: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        required = {
            "disk": {"cx", "cy", "theta", "radius"},
            "superellipse": {"cx", "cy", "theta", "rx", "ry", "power"},
            "capsule": {"cx", "cy", "theta", "half_length", "radius"},
            "wedge": {"cx", "cy", "theta", "length", "width"},
            "rounded_box": {"cx", "cy", "theta", "hx", "hy", "radius"},
        }[self.kind]
        missing = required.difference(self.values)
        if missing:
            raise ValueError(f"{self.kind} is missing parameters {sorted(missing)}")

    @classmethod
    def disk(cls, cx: float, cy: float, radius: float) -> "Primitive":
        return cls("disk", {"cx": cx, "cy": cy, "theta": 0.0, "radius": radius})

    @classmethod
    def superellipse(
        cls, cx: float, cy: float, rx: float, ry: float, power: float = 2.0, theta: float = 0.0
    ) -> "Primitive":
        return cls(
            "superellipse",
            {"cx": cx, "cy": cy, "theta": theta, "rx": rx, "ry": ry, "power": power},
        )

    @classmethod
    def capsule(
        cls, cx: float, cy: float, half_length: float, radius: float, theta: float = 0.0
    ) -> "Primitive":
        return cls(
            "capsule",
            {"cx": cx, "cy": cy, "theta": theta, "half_length": half_length, "radius": radius},
        )

    @classmethod
    def wedge(
        cls, cx: float, cy: float, length: float, width: float, theta: float = 0.0
    ) -> "Primitive":
        return cls(
            "wedge", {"cx": cx, "cy": cy, "theta": theta, "length": length, "width": width}
        )

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _primitive_sdf(self.kind, self.values, x, y)

    def alpha(self, x: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
        """Anti-aliased SDF occupancy, differentiable with respect to all values."""
        return torch.sigmoid(-self.sdf(x, y) / max(float(tau), _EPS))

    def model_bits(self) -> float:
        return _primitive_model_bits(self.kind, self.values)

    def to_dsl(self) -> str:
        values = self.values
        if self.kind == "disk":
            return "disk(" + ",".join(_number(values[k]) for k in ("cx", "cy", "radius")) + ")"
        if self.kind == "superellipse":
            return "superellipse(" + ",".join(
                _number(values[k]) for k in ("cx", "cy", "rx", "ry", "power", "theta")
            ) + ")"
        if self.kind == "capsule":
            return "capsule(" + ",".join(
                _number(values[k]) for k in ("cx", "cy", "half_length", "radius", "theta")
            ) + ")"
        if self.kind == "wedge":
            return "wedge(" + ",".join(
                _number(values[k]) for k in ("cx", "cy", "length", "width", "theta")
            ) + ")"
        return "rounded_box(" + ",".join(
            _number(values[k]) for k in ("cx", "cy", "hx", "hy", "radius", "theta")
        ) + ")"


def _scalar_bits(value: float, delta: float) -> float:
    """Operational universal code for a quantized scalar, in bits."""
    return 1.0 + 0.5 * math.log2(1.0 + (float(value) / delta) ** 2)


def _primitive_model_bits(kind: str, values: Mapping[str, float]) -> float:
    scale_delta = 1.0 / 128.0
    angle_delta = math.pi / 180.0
    bits = _GRAMMAR_BITS[kind] + 1.5  # one union/layer reference
    for name, value in values.items():
        delta = angle_delta if name == "theta" else scale_delta
        bits += _scalar_bits(float(value), delta)
    # The p=2 ellipse is a dictionary default, not an independently expensive
    # real parameter.  Other exponents retain an explicit code.
    if kind == "superellipse" and abs(float(values["power"]) - 2.0) < 0.05:
        bits -= _scalar_bits(float(values["power"]), scale_delta)
    return bits


@dataclass
class SymbolicProgram:
    """An alpha-union program with executable PyTorch SDF rendering."""

    primitives: List[Primitive] = field(default_factory=list)
    tau_pixels: float = 1.25

    def render(
        self,
        height: int,
        width: int,
        device: torch.device | str | None = None,
        tau_pixels: Optional[float] = None,
    ) -> torch.Tensor:
        x, y = normalized_grid(height, width, device)
        tau = (self.tau_pixels if tau_pixels is None else tau_pixels) * 2.0 / max(height - 1, width - 1, 1)
        alpha = torch.zeros_like(x)
        for primitive in self.primitives:
            atom = primitive.alpha(x, y, tau)
            alpha = alpha + (1.0 - alpha) * atom
        return alpha

    def model_bits(self) -> float:
        return sum(primitive.model_bits() for primitive in self.primitives)

    def serialize(self) -> str:
        """Emit a short, human-readable geometry-equation DSL program."""
        if not self.primitives:
            return "p=(2*x-1,2*y-1);alpha=0"
        body = ",".join(primitive.to_dsl() for primitive in self.primitives)
        return f"p=(2*x-1,2*y-1);alpha=sigmoid(-sdf(union({body}))/tau)"

    @property
    def source_length(self) -> int:
        return len(self.serialize())

    def to_current_evaluator_field(self, smooth_tau: float = 0.02) -> Optional[str]:
        """Return a compact, parseable implicit field for the current evaluator.

        Each typed SDF remains one AST call instead of expanding rotated
        coordinate algebra at every use.  This is a genuine description-length
        improvement: the compiler implements the same mathematical fields and
        combines them with a stable soft union.
        """
        expressions = [_current_field_expression(p, smooth_tau) for p in self.primitives]
        if not expressions or any(expression is None for expression in expressions):
            return None
        sharpness = _number(1.0 / max(smooth_tau, _EPS))
        result = expressions[0]
        for expression in expressions[1:]:
            result = f"smin({result},{expression},{sharpness})"
        return result

    def to_current_evaluator_expression(
        self,
        height: int,
        width: int,
        colour: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Emit a complete compact expression for ``GridEvaluator``.

        The output is an alpha mask made from the parser's ``sdfmask`` helper;
        use ``render_mode="linear"`` when evaluating it.  Supplying an RGB
        colour in ``[0, 1]`` multiplies the mask by the quadratic in ``v`` that
        exactly produces those channel values at ``v=0,1,2``.
        """
        softness = self.tau_pixels * 2.0 / max(height - 1, width - 1, 1)
        field = self.to_current_evaluator_field(softness)
        if field is None:
            return None
        alpha = f"sdfmask({field},{_number(softness)})"
        if colour is None:
            return alpha
        if len(colour) != 3:
            raise ValueError("colour must contain exactly three RGB values")
        red, green, blue = (float(value) for value in colour)
        if any(not 0.0 <= value <= 1.0 for value in (red, green, blue)):
            raise ValueError("colour values must lie in [0, 1]")
        # c(0)=R, c(1)=G, c(2)=B for c(v)=c0+c1*v+c2*v^2.
        c2 = (blue - 2.0 * green + red) / 2.0
        c1 = green - red - c2
        colour_expr = f"({_number(red)}+({_number(c1)})*v+({_number(c2)})*v^2)"
        return f"({alpha}*{colour_expr})"


@dataclass(frozen=True)
class FitConfig:
    """Fixed, geometry-derived fitting settings; no stochastic search knobs."""

    max_shapes: int = 6
    max_components: int = 3
    refine_steps: int = 48
    learning_rate: float = 0.045
    tau_pixels: float = 1.25
    foreground_weight: float = 4.0
    boundary_weight: float = 3.0
    model_weight: float = 1.0
    acceptance_margin_bits: float = 2.0
    min_component_pixels: int = 10
    device: Optional[str] = None


@dataclass
class FitResult:
    """Program plus the audited quantities used by greedy MDL acceptance."""

    program: SymbolicProgram
    alpha: torch.Tensor
    target: torch.Tensor
    data_bits: float
    model_bits: float
    mdl_bits: float
    accepted_improvements: List[float]

    @property
    def iou(self) -> float:
        return binary_iou(self.alpha, self.target)


def binary_iou(alpha: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    predicted = alpha.detach() >= threshold
    truth = target.detach() >= threshold
    union = torch.logical_or(predicted, truth).sum().item()
    if union == 0:
        return 1.0
    return float(torch.logical_and(predicted, truth).sum().item() / union)


def _to_binary_mask(mask: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a binary or flat-colour foreground image to ``float32 [H,W]``."""
    data = torch.as_tensor(mask).detach().cpu()
    if data.ndim == 2:
        values = data.float()
        if values.max() > 1.0:
            values = values / 255.0
        return (values >= 0.5).float().to(device)
    if data.ndim != 3:
        raise ValueError("mask must have shape [H,W], [H,W,C], or [C,H,W]")
    if data.shape[0] in (1, 3, 4) and data.shape[-1] not in (1, 3, 4):
        data = data.permute(1, 2, 0)
    if data.shape[-1] not in (1, 3, 4):
        raise ValueError("colour mask must have 1, 3, or 4 channels")
    values = data[..., :3].float()
    if values.max() > 1.0:
        values = values / 255.0
    h, w, _ = values.shape
    corners = torch.stack((values[0, 0], values[0, w - 1], values[h - 1, 0], values[h - 1, w - 1]))
    background = corners.median(dim=0).values
    distance = torch.sqrt((values - background).square().sum(dim=-1))
    # Flat colours need no learned segmentation.  The small tolerance rejects
    # exact/near-exact background while retaining a coloured foreground.
    return (distance > 0.04).float().to(device)


def _weight_map(target: torch.Tensor, config: FitConfig) -> torch.Tensor:
    """Foreground and narrow boundary weighting for the Bernoulli code."""
    value = target[None, None]
    dilated = F.max_pool2d(value, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - value, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    boundary_band = F.max_pool2d(boundary, kernel_size=5, stride=1, padding=2)[0, 0]
    return 1.0 + config.foreground_weight * target + config.boundary_weight * boundary_band


def _data_bits(alpha: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    clipped = alpha.clamp(1e-5, 1.0 - 1e-5)
    nats = -(target * torch.log(clipped) + (1.0 - target) * torch.log1p(-clipped))
    return (weight * nats).sum() / math.log(2.0)


def _components(binary: np.ndarray, min_pixels: int) -> List[np.ndarray]:
    """Deterministic 8-connected components without an OpenCV/SciPy dependency."""
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    found: List[np.ndarray] = []
    for start_y, start_x in zip(*np.nonzero(binary & ~seen)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        pixels: List[Tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            pixels.append((row, col))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nr, nc = row + dy, col + dx
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        if len(pixels) >= min_pixels:
            found.append(np.asarray(pixels, dtype=np.int32))
    return sorted(found, key=len, reverse=True)


def _component_proposals(component: np.ndarray, height: int, width: int) -> List[Primitive]:
    """Analytic candidates from area, bbox, and principal axes of one residual."""
    rows, cols = component[:, 0].astype(np.float64), component[:, 1].astype(np.float64)
    x = (2.0 * cols / max(width - 1, 1)) - 1.0
    y = (2.0 * rows / max(height - 1, 1)) - 1.0
    points = np.column_stack((x, y))
    centre = points.mean(axis=0)
    centered = points - centre
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, 1]
    perpendicular = np.array([-axis[1], axis[0]])
    theta = float(math.atan2(axis[1], axis[0]))
    # Elementwise projections avoid a noisy NumPy 2.2 BLAS warning observed
    # for a perfectly valid [N,2] @ [2] multiply on some macOS builds.
    long_projection = centered[:, 0] * axis[0] + centered[:, 1] * axis[1]
    short_projection = centered[:, 0] * perpendicular[0] + centered[:, 1] * perpendicular[1]
    long_low, long_high = np.quantile(long_projection, (0.01, 0.99))
    short_low, short_high = np.quantile(short_projection, (0.01, 0.99))
    long_half = max(float((long_high - long_low) / 2.0), 2.0 / max(width - 1, 1))
    short_half = max(float((short_high - short_low) / 2.0), 2.0 / max(height - 1, 1))
    bbox_centre = centre + axis * ((long_low + long_high) / 2.0) + perpendicular * ((short_low + short_high) / 2.0)
    moment_long = max(2.0 * math.sqrt(max(float(eigenvalues[1]), 0.0)), 2.0 / max(width - 1, 1))
    moment_short = max(2.0 * math.sqrt(max(float(eigenvalues[0]), 0.0)), 2.0 / max(height - 1, 1))
    pixel_area = 4.0 / max((height - 1) * (width - 1), 1)
    area_radius = max(math.sqrt(len(points) * pixel_area / math.pi), 2.0 / max(height, width))

    # Proposal order is a deterministic prior: smooth mass first, then concise
    # stroke/facet alternatives.  MDL, not this ordering, makes acceptance.
    return [
        Primitive.superellipse(float(centre[0]), float(centre[1]), moment_long, moment_short, 2.0, theta),
        Primitive.superellipse(float(bbox_centre[0]), float(bbox_centre[1]), long_half, short_half, 2.0, theta),
        Primitive.superellipse(float(bbox_centre[0]), float(bbox_centre[1]), long_half, short_half, 4.0, theta),
        Primitive.disk(float(centre[0]), float(centre[1]), area_radius),
        Primitive.capsule(
            float(bbox_centre[0]), float(bbox_centre[1]), max(long_half - 0.75 * short_half, 0.0),
            0.75 * short_half, theta,
        ),
        Primitive.wedge(float(bbox_centre[0]), float(bbox_centre[1]), 2.0 * long_half, 2.0 * short_half, theta),
        Primitive.wedge(
            float(bbox_centre[0]), float(bbox_centre[1]), 2.0 * long_half, 2.0 * short_half, theta + math.pi
        ),
        Primitive(
            "rounded_box",
            {"cx": float(bbox_centre[0]), "cy": float(bbox_centre[1]), "theta": theta,
             "hx": max(long_half - 0.25 * short_half, _EPS), "hy": max(0.75 * short_half, _EPS),
             "radius": 0.25 * short_half},
        ),
    ]


def _trainable_values(primitive: Primitive, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        name: torch.tensor(float(value), dtype=torch.float32, device=device, requires_grad=True)
        for name, value in primitive.values.items()
    }


def _project_values(values: Mapping[str, torch.Tensor]) -> None:
    """Keep direct optimization variables in the valid typed-grammar domain."""
    with torch.no_grad():
        for name, tensor in values.items():
            if name in {"cx", "cy"}:
                tensor.clamp_(-1.35, 1.35)
            elif name in {"radius", "rx", "ry", "half_length", "length", "width", "hx", "hy"}:
                tensor.clamp_(0.003, 2.2)
            elif name == "power":
                tensor.clamp_(1.05, 8.0)
            elif name == "theta":
                tensor.remainder_(2.0 * math.pi)


def _primitive_from_tensors(kind: str, values: Mapping[str, torch.Tensor]) -> Primitive:
    return Primitive(kind, {name: float(value.detach().cpu()) for name, value in values.items()})


def _refine_candidate(
    proposal: Primitive,
    current: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    tau: float,
    config: FitConfig,
) -> Tuple[Primitive, torch.Tensor, float]:
    """Locally optimize a single analytically seeded atom, then restore best."""
    values = _trainable_values(proposal, target.device)
    optimizer = torch.optim.Adam(values.values(), lr=config.learning_rate)
    best_loss = float("inf")
    best_values: Optional[Dict[str, torch.Tensor]] = None

    for _ in range(config.refine_steps):
        optimizer.zero_grad(set_to_none=True)
        atom = torch.sigmoid(-_primitive_sdf(proposal.kind, values, x, y) / tau)
        composite = current + (1.0 - current) * atom
        loss = _data_bits(composite, target, weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(values.values()), max_norm=4.0)
        optimizer.step()
        _project_values(values)
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_values = {name: value.detach().clone() for name, value in values.items()}

    assert best_values is not None
    fitted = _primitive_from_tensors(proposal.kind, best_values)
    with torch.no_grad():
        alpha = current + (1.0 - current) * fitted.alpha(x, y, tau)
        data_bits = float(_data_bits(alpha, target, weight).cpu())
    return fitted, alpha, data_bits


def fit_binary_mask(mask: np.ndarray | torch.Tensor, config: Optional[FitConfig] = None) -> FitResult:
    """Fit a deterministic greedy MDL SDF program to a silhouette.

    ``mask`` may be a ``[H,W]`` binary image (0/1, 0/255, bool) or a flat-colour
    ``[H,W,C]``/``[C,H,W]`` image whose corner median is its background.  The
    function intentionally does not use the repository's evaluator DSL; its
    render/score decisions are executable tensor operations, not string output.
    """
    cfg = config or FitConfig()
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target = _to_binary_mask(mask, device)
    if target.ndim != 2:
        raise ValueError("mask conversion must result in [H,W]")
    height, width = target.shape
    x, y = normalized_grid(height, width, device)
    tau = cfg.tau_pixels * 2.0 / max(height - 1, width - 1, 1)
    weight = _weight_map(target, cfg)
    current = torch.zeros_like(target)
    program = SymbolicProgram([], tau_pixels=cfg.tau_pixels)
    data_bits = float(_data_bits(current, target, weight).cpu())
    current_score = data_bits
    improvements: List[float] = []

    for _round in range(cfg.max_shapes):
        residual = (target * (1.0 - current)).detach().cpu().numpy() > 0.30
        components = _components(residual, cfg.min_component_pixels)[: cfg.max_components]
        if not components:
            break
        proposals: List[Primitive] = []
        seen: set[str] = set()
        for component in components:
            for proposal in _component_proposals(component, height, width):
                key = proposal.to_dsl()
                if key not in seen:
                    proposals.append(proposal)
                    seen.add(key)

        best: Optional[Tuple[Primitive, torch.Tensor, float, float]] = None
        existing_model_bits = program.model_bits()
        for proposal in proposals:
            fitted, alpha, candidate_data_bits = _refine_candidate(
                proposal, current, target, weight, x, y, tau, cfg
            )
            candidate_score = candidate_data_bits + cfg.model_weight * (existing_model_bits + fitted.model_bits())
            if best is None or candidate_score < best[3]:
                best = (fitted, alpha, candidate_data_bits, candidate_score)

        assert best is not None
        improvement = current_score - best[3]
        if improvement <= cfg.acceptance_margin_bits:
            break
        primitive, current, data_bits, current_score = best
        program.primitives.append(primitive)
        improvements.append(float(improvement))

    model_bits = program.model_bits()
    return FitResult(
        program=program,
        alpha=current.detach(),
        target=target.detach(),
        data_bits=float(data_bits),
        model_bits=float(model_bits),
        mdl_bits=float(data_bits + cfg.model_weight * model_bits),
        accepted_improvements=improvements,
    )


def benchmark_mask(mask: np.ndarray | torch.Tensor, config: Optional[FitConfig] = None) -> Dict[str, Any]:
    """Fit a mask and return the compact Equation-Length-versus-Accuracy record."""
    started = time.perf_counter()
    result = fit_binary_mask(mask, config)
    if result.alpha.is_cuda:
        torch.cuda.synchronize(result.alpha.device)
    elapsed = time.perf_counter() - started
    return {
        "equation": result.program.serialize(),
        "source_length": result.program.source_length,
        "iou": result.iou,
        "data_bits": result.data_bits,
        "model_bits": result.model_bits,
        "mdl_bits": result.mdl_bits,
        "runtime_seconds": elapsed,
        "primitive_kinds": [primitive.kind for primitive in result.program.primitives],
        "accepted_improvements": result.accepted_improvements,
        "program": result.program,
    }


def synthetic_fish_mask(size: int = 96, device: torch.device | str = "cpu") -> torch.Tensor:
    """A low-description-length fish: ellipse body plus one triangular tail."""
    x, y = normalized_grid(size, size, device)
    body = Primitive.superellipse(0.12, 0.0, 0.47, 0.255, power=2.0, theta=0.0)
    # For theta=pi the wedge's apex points left and its broad base joins body.
    tail = Primitive.wedge(-0.53, 0.0, 0.48, 0.53, theta=math.pi)
    alpha = 1.0 - (1.0 - body.alpha(x, y, 0.008)) * (1.0 - tail.alpha(x, y, 0.008))
    return (alpha >= 0.5).float()


def benchmark_synthetic_fish(size: int = 96, config: Optional[FitConfig] = None) -> Dict[str, Any]:
    """Targeted reproducible benchmark used by tests and integration checks."""
    return benchmark_mask(synthetic_fish_mask(size), config)


def _current_field_expression(primitive: Primitive, soft_tau: float) -> Optional[str]:
    """Compact typed field expression for the project evaluator."""
    v = primitive.values
    x, y = "(2*x-1)", "(2*y-1)"
    if primitive.kind == "disk":
        return f"disk({x},{y},{_number(v['cx'])},{_number(v['cy'])},{_number(v['radius'])})"
    if primitive.kind == "superellipse":
        return "superellipse(" + ",".join((
            x, y, _number(v["cx"]), _number(v["cy"]), _number(v["rx"]),
            _number(v["ry"]), _number(v["power"]), _number(v["theta"]),
        )) + ")"
    if primitive.kind == "capsule":
        return "capsule(" + ",".join((
            x, y, _number(v["cx"]), _number(v["cy"]), _number(v["half_length"]),
            _number(v["radius"]), _number(v["theta"]),
        )) + ")"
    if primitive.kind == "wedge":
        return "wedge(" + ",".join((
            x, y, _number(v["cx"]), _number(v["cy"]), _number(v["length"]),
            _number(v["width"]), _number(v["theta"]),
        )) + ")"
    return "rounded_box(" + ",".join((
        x, y, _number(v["cx"]), _number(v["cy"]), _number(v["hx"]),
        _number(v["hy"]), _number(v["radius"]), _number(v["theta"]),
    )) + ")"


__all__ = [
    "FitConfig",
    "FitResult",
    "Primitive",
    "SymbolicProgram",
    "benchmark_mask",
    "benchmark_synthetic_fish",
    "binary_iou",
    "fit_binary_mask",
    "normalized_grid",
    "synthetic_fish_mask",
]
