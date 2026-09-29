"""Continuous, resolution-independent shader primitives for ImageToEquation.

The functions in this module operate on normalized continuous coordinates.  A
field has shape ``[1, H, W]`` (or is scalar) and a colour has shape
``[3, H, W]``.  They deliberately contain no image lookup table or pixel-grid
state: changing the output resolution only samples the same function more
densely.

``value_noise2`` and ``gradient_noise2`` use a sine-based *lattice hash* and
quintic interpolation.  The latter is gradient noise in the Perlin family, but
is not a bit-exact implementation of Ken Perlin's reference permutation-table
algorithm; callers and serializers must not label it simply "Perlin noise".
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor


Number = Union[float, Tensor]
MAX_FBM_OCTAVES = 8
_MIN_FREQUENCY = 1.0e-4
_MAX_FREQUENCY = 1024.0


def _positive_frequency(frequency: Tensor) -> Tensor:
    """Keep the coordinate transform finite without introducing pixel units."""
    return torch.clamp(torch.abs(frequency), _MIN_FREQUENCY, _MAX_FREQUENCY)


def _quintic_fade(t: Tensor) -> Tensor:
    """The C2 fade polynomial ``6t^5 - 15t^4 + 10t^3`` on ``[0, 1]``."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: Tensor, b: Tensor, t: Tensor) -> Tensor:
    return a + (b - a) * t


def _lattice_hash(ix: Tensor, iy: Tensor, seed: Tensor) -> Tensor:
    """A bounded, differentiable-in-seed pseudo-random lattice value in [-1, 1].

    Integer cell selection (``floor``) necessarily has a discontinuous
    derivative at lattice boundaries.  The output value itself is made smooth
    across those boundaries by the quintic interpolation below.
    """
    phase = 127.1 * ix + 311.7 * iy + 74.7 * seed + 19.19
    return torch.sin(phase)


def affine_gradient(x: Tensor, y: Tensor, bias: Tensor, dx: Tensor, dy: Tensor) -> Tensor:
    """The compact colour field ``bias + dx*x + dy*y``."""
    return bias + dx * x + dy * y


def bilinear_gradient(
    x: Tensor,
    y: Tensor,
    c00: Tensor,
    c10: Tensor,
    c01: Tensor,
    c11: Tensor,
) -> Tensor:
    """A continuous bilinear field on the normalized unit square.

    ``c00, c10, c01, c11`` refer respectively to the values at
    ``(0,0), (1,0), (0,1), (1,1)``.  Coordinates outside the unit square
    extrapolate continuously, which is preferable to a hard clamp while GD is
    fitting a transform.
    """
    return (
        c00 * (1.0 - x) * (1.0 - y)
        + c10 * x * (1.0 - y)
        + c01 * (1.0 - x) * y
        + c11 * x * y
    )


def value_noise2(x: Tensor, y: Tensor, frequency: Tensor, seed: Tensor) -> Tensor:
    """Continuous 2-D value noise in approximately ``[-1, 1]``.

    Values are deterministically hashed at the four corners of a continuous
    lattice cell and then quintically interpolated.  It is differentiable with
    respect to ``frequency`` and ``seed`` away from cell boundaries, and has
    smooth coordinate derivatives across boundaries.  It is intentionally not
    called Perlin noise: it interpolates lattice *values*, not gradients.
    """
    f = _positive_frequency(frequency)
    px, py = x * f, y * f
    ix, iy = torch.floor(px), torch.floor(py)
    tx, ty = px - ix, py - iy
    ux, uy = _quintic_fade(tx), _quintic_fade(ty)

    v00 = _lattice_hash(ix, iy, seed)
    v10 = _lattice_hash(ix + 1.0, iy, seed)
    v01 = _lattice_hash(ix, iy + 1.0, seed)
    v11 = _lattice_hash(ix + 1.0, iy + 1.0, seed)
    return _lerp(_lerp(v00, v10, ux), _lerp(v01, v11, ux), uy)


def _gradient(ix: Tensor, iy: Tensor, seed: Tensor) -> tuple[Tensor, Tensor]:
    """Unit lattice gradient derived from the deterministic lattice hash."""
    angle = math.pi * (_lattice_hash(ix, iy, seed) + 1.0)
    return torch.cos(angle), torch.sin(angle)


def gradient_noise2(x: Tensor, y: Tensor, frequency: Tensor, seed: Tensor) -> Tensor:
    """C2-interpolated 2-D lattice gradient noise, remapped near ``[0, 1]``.

    This has the same useful construction as improved gradient noise—gradient
    vectors at lattice corners and quintic fading—but the procedural sine hash
    replaces a reference permutation table.  The output is therefore a concise
    deterministic gradient-noise shader, not an assertion of bit-exact Perlin
    noise.
    """
    f = _positive_frequency(frequency)
    px, py = x * f, y * f
    ix, iy = torch.floor(px), torch.floor(py)
    tx, ty = px - ix, py - iy
    ux, uy = _quintic_fade(tx), _quintic_fade(ty)

    g00x, g00y = _gradient(ix, iy, seed)
    g10x, g10y = _gradient(ix + 1.0, iy, seed)
    g01x, g01y = _gradient(ix, iy + 1.0, seed)
    g11x, g11y = _gradient(ix + 1.0, iy + 1.0, seed)
    n00 = g00x * tx + g00y * ty
    n10 = g10x * (tx - 1.0) + g10y * ty
    n01 = g01x * tx + g01y * (ty - 1.0)
    n11 = g11x * (tx - 1.0) + g11y * (ty - 1.0)
    signed_noise = _lerp(_lerp(n00, n10, ux), _lerp(n01, n11, ux), uy)

    # A unit vector's dot product with a local cell offset is bounded by sqrt(2).
    # 0.35 keeps the practical range close to [0, 1] without a clipping kink.
    return 0.5 + 0.35 * signed_noise


def fractal_brownian_motion(
    x: Tensor,
    y: Tensor,
    base_frequency: Tensor,
    gain: Tensor,
    seed: Tensor,
    octaves: int,
) -> Tensor:
    """A bounded-octave, differentiable fBm-style sum of gradient-noise bands.

    ``octaves`` is grammar structure rather than a learned tensor and must be
    an integer from 1 through :data:`MAX_FBM_OCTAVES`.  Each octave doubles
    frequency and multiplies amplitude by ``abs(gain)`` (clamped below one),
    then the sum is normalized by the total amplitude.
    """
    if not isinstance(octaves, int) or not 1 <= octaves <= MAX_FBM_OCTAVES:
        raise ValueError(f"octaves must be an integer in [1, {MAX_FBM_OCTAVES}]")
    amplitude_gain = torch.clamp(torch.abs(gain), 0.0, 0.98)
    total: Tensor | Number = 0.0
    amplitude: Tensor | Number = 1.0
    amplitude_sum: Tensor | Number = 0.0
    for octave in range(octaves):
        frequency = base_frequency * float(2 ** octave)
        octave_seed = seed + float(octave) * 19.19
        total = total + amplitude * gradient_noise2(x, y, frequency, octave_seed)
        amplitude_sum = amplitude_sum + amplitude
        amplitude = amplitude * amplitude_gain
    return total / torch.clamp(torch.as_tensor(amplitude_sum, device=x.device), min=1.0e-6)


def rgb(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    """Construct straight RGB ``[3,H,W]`` from scalar or single-channel fields."""
    r, g, b = torch.broadcast_tensors(r, g, b)
    if r.ndim == 0:
        return torch.stack((r, g, b))
    if r.ndim == 2:
        return torch.stack((r, g, b), dim=0)
    if r.ndim == 3 and r.shape[0] == 1:
        return torch.cat((r, g, b), dim=0)
    raise ValueError("rgb channels must be scalar, [H,W], or [1,H,W] fields")


def _as_rgb(value: Tensor, reference: Tensor) -> Tensor:
    """Promote scalar/single-channel values to straight RGB using ``reference``."""
    if reference.ndim == 0:
        if value.ndim == 0:
            return value.expand(3)
        if value.ndim == 1 and value.shape[0] == 3:
            return value
        raise ValueError("a spatial colour needs a spatial alpha reference")
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
    if reference.ndim != 3 or reference.shape[0] not in (1, 3):
        raise ValueError("alpha/reference must have shape [H,W], [1,H,W], or [3,H,W]")
    spatial_reference = reference[:1]
    if value.ndim == 0:
        return (value + torch.zeros_like(spatial_reference)).expand(3, -1, -1)
    if value.ndim == 1:
        if value.shape[0] != 3:
            raise ValueError("a constant RGB colour must have exactly three channels")
        return value.view(3, 1, 1).expand(-1, spatial_reference.shape[1], spatial_reference.shape[2])
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("colour must be scalar, [H,W], [1,H,W], or [3,H,W]")
    if value.shape[0] == 3:
        return value
    if value.shape[0] == 1:
        return value.expand(3, -1, -1)
    raise ValueError("colour's leading dimension must have one or three channels")


def shade(alpha: Tensor, r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    """Place a continuous RGB shader over black through an SDF alpha matte."""
    a = torch.clamp(alpha, 0.0, 1.0)
    colour = rgb(r, g, b)
    if a.ndim == 0 and colour.ndim == 3:
        a = a + torch.zeros_like(colour[:1])
    return _as_rgb(colour, a) * a


def over(background: Tensor, foreground: Tensor, alpha: Tensor) -> Tensor:
    """Straight-colour Porter--Duff *over* on an opaque canvas.

    ``alpha`` is the foreground coverage/matte.  This returns an RGB canvas;
    the caller may nest ``over`` for ordered layers.  Both colour inputs can be
    RGB or a scalar/single-channel field, in which case they are promoted.
    """
    a = torch.clamp(alpha, 0.0, 1.0)
    if a.ndim == 0:
        for colour in (foreground, background):
            if colour.ndim == 2:
                a = a + torch.zeros_like(colour).unsqueeze(0)
                break
            if colour.ndim == 3:
                a = a + torch.zeros_like(colour[:1])
                break
    fg = _as_rgb(foreground, a)
    bg = _as_rgb(background, a)
    return fg * a + bg * (1.0 - a)


def yeganeh_trig_noise(
    x: Tensor,
    y: Tensor,
    base_frequency: Tensor,
    lacunarity: Tensor,
    octaves: int,
) -> Tensor:
    """A bounded-octave, multi-octave rotated trigonometric noise function."""
    if not isinstance(octaves, int) or not 1 <= octaves <= MAX_FBM_OCTAVES:
        raise ValueError(f"octaves must be an integer in [1, {MAX_FBM_OCTAVES}]")
        
    total: Tensor | Number = 0.0
    f = _positive_frequency(base_frequency)
    lac = torch.clamp(torch.abs(lacunarity), 1.05, 10.0)
    
    for s in range(1, octaves + 1):
        sf = float(s)
        current_freq = f * (lac ** sf)
        
        c7 = math.cos(7.0 * sf)
        s7 = math.sin(7.0 * sf)
        c8 = math.cos(8.0 * sf)
        s8 = math.sin(8.0 * sf)
        c10 = math.cos(10.0 * sf)
        c15 = math.cos(15.0 * sf)
        c17 = math.cos(17.0 * sf)
        c5 = math.cos(5.0 * sf)
        
        rot1 = c7 * x + s7 * y
        rot2 = c7 * y - s7 * x
        rot3 = c8 * x + s8 * y
        
        term1 = torch.cos(current_freq * (1.0 + c10) * (rot1 + 2.0 * c17) + 4.0 * torch.cos(current_freq * rot1) + 2.0 * c5)
        term2 = torch.cos(current_freq * (1.0 + c10) * (rot2 + 2.0 * c15) + 4.0 * torch.cos(current_freq * rot3) + 2.0 * c7)
        
        total = total + term1 * term2
        
    return total / float(octaves)


def dexp_mask(distance: Tensor, sharpness: Tensor) -> Tensor:
    """The Yeganeh double-exponential hard mask."""
    s = torch.clamp(torch.abs(sharpness), min=1.0)
    return torch.exp(-torch.exp(torch.clamp(-s * distance, max=80.0)))


def polar_r(x: Tensor, y: Tensor, cx: Tensor, cy: Tensor) -> Tensor:
    """Returns the radius from (cx, cy)."""
    return torch.hypot(x - cx, y - cy)

def polar_theta(x: Tensor, y: Tensor, cx: Tensor, cy: Tensor) -> Tensor:
    """Returns the angle from (cx, cy)."""
    return torch.atan2(y - cy, x - cx)


__all__ = [
    "MAX_FBM_OCTAVES",
    "affine_gradient",
    "bilinear_gradient",
    "value_noise2",
    "gradient_noise2",
    "fractal_brownian_motion",
    "yeganeh_trig_noise",
    "dexp_mask",
    "polar_r",
    "polar_theta",
    "rgb",
    "shade",
    "over",
]
