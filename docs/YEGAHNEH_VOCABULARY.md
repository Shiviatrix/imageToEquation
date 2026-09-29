# Yeganeh Vocabulary Dictionary: Evidence-led grammar for synthetic data

## Scope and evidence standard

This is a deliberately conservative vocabulary for *Yeganeh-like line, circle,
and parametric-plane art*, not a claim that every work by Hamid Naderi Yeganeh
uses every construct below. Public reporting documents two different workflows:

1. Generate a large parameter sweep of simple formula families, inspect the
   results, then gently retune a promising candidate.
2. Construct a recognisable subject step by step from trigonometric formulae.

For ImageToEquation, the faithful inverse problem is to predict a **small
generator family plus parameters**, not a free-form raster equation.

## Directly evidenced vocabulary

| Construction | Evidence | Finite-AST use |
|---|---|---|
| `sin(omega*t + phi)`, `cos(omega*t + phi)` | The artist says sine and cosine are his most-used functions. Published bird, boat, heart and segment examples use them explicitly. | Core coordinate leaves; small integer frequency, rational amplitude, rational/pi phase. |
| Integer powers, especially even envelopes | Bird formulae contain powers 2, 5, 6, 7, 10, 12, 16; 2015 segment examples contain cubed coordinates. | `pow(expr,n)`, whitelist `{2,3,4,5,6,7,8,10,12,16}`, high MDL cost for large `n`. |
| Sum/product of modulated trig terms | Detailed *A Bird in Flight* uses sums of products such as `cos(...)^16*cos(...)^12*sin(...)`. | `add`, `mul`, `neg`, rational scale; initial depth cap 3–4. |
| Complex-plane coordinates | Fish use complex endpoints; boat multiplies endpoints by `exp(3*pi*i/4)` for a rotation. | Typed `Vec2(x,y)` / complex AST, lowered to real operations. |
| Bounded parameter domain | Heart uses `theta <= t <= 2pi-theta`; bird/boat range over finite `k/t`. | `range(t0,t1)` is a first-class token. |
| Line-segment sweep / ruled surface | Fish and early birds join `A(k)` to `B(k)`; continuous bird is `{lambda*A(t)+(1-lambda)*B(t)}`. | `ruled(A,B,domain)` macro, rendered as a stroke or filled ruled region. |
| Circle sweep / union | Detailed bird is a union of circles with centre `(A(k),B(k))`, radius `R(k)`. | `circle_sweep(center,radius,domain)`, rendered by distance/alpha union. |
| Repetition / symmetry | Sources describe symmetric figures; periodic frequency yields repeated lobes/feathers. | `rotate`, `reflect`, `repeat_radial` as compiler macros (inferred implementation abstraction). |
| Triangle/sawtooth wave (secondary) | Documented 3,000-segment work uses triangle waves; summaries list sawtooth waves. | Hold out initially; later add `tri`/`saw` with low probability. |
| Circles and ellipses | Accounts describe constructions from lines/circles and works including “8,000 Ellipses”. | `circle`, `ellipse`, and their sweeps. |

### Published archetypes to seed exactly

**Fish search family.** The Guardian gives the family: for integer
`a,b,c,d in {1,...,6}`, connect
`A(k)=-2*cos(2*a*pi*k/100)+(i/2)*cos(2*b*pi*k/100)` to
`B(k)=(-2/15)*sin(2*c*pi*k/100)+(4*i/5)*sin(2*d*pi*k/100)`.
It was swept over more than a thousand choices, then rendered at continuous
`k`. This is the most useful exact supervised seed: compact, labelled, and
historically documented.

**Bird line family.** A published bird connects two trig-defined points. The
first includes `1.5*sin(t+pi/3)^7` and `0.25*cos(3*t)^2`; the other uses
low-amplitude sine terms at frequencies 3 and 1. The mechanism is a ruled
sweep, not a Fourier contour.

**Bird circle-sweep family.** The high-fidelity bird has centre `(A(t),B(t))`
and radius `R(t)`, each built from sums/products of high even powers of
sine/cosine. Sources explicitly attribute feather detail to periodicity. It is
structurally faithful but should enter the curriculum after simpler families.

**Boat / rotation.** The documented boat joins two trig-defined complex
endpoints and multiplies both by `exp(3*pi*i/4)`: one global rotation.

**Heart / root-constrained domain.** A published heart has sine/cosine
coordinates and uses `Theta`, a zero of its x-coordinate, as the lower/upper
domain bound. Represent this as `root(expr,bracket)` plus `range`, rather than
an arbitrary learned float.

## Important non-findings

The reviewed primary/near-primary corpus supports **the imaginary unit** `i`
(square root of -1) as complex-plane notation, but does **not** support a
claim that Yeganeh routinely restricts domains by evaluating real
`sqrt(negative)`. I found no reliable public formula demonstrating routine use
of `sgn`, `abs`, `min`, `max`, `tan`, Boolean CSG, or double-exponential masks
in the bird/fish/boat constructions. Those are valid codec/shader tools, but
belong in an **extended ImageToEquation grammar**, not the strict
Yeganeh-style distribution.

There is no defensible corpus-wide frequency table for `sin` versus `cos`.
Cosine and even powers occur often in the detailed bird formula, but that is
one sample, not a global statistic. Start balanced and measure frequencies only
from a curated, rights-cleared transcription corpus.

## Practical finite grammar (v0)

Use typed, canonical prefix tokens. `Rat` is a bounded reduced rational and
`F` a small positive integer.

```ebnf
Scene       := Layer+
Layer       := Stroke(Curve, Width, Colour) | Fill(CircleSweep, Colour)
Curve       := Ruled(Path, Path, Domain) | ParamPath(Path, Domain)
CircleSweep := SweepCircle(Path, Radius, Domain)
Path        := Vec2(Expr, Expr) | Rotate(Path, Angle) | Reflect(Path, Axis)
Expr        := Rat | t | Add(Expr, Expr) | Mul(Expr, Expr) | Neg(Expr)
             | Pow(Trig, N) | Scale(Rat, Expr)
Trig        := Sin(Phase) | Cos(Phase) | Tri(Phase) | Saw(Phase)
Phase       := Add(Scale(F*pi, t), Angle)
Radius      := Positive(Rat + SumOfEvenPowers)
Domain      := Interval(Angle, Angle) | RootBounded(Expr, bracket, side)
Angle       := Rat*pi
N           := 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 16
F           := 1..64  # curriculum starts at 1..12
```

Canonicalise `Add`/`Mul` (sort children), fold rationals, and place one global
affine transform at the scene root. Keep semantic macros such as `RULED`,
`CIRCLE_SWEEP`, and `ROTATE`; do not expand them into thousands of samples.

## Synthetic-data curriculum implied by the evidence

1. **Exact historical families (10–20%).** Enumerate fish and simple
   segment/boat-like families. Split train/test by generator family, not only
   by image seed.
2. **Typed perturbations (50–60%).** Mutate amplitudes, phases, frequencies,
   powers, intervals, rotation and reflection. Reject empty, self-occluding,
   near-duplicate and out-of-frame renders.
3. **Compositions (20–30%).** Layer 1–4 compact paths/sweeps with symmetry and
   palette variation. Sample length with a geometric prior such as
   `P(AST) proportional to exp(-0.18*tokens)`.
4. **Canonicalisation negatives (10%).** Render equivalent/near-equivalent
   trees and retain only the shortest canonical target. Otherwise the network
   learns arbitrary long equations.

Render antialiased vectors at 64² for training, then validate the *same AST*
at 256²/512². Randomise translation, scale, stroke width, supersampling,
background, inversion and modest raster noise—never expose pixel grids or DCT
coefficients as target primitives.

## Sources

1. Alex Bellos, *The Guardian*, “[Catch of the day: mathematician nets weird,
   complex fish](https://www.theguardian.com/science/alexs-adventures-in-numberland/2015/feb/24/catch-of-the-day-mathematician-nets-weird-complex-fish)” (2015): exact fish family, continuous sweep, complex plane, bird/boat links.
2. Washington University in St. Louis, “[Math Art: Hamid Naderi
   Yeganeh](https://www.math.wustl.edu/News2015/News2015_Feb_Yeganeh.html)”
   (2015): explicitly parameterised line-segment constructions.
3. Science Friday, “[Math Is Beautiful](https://www.sciencefriday.com/articles/math-is-beautiful/)”
   (2016): lines/circles, parameter search, refinement process.
4. Art the Science / Polyfield, “[Creators – Hamid Naderi
   Yeganeh](https://artthescience.com/magazine/2016/05/18/creators-hamid-naderi-yeganeh/)”
   (2016): direct statement that sine and cosine are most used.
5. Plus Magazine, “[This is not a bird (or a
   moustache)](https://plus.maths.org/not-bird)” (2015): complex interpolation
   `lambda*A(t)+(1-lambda)*B(t)` and trig endpoints.
6. Plus Magazine, “[Love curve](https://plus.maths.org/love-curve)” (2015):
   sine/cosine heart, restricted interval and root-defined bound.
7. Background pointer only: “[Hamid Naderi
   Yeganeh](https://en.wikipedia.org/wiki/Hamid_Naderi_Yeganeh)”.
