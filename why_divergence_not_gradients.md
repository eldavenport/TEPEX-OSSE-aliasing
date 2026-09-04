# Why the hexagon advantage lives in the divergence figures, not the gradient transfer functions

This gets at exactly why the analysis had to move from the gradient figures to
the divergence figures. Short answer: **the hexagon's advantage is its low
anisotropy `H_x ≈ H_y`, and it only becomes visible once you (a) score against the
area-averaged truth instead of the point gradient, and (b) look at the signed
`H_x − H_y`, which `R_x = |H_x|²` and `R_y = |H_y|²` individually cannot show.**

## What each figure set actually measures

**Gradient figures** (`fig_R_dud{x,y}.png`, and the `R / S / R·S / |H−1|²·S` rows
in `apply_*`): answer "how faithfully is `du/dx` reported at each wavenumber,
judged against its own truth?" — and separately the same for `dv/dy`. Each
component is scored on its own.

**Divergence figures** (`divergence/`): answer "how faithfully is `du/dx + dv/dy`
reported?" — against the footprint-averaged truth `F·(du/dx + dv/dy)`. The skill
here is not in either component; it's in how the errors in the two components
cancel when added.

## Defining a footprint kernel = `F` 

`F` is the footprint-averaging kernel — which is applied per wavenumber. In the code it
is `disk_kernel()` in `divergence_maps.py`.

The plane fit does not estimate divergence at a point. By the divergence theorem,
a plane fit to samples spread over a disk of radius `R` estimates the divergence
averaged over that footprint. Averaging over a uniform disk of radius `R` is a
convolution, which in Fourier space is a multiplication by the disk's transform:

```
F(k,l) = 2·J1(x) / x,     x = |K|·R,     |K| = √(k²+l²)
```

`J1` is the first-order Bessel function; `F = 1` at `x = 0`.

- **`F → 1` at long waves** (`|K|R` small): a wave much larger than the array is
  nearly uniform across the footprint, so footprint-average ≈ point value.
- **`F` decays but oscillates at short waves** (`|K|R` large): a wave smaller than
  the array mostly cancels when averaged over the footprint, but not completely.
  The leftover is the fractional wavelength straddling the disk boundary. (A 1-D 
  box-average of a sine (`sinc`) is zero only for an integer number of
  half-wavelengths and nonzero otherwise.) `2·J1(x)/x` is the 2-D disk version: its
  envelope decays as ~`x^(-3/2)`, and it hits exactly zero only at the zeros of `J1`
  (`x ≈ 3.83, 7.02, 10.17, …`), not everywhere. So the true footprint-averaged
  divergence at short waves is small but generally nonzero.

Comparing to point truth (`F = 1` everywhere) effectively asks the array to reproduce 
short waves that it can't possibly "see". Using `error = |H − F|²·S` matches the plane-fit 
experiment, where the truth is the average divergence over the stencil.

`F` here is a disk of radius `R`, an approximation to the actual polygonal
footprint — close for the near-regular shapes; the exact polygon kernel is available
if wanted.

## Why Power Transfer Functions for d/dx and d/dy don't fully demonstrate the performance

Per Fourier mode, write the two gradient modes as `a = ik·û` and `b = il·v̂`. The
array reports:

```
reported_div = Hx·a + Hy·b
true_div     = F·(a + b)          (footprint-averaged truth)
error        = (Hx−F)·a + (Hy−F)·b
```

Rotate into isotropic + anisotropic coordinates (`div = a+b`, `stretch = a−b`):

```
error = (Hsym − F)·div  +  Δ·stretch
        Hsym = (Hx+Hy)/2      Δ = (Hx−Hy)/2
```

- `Hsym − F` is the isotropic footprint smoothing — roughly common to all shapes.
- `Δ = (Hx−Hy)/2` is the anisotropy, and it leaks the ocean's large stretching-
  deformation field into spurious divergence. Square/diamond: `Δ` big (~0.15).
  Hexagon/octagon: `Δ ≈ 0`. 

The performance driven by the hexagon/octagon is the minimal leakage of stretching into divergence estimates.

Thinking about an error budget (in terms of power):

```
|error|² = |Hx−F|²·|a|²  +  |Hy−F|²·|b|²  +  2·Re[(Hx−F)(Hy−F)*·a·b*]
              R-like x         R-like y            CROSS TERM
```

The first two terms are essentially what the `R_x`, `R_y` maps show (modulo the `F`
reference). However, the cross-term is important and is absent from
`R_x` and `R_y`. 

Two reasons the gradient power maps mess up the cross term:

1. **Squaring kills the sign.** `R_x = |H_x|²` can equal `R_y = |H_y|²` at a given
   `(k,l)` while `H_x` and `H_y` deviate from 1 in opposite directions. The square's 
   two gradient responses are mirror-images, not identical, so `|H_x|² ≈ |H_y|²` 
   pointwise hides a large signed `H_x − H_y`.

2. **No coupling to the ocean.** The cross term is weighted by `a·b*` (`*` =
   complex conjugate), whose time-mean is the cross-spectrum `cxy = ⟨a·b*⟩ =` 
   `⟨(ik·û)(il·v̂)*⟩`. `cxy` is the correlation between `du/dx` and
   `dv/dy` at each wavenumber. `R_x` and `R_y` are geometry-only and per-component; 
   they don't not account for ways in which the deformation field aliases into divergence.

## Is it only the cancellation? No — the reference matters more

It is *not* only a cross-component cancellation. Test each single gradient against
the **area-averaged** truth `F·du/dx` (not the point gradient) and the hexagon
already beats the square per-component. Summed `du/dx + dv/dy` error, geometry only,
band λ ≥ 15 km:

| shape   | vs area-average `F` | vs point (`F=1`) |
|---------|---------------------|------------------|
| square  | 0.27                | 2.9              |
| diamond | 0.40                | 1.4              |
| hexagon | **0.12**            | 2.0              |
| octagon | **0.01**            | 1.8              |

Against the area-average, hexagon/octagon are 2–25× better *per component*. Against
point truth every shape is terrible (order 1–3) and doesn't even rank cleanly. So
"point gradients look uniformly bad" was the **reference**, not the shape: the point
gradient contains sub-footprint waves the array physically can't return — a huge
error common to all shapes that buries the differences. This reference switch
(point → area-average) is what reconciled the ~59% point error with "hexagon nails
divergence," more than any cancellation.

The per-component error against `F` splits exactly:

```
|Hx−F|² + |Hy−F|²  =  2|Hsym−F|²  +  2|Δ|²
                        isotropic       anisotropy       (Δ = (Hx−Hy)/2)
```

(verified: square 0.098 + 0.169, hexagon 0.048 + 0.069, octagon 0.004 + 0.006). The
same anisotropy `Δ` penalizes each single gradient (the `2|Δ|²` term) **and**
divergence (as `Δ·stretch`). Low-`Δ` hexagon/octagon win in both framings — divergence
isn't better by magic cancellation, it's the same anisotropy story re-weighted by the
ocean's div-vs-stretch power.

So why did the original `R_x`/`R_y` figures hide it? Two things: (1) they were drawn
against **point truth** (`F = 1`), where the common unresolved-wave error dominates
and even mis-orders the shapes; (2) `R = |H|²` per component can't display `Δ`, a
*signed* difference between the two transfer functions.

## What *would* have shown it — and where it now lives

The one quantity that exposes the win is the **signed difference `H_x − H_y`** (i.e.
`Δ`). That's precisely the row now in `divergence_mechanism_figure` (`Hx`, `Hy`,
`Hx−Hy`, then the iso vs ani error split). If back in the early gradient work we'd
plotted `H_x − H_y` instead of `|H_x|²` and `|H_y|²`, the hexagon/octagon would have
jumped out immediately as ≈0 everywhere while the square lit up on the diagonals.

Two notes on the gradient figures:

- The `apply_*` gradient figures were fixed to use the footprint reference: the
  transfer row now plots `H − F` (departure from the footprint-averaged target,
  diverging around 0) and the error row is `|H − F|²·S`, not the old point-truth
  `|H − 1|²·S`. The skill metric is now `1 − Σ|H−F|²S / Σ|F|²S`. This makes them
  consistent with the divergence figures and stops charging the array for
  sub-footprint waves. (`R/|F|²` was rejected — it blows up to 10⁶–10⁹ at the
  `F`-zero rings.)
- Even the *correct* per-component error `|Hx−F|²·Sx` and `|Hy−F|²·Sy` still can't
  show the win alone — you need the `2·Re[(Hx−F)(Hy−F)*·cxy]` cross term. That's why
  the mechanism figure decomposes into iso+ani rather than dudx+dvdy.

## Bottom line

The gradient transfer functions aren't wrong, they're shown against the wrong
*reference* (point truth) and in the wrong *form* (`|H|²`, which hides the signed
anisotropy `Δ`). Score them against the area-averaged truth and plot `H_x − H_y`
instead, and the hexagon's advantage — low anisotropy — is visible per component,
no divergence figure required. Divergence just makes it unavoidable.
