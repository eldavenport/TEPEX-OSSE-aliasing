# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An OSSE (Observing System Simulation Experiment) aliasing analysis for the TEPEX
project. It answers: given a small cluster of sampling points (a swarm/array of
vehicles) that estimates horizontal velocity gradients (du/dx, dv/dy) by fitting
a plane `y = a + b·x + c·y` to samples, **which wavenumbers does the array report
faithfully and which alias through as error?** The answer depends only on the
array *geometry*, not on the ocean field.

There are no vehicles simulated. The core insight (see `apply_to_model.py` header)
is that the plane-fit estimate of a gradient is a linear filter `H` on each Fourier
mode, so applying `H` to a model's spectrum via one FFT pair gives the array's
report *as if centered at every grid point at once* — turning an O(N_positions)
Monte Carlo into O(1).

## The math (shared by all files)

```
E      = [1, x_i, y_i]                 (N x 3)  plane-fit design matrix
E*     = (E^T E)^-1 E^T                 (3 x N)  weights; rows g0, gx, gy
E_true[i,m] = exp(i(k_m x_i + l_m y_i))          true modes at sample points
T = E* E_true                           (3 x K)  fit coeffs per mode
H_x = T[1]/(i k),  H_y = T[2]/(i l)              complex transfer functions
R_x = |H_x|^2,  R_y = |H_y|^2                     power response, per mode
```
`R = 1` = mode reported faithfully (good near origin, the signal you want);
`R` away from 0 for short waves = unresolved energy aliasing straight through.
Main-lobe width is set by aperture; N (sample count) sets rejection outside it.
`inv()` is used over `pinv()` deliberately so a collinear array *raises* instead
of silently returning a min-norm fit.

## Files

- **`array_filters.py`** — the real, importable, side-effect-free module. Geometry
  → transfer functions → summary metrics. Everything else depends on it. Key
  config constants at top: `A_KM` (aperture half-width), `NX/NY/DX_DEG` (must match
  the `tpose24` model grid: 20×15° at 1/24°), `SIGNAL_CUT_KM` (100 km signal/leakage
  split). `ARRAYS` dict holds the candidate geometries (triangle, square, diamond,
  hexagon, octagon, plus rotated variants). `transfer()` is the workhorse;
  `metrics()` returns one summary dict per array.
- **`array_transfer_functions.ipynb`** — `import array_filters as af` then the
  matplotlib plotting that produces the whole-array `fig_R_dud{x,y}.png` summary
  figures. No longer duplicates the module (single source of truth). Run with
  `jupyter nbconvert --execute`.
- **`apply_to_model.py`** — runnable. Loads tpose24 U/V, builds the time-mean
  true-gradient spectra `S`, applies each array's `H` from `af.transfer`, and writes
  per-stencil figures to `figures/apply_<array>_<depth>m.png` (transfer deviation
  `H-F`, spectrum `S`, reported `R·S`, aliasing `|H-F|²·S`, for du/dx and dv/dy).
  `F = footprint(R)` (Bessel disk kernel `2·J1(|K|R)/(|K|R)`) is the honest
  footprint-averaged target — **not** point truth `1` — so short waves the array
  can't resolve aren't miscounted as error (see `why_divergence_not_gradients.md`;
  same reference `W` used by `divergence_maps.py`). `brute_force_check` validates the
  spectral shortcut against an actual plane fit — keep it passing. Model path,
  depths, and time subsampling are the config block at the top.
- **`fig_R_dudx.png`, `fig_R_dudy.png`, `fig_summary_table.png`** — whole-array
  summary figures (from the notebook). `figures/` holds the per-stencil outputs.

## Wavenumber grid conventions (easy to get wrong)

- `array_filters` builds its grid **fftshift-ed** (origin centered) via
  `model_wavenumbers()`. `FX_C/FY_C` are 1-D (cyc/km); `KX/KY` are 2-D (rad/m).
- To multiply against a raw `np.fft.fft2` output you must `np.fft.ifftshift` both
  `af.KX/af.KY` and the `H_x/H_y` from `transfer()` first (see `apply_to_model.py`
  steps 3–4). Mismatched shifts silently corrupt everything.
- Any real model field must match `af.NX/NY/DX_DEG` exactly — `H` is defined only
  on that grid.

## Running

Use the **`tpose` conda env** (`/home/edavenport/miniforge3/envs/tpose/bin/python`)
— it has numpy, xarray, xmitgcm, matplotlib, scipy. The base env has no numpy.

```bash
PY=/home/edavenport/miniforge3/envs/tpose/bin/python
$PY -c "import array_filters as af; print(af.metrics())"   # geometry only
$PY apply_to_model.py                                       # reads model, writes figures/
$PY -m jupyter nbconvert --to notebook --execute --inplace array_transfer_functions.ipynb
```
No build, no test suite, no linter. `apply_to_model.py`'s `brute_force_check` is the
de-facto test. tpose24 model data lives at
`/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3/` (MITgcm big-endian float32
`diag_state.<step>.data`, `(7,138,384,512)`, UVEL=rec 2, VVEL=rec 3; depths in
`RC.data`). `.nc/.cdf/.pptx` are gitignored; PNGs are tracked outputs.
