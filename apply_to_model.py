"""
apply_to_model.py  -- PSEUDOCODE
================================
Everything up to this point is geometry: R = |E* E_true|^2 / |i k|^2 tells you
the per-mode response, but says nothing about how much energy your ocean
actually has at each wavenumber. This script supplies that half.

Core identity, per plane wave:

    (du/dx)_estimated (k,l) = H_x(k,l) * (du/dx)_true (k,l)

So if alpha(k,l,t) are your model's Fourier coefficients, then multiplying by
H_x and transforming back gives EXACTLY what the array would have reported --
no need to simulate vehicles at all.

    est_field = IFFT2[ H_x * (i k) * alpha ]
    true_field = IFFT2[ (i k) * alpha ]
    error_field = est_field - true_field

The key trick in step 4: because the inverse transform is evaluated on the
whole grid, you get the estimate as if the array were centered at EVERY grid
point simultaneously. One FFT pair per snapshot per array, not one per array
position. That turns an O(N_positions) Monte Carlo into an O(1) operation.

Sections marked  # >>> TODO  are model-specific.

Requires: array_filters.py (for transfer(), ARRAYS, and the wavenumber grid).
"""
import numpy as np
import array_filters as af


# =====================================================================
# 1. LOAD  -- model velocity snapshots on the horizontal grid
# =====================================================================
# >>> TODO: read u, v at your target depth, shape (n_time, ny, nx), in m/s.
#     Read the grid from the FILE, not from a config, and check it matches
#     array_filters.NX/NY/DX_DEG -- a mismatch here silently invalidates
#     everything downstream because H is defined on that exact grid.
u, v = load_model_velocity(depth=..., times=...)          # (nt, ny, nx)
dx_m, dy_m = grid_spacing_in_meters()                      # scalars

# >>> TODO: land / NaN handling. An FFT over masked data is meaningless.
#     Either crop to a fully wet sub-box or fill and report the induced error.
#     Do NOT silently nan_to_num.
assert np.isfinite(u).all(), "masked points present -- crop or fill first"


# =====================================================================
# 2. WINDOW  -- your domain is not periodic
# =====================================================================
# A plain FFT of a non-periodic field leaks, and that leakage will be
# misattributed to the array. Detrend, then taper, then renormalize variance.
u = detrend_2d(u)                    # remove plane fit over the whole domain
v = detrend_2d(v)
w2d = np.outer(np.hanning(ny), np.hanning(nx))
w2d /= np.sqrt((w2d**2).mean())      # preserve variance
u, v = u * w2d, v * w2d
# >>> TODO: run steps 2-6 with and without the window; report the difference
#     as your leakage uncertainty.


# =====================================================================
# 3. FORWARD TRANSFORM  -- alpha(k,l,t)
# =====================================================================
alpha_u = np.fft.fft2(u, axes=(-2, -1))      # (nt, ny, nx) complex
alpha_v = np.fft.fft2(v, axes=(-2, -1))

# The wavenumber grids MUST match array_filters (which is fftshift-ed).
KX = np.fft.ifftshift(af.KX)                 # rad/m, unshifted to match fft2
KY = np.fft.ifftshift(af.KY)


# =====================================================================
# 4. APPLY EACH FILTER  -- the whole point
# =====================================================================
results = {}
for name, pts in af.ARRAYS.items():
    H_x, H_y, R_x, R_y, noise_amp = af.transfer(pts)
    Hx, Hy = np.fft.ifftshift(H_x), np.fft.ifftshift(H_y)

    # true gradients (spectral derivative -- use the SAME operator throughout;
    # centered differences have their own transfer function, sin(k dx)/(k dx),
    # which is 0.68 at 20 km on a 1/24 deg grid and would contaminate this)
    dudx_true = np.real(np.fft.ifft2(1j * KX * alpha_u, axes=(-2, -1)))
    dvdy_true = np.real(np.fft.ifft2(1j * KY * alpha_v, axes=(-2, -1)))

    # what the array would report, at every grid point at once
    dudx_est = np.real(np.fft.ifft2(Hx * 1j * KX * alpha_u, axes=(-2, -1)))
    dvdy_est = np.real(np.fft.ifft2(Hy * 1j * KY * alpha_v, axes=(-2, -1)))

    # >>> TODO: for divergence, note that the SAME geometry filters both
    #     components, and the cross terms matter:
    #         div_true = dudx_true + dvdy_true
    #         div_est  = dudx_est  + dvdy_est
    #     A rotational (zero-divergence) wave can still produce a nonzero
    #     div_est whenever H_x != H_y at that (k,l).

    err = dudx_est - dudx_true

    # instrument noise is NOT in the model output -- add it explicitly or
    # compare against a noise-free prediction, but do not mix the two
    sigma_u = 0.01                                    # >>> TODO: m/s
    noise_var = (sigma_u * noise_amp) ** 2            # (s^-1)^2

    results[name] = dict(
        rms_true=np.std(dudx_true),
        rms_err=np.std(err),
        skill=1 - np.var(err) / np.var(dudx_true),
        noise_rms=np.sqrt(noise_var),
        total_rms=np.sqrt(np.var(err) + noise_var),
    )


# =====================================================================
# 5. SPECTRA  -- where in wavenumber the error actually lives
# =====================================================================
# The map R told you the per-mode response; this tells you which modes
# carried enough energy to matter. Absolute contribution goes as
# R * k^2 * S(k), so a red spectrum pulls the weight toward long waves --
# but the k^2 can keep short-wave leakage relevant.
S_true = np.mean(np.abs(1j * KX * alpha_u)**2, axis=0) / (nx * ny)**2
S_err = {nm: np.mean(np.abs((np.fft.ifftshift(af.transfer(p)[0]) - 1)
                            * 1j * KX * alpha_u)**2, axis=0) / (nx*ny)**2
         for nm, p in af.ARRAYS.items()}
# >>> TODO: plot azimuthal averages of S_true and S_err vs wavelength,
#     and the cumulative error contribution, to find the wavelength band
#     that dominates your error budget.


# =====================================================================
# 6. DIRECT CHECK  -- mandatory, not optional
# =====================================================================
# Steps 3-4 are exact only if the field is periodic and the transform pair is
# consistent. Verify against a brute-force plane fit before trusting anything.
for name, pts in af.ARRAYS.items():
    x, y = pts[:, 0] * 1e3, pts[:, 1] * 1e3
    G = np.column_stack([np.ones_like(x), x, y])
    E_star = np.linalg.pinv(G)

    brute = []
    for _ in range(1000):
        # >>> TODO: sample positions uniformly from the fully wet interior,
        #     keeping at least one aperture away from every boundary
        x0, y0, t = random_interior_position_and_time()
        u_samp = interp_bilinear(u[t], x0 + x, y0 + y)      # >>> TODO
        dudx_fit = E_star[1] @ u_samp
        brute.append(dudx_fit - dudx_true[t, iy(y0), ix(x0)])

    assert np.isclose(np.std(brute), results[name]["rms_err"], rtol=0.2), (
        f"{name}: spectral prediction and brute force disagree. "
        "Do NOT tune this away -- it means the random-phase / periodicity "
        "assumptions fail for your field, and the brute-force number is the "
        "one to trust.")


# =====================================================================
# 7. WHAT TO REPORT
# =====================================================================
# Per array:
#   - rms error vs rms true gradient, and skill
#   - the wavelength band carrying most of the error (from step 5)
#   - bias (filtering) vs noise contribution, so you can see which side of
#     the aperture trade-off you are on: shrinking the array cuts the
#     filtering error but raises noise as 2/(R*sqrt(N))
#
# Then repeat the whole thing over a few apertures. Because R depends only on
# the product kappa*R, changing aperture just slides your model spectrum under
# a fixed filter shape -- the optimum is where the two error terms cross.
