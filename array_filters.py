"""
array_filters.py
================
Geometry -> E*, E_true -> normalized power
transfer functions R_x, R_y -> summary metrics.

    y      = E x                E  = [1, x_i, y_i]              (N x 3)
    E*     = (E^T E)^-1 E^T                                     (3 x N)
    y      = E_true alpha       E_true[i,m] = exp(i(k_m x_i + l_m y_i))
    x*     = E* E_true alpha = T alpha                          (3 x K)

    t0, tx, ty = rows of T
    H_x = tx / (i k)     R_x = |H_x|^2      -> 1 = faithfully reported
    H_y = ty / (i l)     R_y = |H_y|^2      -> 0 = invisible to the array

R is a PER-MODE ratio against the true derivative. R = 1 is good near the
origin (signal you want) and bad away from it (unresolved energy passing
straight through). The main lobe width is set by APERTURE; N controls how
well everything outside it is rejected.
"""
import numpy as np

# ------------------------------------------------------------------ config
R_KM = 15.0                      # array circumradius: vertices lie on this circle
LAT0, LON0 = 0.0, -140.0
DEG_LON_KM, DEG_LAT_KM = 111.320, 110.570
NX, NY, DX_DEG = 512, 384, 1 / 24        # tpose24 grid: 512 x 384 at 1/24 deg
SIGNAL_CUT_KM = 100.0            # lambda > this = signal, < this = leakage


def polygon(n, R, phi0_deg=0.0):
    """Regular n-gon on a circle of radius R, plus a center sample."""
    th = np.deg2rad(phi0_deg) + 2 * np.pi * np.arange(n) / n
    return np.vstack([R * np.column_stack([np.cos(th), np.sin(th)]),
                      [[0.0, 0.0]]])


# Rotation choices: a regular N-gon repeats every 360/N degrees, and for ODD N
# a further rotation of 180/N maps it onto its own point-inversion, leaving |H|
# unchanged. So the distinct range is 360/N (even N) or 180/N (odd N).
def make_arrays(R):
    """The candidate stencils, all inscribed in a circle of radius R (km):
    every vertex (including the square's corners) lies on that circle."""
    a = R / np.sqrt(2)           # square half-width so its corners hit the circle
    return {
        "triangle":     polygon(3, R, 0.0),
        "triangle 90":  polygon(3, R, 90.0),
        "square":       np.array([[a, a], [-a, a], [-a, -a], [a, -a], [0.0, 0.0]]),
        "diamond":      polygon(4, R, 0.0),
        "hexagon":      polygon(6, R, 0.0),
        "hexagon 90":   polygon(6, R, 90.0),
        "octagon":      polygon(8, R, 0.0),
        "octagon 22.5": polygon(8, R, 22.5),
    }


ARRAYS = make_arrays(R_KM)
LABEL = {"triangle": "vertex east", "triangle 90": "vertex north",
         "square": "corners at $\\pm$45$\\degree$",
         "diamond": "square rotated 45$\\degree$",
         "hexagon": "vertex on $+x$", "hexagon 90": "rotated 90$\\degree$",
         "octagon": "vertex on $+x$", "octagon 22.5": "rotated 22.5$\\degree$"}
CIRC = {k: R_KM for k in ARRAYS}


def model_wavenumbers():
    """(k, l) the model can represent. fx, fy in cyc/km; KX, KY in rad/m."""
    fx = np.fft.fftshift(np.fft.fftfreq(NX, d=DX_DEG)) / DEG_LON_KM
    fy = np.fft.fftshift(np.fft.fftfreq(NY, d=DX_DEG)) / DEG_LAT_KM
    FX, FY = np.meshgrid(fx, fy)
    return fx, fy, FX, FY, 2 * np.pi * FX / 1e3, 2 * np.pi * FY / 1e3


FX_C, FY_C, FX, FY, KX, KY = model_wavenumbers()
KABS = np.hypot(FX, FY)                      # cyc/km


def transfer(pts_km):
    """Return H_x, H_y (complex), R_x, R_y, noise amplification."""
    x, y = pts_km[:, 0] * 1e3, pts_km[:, 1] * 1e3
    E_geom = np.column_stack([np.ones_like(x), x, y])
    E_star = np.linalg.inv(E_geom.T @ E_geom) @ E_geom.T
    E_true = np.exp(1j * (np.outer(x, KX.ravel()) + np.outer(y, KY.ravel())))
    T = E_star @ E_true
    tx, ty = T[1].reshape(KX.shape), T[2].reshape(KX.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_x, H_y = tx / (1j * KX), ty / (1j * KY)
    # L'Hopital on the singular lines: a wave with no variation in that
    # direction has zero true derivative, so the ratio is 0/0 there.
    kz, lz = KX == 0, KY == 0
    H_x[kz] = np.exp(1j * np.outer(y, KY[kz])).T @ (E_star[1] * x)
    H_y[lz] = np.exp(1j * np.outer(x, KX[lz])).T @ (E_star[2] * y)
    namp = np.sqrt((E_star[1] ** 2).sum() + (E_star[2] ** 2).sum())
    return H_x, H_y, np.abs(H_x) ** 2, np.abs(H_y) ** 2, namp


def mirror(pts, ax, tol=1e-6):
    o = np.round(pts[:-1], 6)
    q = o.copy(); q[:, ax] *= -1
    key = lambda a: a[np.lexsort((a[:, 1], a[:, 0]))]
    return bool(np.abs(key(o) - key(q)).max() < tol)


def half_power(Rmap, along):
    """Wavelength (km) where the cut through the origin first drops below 0.5."""
    f, c = ((FX_C, Rmap[np.argmin(np.abs(FY_C)), :]) if along == "k"
            else (FY_C, Rmap[:, np.argmin(np.abs(FX_C))]))
    m = f > 0
    fp, cp = f[m], c[m]
    return 1 / fp[cp < 0.5][0] if (cp < 0.5).any() else np.inf


def anisotropy(Rmap, n_ray=73):
    """Ratio of longest to shortest half-power wavelength over direction."""
    kk = np.linspace(1e-6, FX_C.max(), 2000)
    hp = []
    for th in np.linspace(0, np.pi, n_ray):
        ix = np.clip(np.searchsorted(FX_C, kk * np.cos(th)), 0, len(FX_C) - 1)
        iy = np.clip(np.searchsorted(FY_C, kk * np.sin(th)), 0, len(FY_C) - 1)
        b = np.where(Rmap[iy, ix] < 0.5)[0]
        if len(b):
            hp.append(1 / kk[b[0]])
    return (max(hp) / min(hp)) if hp else np.nan


def metrics():
    """Summary row per array. Returns a list of dicts."""
    sig = (KABS > 0) & (KABS <= 1 / SIGNAL_CUT_KM)
    leak = KABS > 1 / SIGNAL_CUT_KM
    out = []
    for nm, p in ARRAYS.items():
        _, _, Rx, Ry, namp = transfer(p)
        out.append(dict(
            array=nm, N=len(p) - 1,
            noise=namp * 1e5,
            hp_k=half_power(Rx, "k"), hp_l=half_power(Rx, "l"),
            aniso=anisotropy(Rx),
            R_sig=Rx[sig].mean(), R_leak=Rx[leak].mean(),
            leak_frac=100 * (Rx[leak] > 0.5).sum() / leak.sum(),
            maxR=Rx.max(),
            mirror_y=mirror(p, 0), mirror_x=mirror(p, 1),
        ))
    return out
