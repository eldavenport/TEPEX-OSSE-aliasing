"""
apply_to_model.py
=================
Apply the glider-array spatial filters (array_filters.py) to the tpose24 model
spectrum, to see where the ocean's velocity-gradient energy sits relative to
each filter -- i.e. where aliasing happens.

Core identity, per plane wave (see array_filters.py for the derivation of H):

    (du/dx)_reported (k,l) = H_x(k,l) * (du/dx)_true (k,l)

So we never simulate vehicles. We take the model spectrum alpha(k,l) of U and V,
form the true gradient spectra S = |i k alpha|^2, and read off:

    reported energy = R * S          R = |H|^2   (what the array reports)
    aliasing error  = |H - F|^2 * S              (what it gets WRONG)

The array does not measure a point gradient -- a plane fit over a disk of radius R
returns the gradient averaged over that footprint. So the target is the
footprint-average kernel F = 2 J1(|K|R)/(|K|R), not 1: F -> 1 for long waves and
rolls off (oscillating through zero) for sub-array waves the footprint smooths out.
The transfer row plots H - F, the departure from that target; energy in S where H
strays from F is the aliased contamination. (Comparing to 1 instead would charge the
array for not reproducing waves smaller than itself -- see why_divergence_not_gradients.md.)

Model: tpose24, an open equatorial-Pacific box, 512 x 384 at 1/24 deg. UVEL/VVEL are records 2/3 of diag_state.

Run with the `tpose` conda env:
    /home/edavenport/miniforge3/envs/tpose/bin/python apply_to_model.py
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from scipy.special import j1

import array_filters as af

# ------------------------------------------------------------------ config
MODEL_DIR = "/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3"
NZ = 138                                    # model vertical levels
REC_U, REC_V = 2, 3                         # UVEL, VVEL records in diag_state
DEPTHS_M = [25, 50, 75]                     # target depths -> nearest RC level
SUBSAMPLE = 6                               # use every Nth 3-hourly snapshot
FIG_DIR = "figures"
SPEC_DECADES = 3                            # log color range shown on spectrum maps
ARRAY_LON, ARRAY_LAT = 220.0, 0.0          # array centre: 140 W, 0 N
BOX_DEG = 5.0                               # side of the local spectrum box there
SIZES_KM = [15, 25, 50]                     # array circumradii to compare
COMPARE_SHAPES = ["triangle", "square", "diamond", "hexagon"]
COMPARE_DEPTH_M = 25                        # depth used for the size-comparison figs

DX_KM = af.DX_DEG * af.DEG_LON_KM           # ~4.64 km zonal grid spacing
DY_KM = af.DX_DEG * af.DEG_LAT_KM           # ~4.61 km meridional

# wavenumber grids, ifftshift-ed to line up with a raw (unshifted) fft2 output
KXu = np.fft.ifftshift(af.KX)               # rad/m
KYu = np.fft.ifftshift(af.KY)
KABS_RAD = np.hypot(af.KX, af.KY)           # rad/m, shifted (matches plotting grid)


def footprint(R_km):
    """Footprint-average transfer F(k,l) = 2 J1(|K|R)/(|K|R) on the shifted grid.
    The array reports the gradient averaged over its disk, so F -- not 1 -- is the
    honest target (F -> 1 for long waves, rolls off/oscillates for sub-array waves)."""
    x = KABS_RAD * R_km * 1e3
    F = np.ones_like(x)
    F[x > 0] = 2 * j1(x[x > 0]) / x[x > 0]
    return F


# ------------------------------------------------------------------ model I/O
def depth_levels():
    """Map DEPTHS_M to the nearest model level index, using RC.data."""
    RC = np.fromfile(f"{MODEL_DIR}/RC.data", dtype=">f4")
    return {d: int(np.argmin(np.abs(RC + d))) for d in DEPTHS_M}


def snapshot_files():
    return sorted(glob.glob(f"{MODEL_DIR}/diag_state.*.data"))[::SUBSAMPLE]


def read_level(path, rec, k):
    """One (ny, nx) velocity slice: record `rec`, level `k`, from a diag_state."""
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(7, NZ, af.NY, af.NX))
    return np.array(mm[rec, k], dtype=np.float64)     # copy out of the memmap


# ------------------------------------------------------------------ windowing
def _local_box():
    """Grid slice of the BOX_DEG-wide square centred on the array (0 N, 140 W).
    Localising the spectrum here keeps aliasing tied to the water the array
    actually samples, without changing the wavenumber grid (still full-domain
    FFT), so R*S stays valid and the map limits are unchanged."""
    XC = np.fromfile(f"{MODEL_DIR}/XC.data", ">f4").reshape(af.NY, af.NX)
    YC = np.fromfile(f"{MODEL_DIR}/YC.data", ">f4").reshape(af.NY, af.NX)
    jc = int(np.argmin(np.abs(XC[0] - ARRAY_LON)))
    ic = int(np.argmin(np.abs(YC[:, 0] - ARRAY_LAT)))
    h = int(round(BOX_DEG / af.DX_DEG / 2))
    return (slice(ic - h, ic + h), slice(jc - h, jc + h)), 2 * h


_BOX, _BN = _local_box()
_HANN = np.outer(np.hanning(_BN), np.hanning(_BN))
_HANN /= np.sqrt((_HANN ** 2).mean())       # preserve variance
_yy, _xx = np.mgrid[0:_BN, 0:_BN]
_PLANE = np.column_stack([np.ones(_BN * _BN), _xx.ravel(), _yy.ravel()])


def window(field):
    """Extract the local box, remove its plane fit, taper with Hann, zero the
    rest of the domain. Full-domain FFT of the result gives the local spectrum
    on the model's wavenumber grid."""
    sub = field[_BOX]
    c, *_ = np.linalg.lstsq(_PLANE, sub.ravel(), rcond=None)
    out = np.zeros_like(field)
    out[_BOX] = (sub - (_PLANE @ c).reshape(sub.shape)) * _HANN
    return out


# ------------------------------------------------------------------ spectra
def gradient_spectra(level):
    """Time-mean spectra at a model level, fftshift-ed onto array_filters' grid:
    gradient power S_dudx, S_dvdy (k^2|alpha|^2) AND the underlying velocity
    power Eu, Ev (|alpha|^2 -- where the waves are visible, un-suppressed by k^2)."""
    files = snapshot_files()
    norm = (af.NX * af.NY) ** 2             # Parseval: <f^2> = sum|alpha|^2 / norm
    Sx = Sy = Eu = Ev = 0.0
    for f in files:
        pu = np.abs(np.fft.fft2(window(read_level(f, REC_U, level)))) ** 2 / norm
        pv = np.abs(np.fft.fft2(window(read_level(f, REC_V, level)))) ** 2 / norm
        Eu += pu; Ev += pv
        Sx += KXu ** 2 * pu; Sy += KYu ** 2 * pv
    n = len(files)
    shift = np.fft.fftshift
    return shift(Sx / n), shift(Sy / n), shift(Eu / n), shift(Ev / n)


# ------------------------------------------------------------------ sanity check
def brute_force_check(level, pts, n_pos=500, seed=0):
    """Verify the spectral shortcut reproduces an actual plane fit.

    est_field = IFFT2[H_x * i k * alpha_u] must equal the plane-fit du/dx if the
    array were centred at that point. Sample one real snapshot at random interior
    points, fit the plane by hand, and compare to the spectral field.
    """
    f = snapshot_files()[len(snapshot_files()) // 2]
    U = read_level(f, REC_U, level)                        # raw, un-windowed
    au = np.fft.fft2(U)
    Hx = np.fft.ifftshift(af.transfer(pts)[0])
    est_field = np.real(np.fft.ifft2(Hx * 1j * KXu * au))  # array report everywhere

    # sample offsets (km) -> grid-index offsets
    ox, oy = pts[:, 0] / DX_KM, pts[:, 1] / DY_KM
    G = np.column_stack([np.ones(len(pts)), pts[:, 0] * 1e3, pts[:, 1] * 1e3])
    gx = np.linalg.pinv(G)[1]                              # d/dx weights, 1/m

    rng = np.random.default_rng(seed)
    radius_km = np.hypot(pts[:, 0], pts[:, 1]).max()
    pad = int(np.ceil(radius_km / min(DX_KM, DY_KM))) + 2
    iy = rng.integers(pad, af.NY - pad, n_pos)
    ix = rng.integers(pad, af.NX - pad, n_pos)
    fit, ref = [], []
    for j in range(n_pos):
        samp = map_coordinates(U, [iy[j] + oy, ix[j] + ox], order=1)
        fit.append(gx @ samp)
        ref.append(est_field[iy[j], ix[j]])
    fit, ref = np.array(fit), np.array(ref)
    r = np.corrcoef(fit, ref)[0, 1]
    ratio = fit.std() / ref.std()
    print(f"  brute-force check: corr={r:.3f}, std ratio={ratio:.3f}")
    assert r > 0.8 and 0.7 < ratio < 1.3, (
        "spectral prediction and brute-force plane fit disagree -- the "
        "random-phase/periodicity assumption may fail for this field.")


# ------------------------------------------------------------------ figures
_IK0 = int(np.argmin(np.abs(af.FX_C)))      # k = 0 column index (centered grid)
_IL0 = int(np.argmin(np.abs(af.FY_C)))      # l = 0 row index


def mask_cross(field):
    """Blank the k=0 / l=0 axes -- domain-scale + window leakage lives there and
    otherwise dominates the color scale, washing out the mesoscale."""
    f = field.copy()
    f[_IL0, :] = np.nan
    f[:, _IK0] = np.nan
    return f


def azimuthal(field, nbin=50):
    """Isotropic 1-D spectrum E(k): sum the 2-D power over each |k| ring and
    divide by ring width, so that sum(E*dk) = total variance -- the conventional
    spectrum (shallower than the 2-D density by one power of k). Log-spaced |k|
    bins for even log-log sampling. Returns (wavelength_km, E)."""
    kabs = af.KABS.ravel()
    power = field.ravel()
    kmin = kabs[kabs > 0].min()
    edges = np.logspace(np.log10(kmin), np.log10(min(af.FX_C.max(), af.FY_C.max())),
                        nbin + 1)
    who = np.digitize(kabs, edges)
    lam, E = [], []
    for b in range(1, nbin + 1):
        m = who == b
        if m.any():
            E.append(power[m].sum() / (edges[b] - edges[b - 1]))
            lam.append(1 / np.sqrt(edges[b - 1] * edges[b]))   # geometric center
    return np.array(lam), np.array(E)


def _map(ax, field, *, norm, cmap, hp_radius):
    m = ax.pcolormesh(af.FX_C, af.FY_C, field, norm=norm, cmap=cmap,
                      shading="auto", rasterized=True)
    ax.set_xlim(af.FX_C.min(), af.FX_C.max())          # full model band (to Nyquist)
    ax.set_ylim(af.FY_C.min(), af.FY_C.max())
    ax.set_aspect("equal")
    if hp_radius and np.isfinite(hp_radius):          # array's half-power circle
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(hp_radius * np.cos(th), hp_radius * np.sin(th),
                "k--", lw=0.8, alpha=0.7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    return m


def stencil_figure(name, pts, radius, depth_m, Sx, Sy, Eu, Ev, outdir):
    """One figure per stencil per depth: geometry + 4 map rows x (du/dx, dv/dy),
    plus a bottom row of 1-D azimuthal velocity spectra (where the waves show)."""
    Hx, Hy, Rx, Ry, _ = af.transfer(pts)
    F = footprint(radius)               # honest target: footprint-averaged gradient

    # zero-axis cross removed from the spectrum maps so the mesoscale gets contrast
    Sxm, Sym = mask_cross(Sx), mask_cross(Sy)
    cols = [  # (label, du/dx field, dv/dy field, norm-kind)
        ("$H-F$ (dev. from footprint)", (Hx - F).real, (Hy - F).real, "dev"),
        ("spectrum $S$", Sxm, Sym, "S"),
        ("reported $R\\,S$", Rx * Sxm, Ry * Sym, "S"),
        ("aliasing $|H-F|^2 S$", np.abs(Hx - F) ** 2 * Sxm,
         np.abs(Hy - F) ** 2 * Sym, "err"),
    ]
    # tight (SPEC_DECADES) log range off the masked peak so features stand out;
    # spectrum & reported share a scale so R*S is directly comparable to S
    d = 10 ** SPEC_DECADES
    smax = np.nanmax([Sxm, Sym])
    snorm = LogNorm(vmin=smax / d, vmax=smax)
    emax = np.nanmax([cols[3][1], cols[3][2]])
    enorm = LogNorm(vmin=emax / d, vmax=emax)
    dmax = np.nanmax(np.abs([(Hx - F).real, (Hy - F).real])) or 1.0
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    # half-power circle per component (radius in cyc/km = 1 / half-power lambda)
    hpx, hpy = af.half_power(Rx, "k"), af.half_power(Ry, "l")
    hp_rad = [1 / hpx if np.isfinite(hpx) else None,
              1 / hpy if np.isfinite(hpy) else None]

    fig = plt.figure(figsize=(7.2, 18))
    gs = GridSpec(6, 2, height_ratios=[1.1, 1, 1, 1, 1, 1.15], figure=fig,
                  hspace=0.4, wspace=0.25)

    # top: array geometry (spans both columns)
    axg = fig.add_subplot(gs[0, :])
    axg.scatter(pts[:, 0], pts[:, 1], s=60, c="tab:blue", zorder=3)
    th = np.linspace(0, 2 * np.pi, 200)
    axg.plot(radius * np.cos(th), radius * np.sin(th), "0.6", lw=1)
    axg.set_aspect("equal")
    axg.axhline(0, color="0.85", lw=0.6, zorder=0)
    axg.axvline(0, color="0.85", lw=0.6, zorder=0)
    axg.set_xlabel("x (km)")
    axg.set_ylabel("y (km)")
    hp = af.half_power(Rx, "k")
    axg.set_title(f"{name} ({af.LABEL[name]}) -- N={len(pts) - 1}, "
                  f"R={radius:.0f} km, {depth_m} m\nhalf-power $\\lambda_x$ = "
                  f"{hp:.0f} km" + (" (unresolved)" if np.isinf(hp) else ""))

    for r, (lab, fx, fy, kind) in enumerate(cols, start=1):
        norm, cmap = {"dev": (dnorm, "RdBu_r"),
                      "S": (snorm, "viridis"),
                      "err": (enorm, "magma")}[kind]
        for c, field, comp in [(0, fx, "du/dx"), (1, fy, "dv/dy")]:
            ax = fig.add_subplot(gs[r, c])
            m = _map(ax, field, norm=norm, cmap=cmap, hp_radius=hp_rad[c])
            if r == 1:
                ax.set_title(comp)
            if c == 0:
                ax.set_ylabel(f"{lab}\n$l$ (cyc/km)")
            if r == 4:
                ax.set_xlabel("$k$ (cyc/km)")
        fig.colorbar(m, ax=[fig.axes[-2], fig.axes[-1]], shrink=0.9,
                     pad=0.02, location="right")

    # bottom: 1-D azimuthal GRADIENT spectra (du/dx, dv/dy -- what we care about),
    # with the velocity spectrum as a faint reference (waves show there, un-hidden
    # by the k^2 weighting). Half-power wavelength marks the array's resolution.
    box_km = BOX_DEG * af.DEG_LON_KM        # longest wavelength the box resolves
    for c, (G, E, comp, dl, lam_hp) in enumerate(
            [(Sx, Eu, "u", "\\partial u/\\partial x", hpx),
             (Sy, Ev, "v", "\\partial v/\\partial y", hpy)]):
        lg, pg = azimuthal(G); kg = lg <= box_km
        lv, pv = azimuthal(E); kv = lv <= box_km
        ax = fig.add_subplot(gs[5, c])
        ax.loglog(lg[kg], pg[kg], "k", lw=1.4, label=f"${dl}$")
        axr = ax.twinx()                               # velocity on its own scale
        axr.loglog(lv[kv], pv[kv], color="0.6", lw=1, ls=":")
        axr.set_yticks([]); axr.set_ylabel(f"$|{comp}|^2$ ref", color="0.6",
                                           fontsize=7)
        if np.isfinite(lam_hp):
            ax.axvline(lam_hp, color="tab:red", ls="--", lw=1,
                       label=f"half-power {lam_hp:.0f} km")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel("wavelength (km)")
        ax.set_ylabel(f"azimuthal ${dl}$ power")
        ax.set_title(f"{comp}: gradient spectrum ({BOX_DEG:.0f}$\\degree$ box)")
        ax.grid(alpha=0.25, which="both")
        ax.invert_xaxis()                              # long waves on the left

    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/apply_{name.replace(' ', '_')}_{depth_m}m.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def compare_fields(shape, Sxm, Sym):
    """Per-size fields for one shape: transfer deviation H-F, reported R*S, aliasing
    |H-F|^2 S, plus half-powers and filtering skill (1 - error/footprint-signal,
    per component)."""
    per = []
    for R in SIZES_KM:
        pts = af.make_arrays(R)[shape]
        Hx, Hy, Rx, Ry, namp = af.transfer(pts)
        F = footprint(R)
        ex, ey = np.abs(Hx - F) ** 2 * Sxm, np.abs(Hy - F) ** 2 * Sym
        fsx, fsy = np.abs(F) ** 2 * Sxm, np.abs(F) ** 2 * Sym   # footprint-avg signal
        per.append(dict(
            R=R, pts=pts, dvx=(Hx - F).real, dvy=(Hy - F).real,
            rsx=Rx * Sxm, rsy=Ry * Sym, ex=ex, ey=ey,
            hpx=af.half_power(Rx, "k"), hpy=af.half_power(Ry, "l"), namp=namp * 1e5,
            skx=1 - np.nansum(ex) / np.nansum(fsx),
            sky=1 - np.nansum(ey) / np.nansum(fsy)))
    return per


def comparison_figure(shape, depth_m, Sx, Sy, dnorm, rsnorm, anorm):
    """One shape, all SIZES_KM side by side: 6 columns (du/dx at each size, then
    dv/dy) x rows [geometry, transfer H-F, reported R*S, aliasing |H-F|^2 S]. The
    H-F, R*S and aliasing color scales (dnorm, rsnorm, anorm) are shared across ALL
    comparison figures so brightness is comparable between shapes."""
    per = compare_fields(shape, mask_cross(Sx), mask_cross(Sy))

    ns = len(SIZES_KM)
    # dedicated last column for colorbars so every row shares the same map columns
    fig = plt.figure(figsize=(2.6 * 2 * ns + 1.0, 11), constrained_layout=True)
    gs = GridSpec(4, 2 * ns + 1, width_ratios=[1] * (2 * ns) + [0.06],
                  height_ratios=[1.0, 1.1, 1.1, 1.1], figure=fig)
    fig.suptitle(f"{shape} ({af.LABEL[shape]}) vs array radius -- {depth_m} m   "
                 "[left 3 = $\\partial u/\\partial x$, right 3 = "
                 "$\\partial v/\\partial y$]", fontsize=12)

    for i, p in enumerate(per):              # geometry, one per size (spans 2 cols)
        axg = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        axg.scatter(p["pts"][:, 0], p["pts"][:, 1], s=35, c="tab:blue", zorder=3)
        th = np.linspace(0, 2 * np.pi, 200)
        axg.plot(p["R"] * np.cos(th), p["R"] * np.sin(th), "0.6", lw=1)
        axg.set_aspect("equal")
        axg.set_xlim(-52, 52); axg.set_ylim(-52, 52)   # common scale across sizes
        axg.set_title(f"R = {p['R']:.0f} km, noise$\\times${p['namp']:.1f}")
        axg.tick_params(labelsize=7)

    m = {}
    for i, p in enumerate(per):              # cols: du/dx block then dv/dy block
        for comp, col, Dm, rs, al, hp, sk in (
                ("u/\\partial x", i, p["dvx"], p["rsx"], p["ex"], p["hpx"], p["skx"]),
                ("v/\\partial y", ns + i, p["dvy"], p["rsy"], p["ey"], p["hpy"], p["sky"])):
            hr = 1 / hp if np.isfinite(hp) else None
            for row, field, nm, cmap in ((1, Dm, "dev", "RdBu_r"),
                                         (2, rs, "rs", "viridis"),
                                         (3, al, "a", "magma")):
                ax = fig.add_subplot(gs[row, col])
                norm = {"dev": dnorm, "rs": rsnorm, "a": anorm}[nm]
                m[nm] = _map(ax, field, norm=norm, cmap=cmap, hp_radius=hr)
                if row == 1:
                    ax.set_title(f"R={p['R']:.0f} km, $\\partial {comp}$", fontsize=8)
                if row == 3:
                    ax.set_xlabel("$k$ (cyc/km)", fontsize=8)
                    ax.text(0.5, 0.02, f"skill {sk:.2f}", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=7, color="w")
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel({"dev": "$H-F$ (dev.)", "rs": "reported $R\\,S$",
                                   "a": "aliasing $|H-F|^2 S$"}[nm]
                                  + "\n$l$ (cyc/km)", fontsize=8)
                else:
                    ax.set_yticklabels([])
    fig.colorbar(m["dev"], cax=fig.add_subplot(gs[1, -1]), label="$H-F$")
    fig.colorbar(m["rs"], cax=fig.add_subplot(gs[2, -1]), label="$R\\,S$")
    fig.colorbar(m["a"], cax=fig.add_subplot(gs[3, -1]), label="$|H-F|^2 S$")

    outdir = f"{FIG_DIR}/comparison"
    os.makedirs(outdir, exist_ok=True)
    out = f"{outdir}/compare_{shape}_{depth_m}m.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
def main():
    levels = depth_levels()
    print(f"snapshots: {len(snapshot_files())} (every {SUBSAMPLE}th)")
    for depth_m, k in levels.items():
        print(f"depth {depth_m} m (level {k}):")
        Sx, Sy, Eu, Ev = gradient_spectra(k)
        brute_force_check(k, af.make_arrays(SIZES_KM[-1])["triangle"])
        for R in SIZES_KM:                    # full stencil set per size -> subfolder
            arrays = af.make_arrays(R)
            outdir = f"{FIG_DIR}/{R:.0f}km"
            for name, pts in arrays.items():
                stencil_figure(name, pts, R, depth_m, Sx, Sy, Eu, Ev, outdir)
            print(f"  R={R:.0f} km: {len(arrays)} figures -> {outdir}")
        if depth_m == COMPARE_DEPTH_M:        # size-comparison figures at one depth
            Sxm, Sym = mask_cross(Sx), mask_cross(Sy)
            allf = {s: compare_fields(s, Sxm, Sym) for s in COMPARE_SHAPES}
            rsmax = np.nanmax([[p["rsx"], p["rsy"]] for pf in allf.values() for p in pf])
            amax = np.nanmax([[p["ex"], p["ey"]] for pf in allf.values() for p in pf])
            dmax = np.nanmax([[np.abs(p["dvx"]), np.abs(p["dvy"])]
                              for pf in allf.values() for p in pf])
            dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
            rsnorm = LogNorm(vmin=rsmax / 10 ** SPEC_DECADES, vmax=rsmax)
            anorm = LogNorm(vmin=amax / 10 ** SPEC_DECADES, vmax=amax)
            print("  shape/size ranking (higher skill + lower noise = better):")
            print("    shape      R(km)  skill_x  skill_y  noise x1e5")
            for s in COMPARE_SHAPES:
                for p in allf[s]:
                    print(f"    {s:9s} {p['R']:5.0f}  {p['skx']:7.3f}  "
                          f"{p['sky']:7.3f}  {p['namp']:8.2f}")
                out = comparison_figure(s, depth_m, Sx, Sy, dnorm, rsnorm, anorm)
                print(f"    wrote {out}")


if __name__ == "__main__":
    main()
