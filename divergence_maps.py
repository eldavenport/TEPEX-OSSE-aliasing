"""
divergence_maps.py
==================
Intermediate wavenumber-space maps that walk from the transfer functions H to
the gradient/divergence ERROR, so the mechanism is visible.

The array does not measure a point derivative -- it measures the derivative
AVERAGED over its footprint. So the honest reference is the footprint-average
kernel F (a disk of radius R), not 1. Per mode:

    du/dx estimate = H_x (ik) u_hat        footprint truth = F (ik) u_hat
    du/dx error    = (H_x - F)(ik) u_hat            density |H_x - F|^2 S_dudx

For DIVERGENCE the two gradients combine, and there is an exact split:

    div_error = (H_sym - F) * div_true  +  Delta * stretch_true
        H_sym = (H_x + H_y)/2   Delta = (H_x - H_y)/2   (the ANISOTROPY)
        div    = du/dx + dv/dy   stretch = du/dx - dv/dy  (deformation, large)

Isotropic arrays (hexagon, octagon) have Delta ~ 0, so the large stretching
deformation does NOT leak into spurious divergence -- that is why they win.

Run with the tpose env:  .../envs/tpose/bin/python divergence_maps.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec

import apply_to_model as A
import array_filters as af

KXu, KYu = A.KXu, A.KYu                       # unshifted, rad/m (match raw fft2)
DECADES = 3
SHAPES = ["square", "diamond", "hexagon"]
OUT = f"{A.FIG_DIR}/divergence"

footprint = A.footprint                       # footprint-average transfer F = 2 J1(|K|R)/(|K|R)


def hxy(shape, R):
    """H_x, H_y (COMPLEX) on the shifted grid. Symmetric arrays are real, but the
    triangle is genuinely complex (|Im H| ~ 30), so keep the full value -- taking
    .real would corrupt any asymmetric stencil."""
    Hx, Hy, *_ = af.transfer(af.make_arrays(R)[shape])
    return Hx, Hy


def spectra_bundle(level):
    """Time-mean spectra on the shifted grid: gradient S_dudx/S_dvdy, divergence
    S_div, stretching deformation S_str, the div-stretch cross spectrum cds, and
    the du/dx-dv/dy cross spectrum cxy (needed for exact error with complex H)."""
    acc = dict(dudx=0.0, dvdy=0.0, div=0.0, str=0.0, cds=0.0, cxy=0.0)
    files = A.snapshot_files()
    for f in files:
        au = np.fft.fft2(A.window(A.read_level(f, A.REC_U, level)))
        av = np.fft.fft2(A.window(A.read_level(f, A.REC_V, level)))
        dx, dy = 1j * KXu * au, 1j * KYu * av
        acc["dudx"] += np.abs(dx) ** 2
        acc["dvdy"] += np.abs(dy) ** 2
        acc["div"] += np.abs(dx + dy) ** 2
        acc["str"] += np.abs(dx - dy) ** 2
        acc["cds"] += (dx + dy) * np.conj(dx - dy)
        acc["cxy"] += dx * np.conj(dy)
    n = len(files)
    return {k: np.fft.fftshift(v / n) for k, v in acc.items()}


# ------------------------------------------------------------------ global norms
def global_norms(B):
    """One set of color scales shared by EVERY divergence figure, so maps are
    comparable across sizes and shapes. Covers: transfer H, transfer-difference
    (H-F and Hx-Hy), spectrum S, and error density."""
    Sx, Sy = A.mask_cross(B["dudx"]), A.mask_cross(B["dvdy"])
    Sdiv, Sstr = A.mask_cross(B["div"]), A.mask_cross(B["str"])
    Cds = A.mask_cross(B["cds"].real)
    dmax = emax = 0.0
    for R in A.SIZES_KM:
        F = footprint(R)
        for s in SHAPES + ["octagon"]:
            Hx, Hy = hxy(s, R)
            Hs, D = (Hx + Hy) / 2, (Hx - Hy) / 2
            dmax = max(dmax, np.nanmax(np.abs([Hx - Hy, Hx - F, Hy - F])))
            iso, ani = np.abs(Hs - F) ** 2 * Sdiv, np.abs(D) ** 2 * Sstr
            tot = iso + ani + 2 * (Hs - F) * D * Cds
            gx, gy = np.abs(Hx - F) ** 2 * Sx, np.abs(Hy - F) ** 2 * Sy
            emax = max(emax, np.nanmax([iso, ani, np.abs(tot), gx, gy]))
    smax = np.nanmax([Sx, Sy, Sdiv, Sstr])
    return dict(h=Normalize(-0.3, 1.05),
                d=TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax),
                s=LogNorm(smax / 10 ** DECADES, smax),
                e=LogNorm(emax / 10 ** DECADES, emax))


# ------------------------------------------------------------------ figure 1
def gradient_chain_figure(shape, R, depth_m, B, N):
    """Per-gradient error chain for one stencil: 2 columns (du/dx, dv/dy),
    rows [H, H-F, S, |H-F|^2 S] -- how energy at each wavenumber becomes error."""
    Hx, Hy = hxy(shape, R)
    F = footprint(R)
    Sx, Sy = A.mask_cross(B["dudx"]), A.mask_cross(B["dvdy"])
    ex, ey = np.abs(Hx - F) ** 2 * Sx, np.abs(Hy - F) ** 2 * Sy

    snorm, enorm, hnorm, dnorm = N["s"], N["e"], N["h"], N["d"]
    rows = [("filter $H$", Hx.real, Hy.real, hnorm, "viridis"),
            ("$H-F$ (dev. from footprint truth)", (Hx - F).real, (Hy - F).real,
             dnorm, "RdBu_r"),
            ("spectrum present $S$", Sx, Sy, snorm, "cividis"),
            ("error density $|H-F|^2 S$", ex, ey, enorm, "magma")]

    fig = plt.figure(figsize=(7.5, 13), constrained_layout=True)
    gs = GridSpec(5, 3, width_ratios=[1, 1, 0.06], height_ratios=[0.5, 1, 1, 1, 1],
                  figure=fig)
    fig.suptitle(f"{shape} R={R:.0f} km, {depth_m} m -- per-gradient error chain "
                 "(reference $F$ = footprint average)", fontsize=11)
    # top: the footprint reference F (isotropic, same for both gradients)
    axw = fig.add_subplot(gs[0, :2])
    mw = A._map(axw, F, norm=hnorm, cmap="viridis", hp_radius=None)
    axw.set_title("footprint-average reference $F$", fontsize=9)
    fig.colorbar(mw, cax=fig.add_subplot(gs[0, 2]))

    for r, (lab, fx, fy, norm, cmap) in enumerate(rows, start=1):
        for c, field, comp in [(0, fx, "\\partial u/\\partial x"),
                               (1, fy, "\\partial v/\\partial y")]:
            ax = fig.add_subplot(gs[r, c])
            m = A._map(ax, field, norm=norm, cmap=cmap, hp_radius=None)
            if r == 1:
                ax.set_title(f"${comp}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{lab}\n$l$ (cyc/km)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 4:
                ax.set_xlabel("$k$ (cyc/km)", fontsize=8)
            else:
                ax.set_xticklabels([])
        fig.colorbar(m, cax=fig.add_subplot(gs[r, 2]))

    import os
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/gradient_chain_{shape}_{R:.0f}km_{depth_m}m.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ figure 2
def divergence_mechanism_figure(R, depth_m, B, N):
    """Why isotropy wins: columns = SHAPES, rows walk H_x, H_y -> anisotropy
    (H_x-H_y) -> the two error terms -> total divergence error. Shared scales."""
    F = footprint(R)
    Sdiv, Sstr = A.mask_cross(B["div"]), A.mask_cross(B["str"])
    Cds = A.mask_cross(B["cds"].real)
    ref = np.sqrt(np.nansum(F ** 2 * B["div"]))           # footprint-avg div RMS

    per = {}
    for s in SHAPES:
        Hx, Hy = hxy(s, R)
        Hs, D = (Hx + Hy) / 2, (Hx - Hy) / 2
        iso = np.abs(Hs - F) ** 2 * Sdiv
        ani = np.abs(D) ** 2 * Sstr
        tot = iso + ani + 2 * (Hs - F) * D * Cds
        pct = np.sqrt(np.nansum(np.abs(tot))) / ref * 100
        per[s] = dict(Hx=Hx.real, Hy=Hy.real, D=(Hx - Hy).real,
                      iso=iso, ani=ani, tot=tot, pct=pct)

    hnorm, dnorm, enorm = N["h"], N["d"], N["e"]
    rows = [("$H_x$", "Hx", hnorm, "viridis"),
            ("$H_y$", "Hy", hnorm, "viridis"),
            ("anisotropy $H_x-H_y$", "D", dnorm, "RdBu_r"),
            ("iso. err $|H_{sym}-F|^2 S_{div}$", "iso", enorm, "magma"),
            ("aniso. err $|\\Delta|^2 S_{str}$", "ani", enorm, "magma"),
            ("total div. error", "tot", enorm, "magma")]

    ns = len(SHAPES)
    fig = plt.figure(figsize=(3.0 * ns + 0.8, 15), constrained_layout=True)
    gs = GridSpec(7, ns + 1, width_ratios=[1] * ns + [0.06],
                  height_ratios=[0.8] + [1] * 6, figure=fig)
    fig.suptitle(f"Divergence error mechanism -- R={R:.0f} km, {depth_m} m "
                 "(reference $F$ = footprint average)", fontsize=12)

    for c, s in enumerate(SHAPES):                        # geometry row
        axg = fig.add_subplot(gs[0, c])
        pts = af.make_arrays(R)[s]
        axg.scatter(pts[:, 0], pts[:, 1], s=30, c="tab:blue")
        th = np.linspace(0, 2 * np.pi, 120)
        axg.plot(R * np.cos(th), R * np.sin(th), "0.6", lw=1)
        axg.set_aspect("equal"); axg.set_xlim(-1.2 * R, 1.2 * R)
        axg.set_ylim(-1.2 * R, 1.2 * R); axg.tick_params(labelsize=6)
        axg.set_title(f"{s}\ntotal div err = {per[s]['pct']:.0f}%", fontsize=9)

    for r, (lab, key, norm, cmap) in enumerate(rows, start=1):
        for c, s in enumerate(SHAPES):
            ax = fig.add_subplot(gs[r, c])
            m = A._map(ax, per[s][key] if key != "tot" else np.abs(per[s]["tot"]),
                       norm=norm, cmap=cmap, hp_radius=None)
            if c == 0:
                ax.set_ylabel(f"{lab}\n$l$", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 6:
                ax.set_xlabel("$k$ (cyc/km)", fontsize=8)
            else:
                ax.set_xticklabels([])
        fig.colorbar(m, cax=fig.add_subplot(gs[r, -1]))

    import os
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/divergence_mechanism_{R:.0f}km_{depth_m}m.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


ALL_SHAPES = ["triangle", "square", "diamond", "hexagon", "octagon"]


def error_density(shape, R, B, ref=None):
    """Exact divergence-error spectral density (complex H): |(Hx-ref)dx +
    (Hy-ref)dy|^2 expanded with the du/dx-dv/dy cross spectrum. ref defaults to the
    footprint average F; pass ref=1.0 for the (inconsistent) point-truth version."""
    Hx, Hy = hxy(shape, R)
    if ref is None:
        ref = footprint(R)
    ax, ay = Hx - ref, Hy - ref
    return (np.abs(ax) ** 2 * B["dudx"] + np.abs(ay) ** 2 * B["dvdy"]
            + 2 * np.real(ax * np.conj(ay) * B["cxy"]))


# ------------------------------------------------------------------ point vs avg
OUT_CMP = f"{A.FIG_DIR}/compare_pointwise_average"
REFS = [("pointwise ($H\\!=\\!1$)", 1.0), ("area-average ($H\\!=\\!F$)", None)]


def _cmp_figure(title, fname, filt, filt_lab, devs, errs, pct, N):
    """Shared 2-column (pointwise | area-average) x 4-row layout: the common filter
    on top, then reference, deviation-from-reference, and error density."""
    import os
    F = footprint(15)                                    # only for the F reference map
    refmaps = [np.ones_like(F), F]
    dmax = np.nanmax([np.nanmax(np.abs(d)) for d in devs]) or 1.0
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    emax = np.nanmax([np.nanmax(e) for e in errs])
    enorm = LogNorm(emax / 10 ** DECADES, emax)
    hnorm = N["h"]

    fig = plt.figure(figsize=(6.8, 11), constrained_layout=True)
    gs = GridSpec(4, 3, width_ratios=[1, 1, 0.06], figure=fig)
    fig.suptitle(title, fontsize=10)

    axf = fig.add_subplot(gs[0, :2])                     # common filter, spans cols
    mf = A._map(axf, filt, norm=hnorm, cmap="viridis", hp_radius=None)
    axf.set_title(f"{filt_lab} (identical; only the reference differs)", fontsize=9)
    fig.colorbar(mf, cax=fig.add_subplot(gs[0, 2]))

    rows = [("reference", refmaps, hnorm, "viridis", None),
            ("filter $-$ reference", devs, dnorm, "RdBu_r", None),
            ("error density", errs, enorm, "magma", pct)]
    for r, (lab, fields, norm, cmap, ann) in enumerate(rows, start=1):
        for c in range(2):
            ax = fig.add_subplot(gs[r, c])
            m = A._map(ax, fields[c], norm=norm, cmap=cmap, hp_radius=None)
            if r == 1:
                ax.set_title(REFS[c][0], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{lab}\n$l$ (cyc/km)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 3:
                ax.set_xlabel("$k$ (cyc/km)", fontsize=8)
                ax.text(0.5, 0.02, f"RMS err {ann[c]:.0f}%", transform=ax.transAxes,
                        ha="center", va="bottom", color="w", fontsize=9)
            else:
                ax.set_xticklabels([])
        fig.colorbar(m, cax=fig.add_subplot(gs[r, 2]))

    os.makedirs(OUT_CMP, exist_ok=True)
    out = f"{OUT_CMP}/{fname}"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def compare_gradient_refs(shape, R, depth_m, B, N):
    """du/dx only: point-truth (ref=1) vs footprint-average (ref=F), side by side,
    so the penalty of the point reference is visible directly. Error % is RMS vs the
    matching truth (point gradient, or footprint-averaged gradient)."""
    Hx, _ = hxy(shape, R)
    F = footprint(R)
    S = A.mask_cross(B["dudx"])
    devs = [(Hx - r).real for r in (1.0, F)]
    errs = [A.mask_cross(np.abs(Hx - r) ** 2 * B["dudx"]) for r in (1.0, F)]
    sig = [np.sqrt(np.nansum(S)), np.sqrt(np.nansum(np.abs(F) ** 2 * S))]
    pct = [np.sqrt(np.nansum(e)) / s * 100 for e, s in zip(errs, sig)]
    return _cmp_figure(
        f"{shape} R={R:.0f} km, {depth_m} m -- $\\partial u/\\partial x$: "
        "point vs footprint-average reference",
        f"gradient_chain_{shape}_{R:.0f}km_{depth_m}m.png",
        Hx.real, "filter $H_x$", devs, errs, pct, N)


def compare_divergence_refs(shape, R, depth_m, B, N):
    """Divergence: point-truth (ref=1) vs footprint-average (ref=F). The anisotropy
    term (Hx-Hy) is reference-independent, so the drop from left to right is purely
    the isotropic part; the residual on the right is the shape's true skill."""
    Hx, Hy = hxy(shape, R)
    F = footprint(R)
    Hs = (Hx + Hy) / 2
    Sdiv = A.mask_cross(B["div"])
    devs = [(Hs - r).real for r in (1.0, F)]
    errs = [A.mask_cross(error_density(shape, R, B, ref=r)) for r in (1.0, F)]
    sig = [np.sqrt(np.nansum(Sdiv)), np.sqrt(np.nansum(np.abs(F) ** 2 * Sdiv))]
    pct = [np.sqrt(np.nansum(e)) / s * 100 for e, s in zip(errs, sig)]
    return _cmp_figure(
        f"{shape} R={R:.0f} km, {depth_m} m -- divergence: "
        "point vs footprint-average reference",
        f"divergence_chain_{shape}_{R:.0f}km_{depth_m}m.png",
        Hs.real, "filter $H_{sym}=(H_x{+}H_y)/2$", devs, errs, pct, N)


# ------------------------------------------------------------------ final figure
def final_summary_figure(depth_m, B):
    """Headline result: how well each stencil (triangle->octagon) recovers the
    footprint-averaged DIVERGENCE, and how that ties to the underlying spectrum."""
    box_km = A.BOX_DEG * af.DEG_LON_KM
    clip = lambda lam, p: (lam[lam <= box_km], p[lam <= box_km])

    pct = {}
    for R in A.SIZES_KM:
        ref = np.nansum(footprint(R) ** 2 * B["div"])
        for s in ALL_SHAPES:
            pct[(s, R)] = np.sqrt(np.nansum(error_density(s, R, B))) / np.sqrt(ref) * 100

    fig = plt.figure(figsize=(13, 9.5), constrained_layout=True)
    gs = GridSpec(3, len(ALL_SHAPES), height_ratios=[0.55, 1.15, 1.0], figure=fig)
    fig.suptitle(f"Divergence estimation vs stencil shape -- {depth_m} m, "
                 "0N/140W (error = RMS vs footprint-averaged true divergence)",
                 fontsize=13)

    # row 0: the shapes, triangle -> octagon (increasingly isotropic)
    for c, s in enumerate(ALL_SHAPES):
        ax = fig.add_subplot(gs[0, c])
        pts = af.make_arrays(15)[s]
        ax.scatter(pts[:, 0], pts[:, 1], s=25, c="tab:blue")
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(15 * np.cos(th), 15 * np.sin(th), "0.7", lw=1)
        ax.set_aspect("equal"); ax.set_xlim(-19, 19); ax.set_ylim(-19, 19)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{s}\nN={len(pts)-1}", fontsize=9)

    # row 1: headline -- divergence error % vs shape, grouped by size
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(ALL_SHAPES)); w = 0.26
    for i, R in enumerate(A.SIZES_KM):
        vals = [pct[(s, R)] for s in ALL_SHAPES]
        b = axb.bar(x + (i - 1) * w, vals, w, label=f"R = {R} km")
        axb.bar_label(b, fmt="%.0f%%", fontsize=8, padding=1)
    axb.set_xticks(x); axb.set_xticklabels(ALL_SHAPES)
    axb.set_ylabel("divergence error\n(% of footprint-avg true div RMS)")
    axb.set_title("Performance: more isotropic stencil -> smaller divergence error "
                  "(N=8 octagon ~4x better than N=3 triangle)", fontsize=10)
    axb.legend(loc="upper right"); axb.grid(axis="y", alpha=0.3)

    # row 2 left: the underlying spectrum -- the deformation reservoir dwarfs the
    # divergence signal, so any anisotropy is expensive
    axs = fig.add_subplot(gs[2, :2])
    for fld, lab, col in [("str", "stretching deformation (leaks in)", "tab:red"),
                          ("div", "divergence (the signal)", "k")]:
        lam, p = clip(*A.azimuthal(B[fld]))
        axs.loglog(lam, p, col, lw=1.6, label=lab)
    axs.invert_xaxis(); axs.set_xlabel("wavelength (km)")
    axs.set_ylabel("azimuthal power"); axs.grid(alpha=0.25, which="both")
    axs.set_title("Underlying spectrum: deformation >> divergence", fontsize=10)
    axs.legend(fontsize=8)

    # row 2 right: divergence error spectrum per shape at R=15 vs the recoverable
    # (footprint-averaged) divergence signal -- where each shape loses/corrupts it
    axe = fig.add_subplot(gs[2, 2:])
    lam, p = clip(*A.azimuthal(footprint(15) ** 2 * B["div"]))
    axe.loglog(lam, p, "k", lw=2.2, label="recoverable signal (R=15)")
    for s, col in zip(ALL_SHAPES, ["tab:purple", "tab:blue", "tab:cyan",
                                   "tab:green", "tab:olive"]):
        lam, p = clip(*A.azimuthal(error_density(s, 15, B)))
        axe.loglog(lam, p, col, lw=1.4, label=f"{s} err ({pct[(s,15)]:.0f}%)")
    axe.invert_xaxis(); axe.set_xlabel("wavelength (km)")
    axe.set_ylabel("azimuthal power"); axe.grid(alpha=0.25, which="both")
    axe.set_title("Error vs scale (R=15 km): isotropic stencils stay far below "
                  "the signal", fontsize=10)
    axe.legend(fontsize=7, ncol=2)

    import os
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/SUMMARY_divergence_vs_shape_{depth_m}m.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    k = A.depth_levels()[A.COMPARE_DEPTH_M]
    print("building spectra bundle...")
    B = spectra_bundle(k)
    N = global_norms(B)                          # one shared scale for all figures
    for R in A.SIZES_KM:
        for s in SHAPES:
            print(gradient_chain_figure(s, R, A.COMPARE_DEPTH_M, B, N))
        print(divergence_mechanism_figure(R, A.COMPARE_DEPTH_M, B, N))
    # point vs footprint-average reference, du/dx and divergence, per shape at 15 km
    for s in ALL_SHAPES:
        print(compare_gradient_refs(s, 15, A.COMPARE_DEPTH_M, B, N))
        print(compare_divergence_refs(s, 15, A.COMPARE_DEPTH_M, B, N))
    print(final_summary_figure(A.COMPARE_DEPTH_M, B))


if __name__ == "__main__":
    main()
