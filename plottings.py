"""Figure generation.

    python plottings.py figures         write figures to figures/ as PDF and PNG
    python plottings.py fetch-basemap   cache the Natural Earth relief raster

Reads data/results. No solver required.

Figure 12 plots the four stage contributions against the DE0 to SP gap, so they sum
to one. Table 8 reports the two perfect-information contributions against the
IND to SP distance.
"""
from __future__ import annotations

import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import numpy as np                                # noqa: E402

import config as C                                # noqa: E402
from analysis import (RESULTS, FIGURES, GEO, ORDER, DEEP, FIG_NAME, VARS, VLAB,
                      _load, _annual_weights)     # noqa: E402

# cartopy and rasterio are required only by the site map.
try:
    import cartopy.crs as ccrs                                     # noqa: E402
    import cartopy.feature as cfeature                             # noqa: E402
    import cartopy.io.shapereader as shpreader                     # noqa: E402
    import matplotlib.patheffects as pe                            # noqa: E402
    import rasterio                                                # noqa: E402
    from matplotlib.patches import Rectangle                       # noqa: E402
    from rasterio.enums import Resampling                          # noqa: E402
    from rasterio.features import rasterize                        # noqa: E402
    from rasterio.transform import from_bounds as affine_from_bounds   # noqa: E402
    from rasterio.windows import from_bounds                       # noqa: E402
    HAVE_MAP = True
except ImportError:
    HAVE_MAP = False

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.bbox": "tight",
})

# --------------------------------------------------------------------------
# Shared figure style. One palette for the whole paper, taken from the thermal-mass
# and uncertainty figures. Text sits at 8 pt once the figure is scaled to the width
# it is included at, lines are 1.6 pt, and every line plot marks its data points.
# --------------------------------------------------------------------------
NAVY, TEAL, CORAL, SAND, SLATE = "#14507d", "#2a9d8f", "#e67e22", "#c8a24a", "#5b6a73"
PAL = [NAVY, TEAL, CORAL, SAND, SLATE]
FILL_HI, FILL_LO = "#8ecae6", "#bfe3dc"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.6, "lines.markersize": 3.4,
    "axes.prop_cycle": plt.cycler(color=PAL),
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 140, "savefig.bbox": "tight",
})


def _legend_outside(ax, ncol=1, where="upper center"):
    """Put a legend clear of the data, above the axes."""
    ax.legend(loc=where, bbox_to_anchor=(0.5, 1.26), ncol=ncol, frameon=False)



def _save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES, name + ".pdf"))
    fig.savefig(os.path.join(FIGURES, name + ".png"))
    plt.close(fig)
    print(f"  {name}")



def fig_concept(main):
    d = main["virum"]["series"]
    I = np.array(d["I"]); E = np.array(d["E"]); CT = np.array(d["CURT"]); PV = np.array(d["PV"])
    cap = float(main["virum"]["hosting"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), sharey=True)
    for ax, (a, b, ttl) in zip(axes, [(0, 168, "Winter week (deficit-dominated)"),
                                      (336, 504, "Summer week (surplus-dominated)")]):
        h = np.arange(b - a)
        ax.fill_between(h, 0, PV[a:b], color=SAND, alpha=.28, lw=0,
                        label="PV generation")
        ax.fill_between(h, 0, I[a:b], color=NAVY, alpha=.75, lw=0,
                        label="grid import (deficit)")
        ax.fill_between(h, 0, -(E[a:b] + CT[a:b]), color=TEAL, alpha=.75, lw=0,
                        label="export + curtailment (surplus)")
        # the clipping the text describes happens at the hosting limit
        ax.axhline(-cap, color=CORAL, lw=1.2, ls="--", label="hosting limit")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("hour [h]")
        ax.set_title(ttl)
    axes[0].set_ylabel("power [kW]")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.30), ncol=4, frameon=False)
    _save_fig(fig, "F1_concept")


def fig_waterfall(main):
    """Coordination gap by stage, from a zero baseline.

    The earlier version plotted annual cost per stage with the y axis starting near the
    minimum, which made a few per cent look like a collapse. Plotting the gap itself from
    zero gives a true waterfall and removes the truncated-axis objection.
    """
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.8), sharey=True)
    for ax, pid in zip(axes, DEEP):
        r = main[pid]
        c = {k: r["variants"][k]["cost_kEUR"] for k in VARS}
        c["SP"] = r["SP"]["cost_kEUR"]
        gap = [c[k] - c["SP"] for k in VARS] + [0.0]
        labels = VLAB + ["SP"]
        # experiment 1 is the first step, experiment 2 runs from IND onward
        cols = [NAVY, NAVY, TEAL, TEAL, TEAL, SLATE]
        ax.bar(range(len(gap)), gap, color=cols, width=0.68)
        for i, v in enumerate(gap):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6)
        ax.axvline(1.5, color=SLATE, ls="--", lw=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(FIG_NAME[pid], fontsize=8)
    # one limit across the four panels, set by the largest gap so no bar is clipped
    top = max(main[q]["variants"]["DE0"]["cost_kEUR"] - main[q]["SP"]["cost_kEUR"]
              for q in DEEP)
    axes[0].set_ylim(0, top * 1.16)
    axes[0].set_ylabel("coordination gap [k\u20ac]")
    _save_fig(fig, "F2_waterfall")


def fig_prices(main):
    d = main["virum"]["series"]
    pi = np.array(d["pi"]); tau = np.array(d["tau"])
    vol = np.array(d.get("vol", np.ones_like(pi)))
    nw = len(tau) // 168
    binding = [int(np.sum(tau[wk * 168:(wk + 1) * 168] > 1e-4)) for wk in range(nw)]
    wk = int(np.argmax(binding)); a = wk * 168; b = a + 168
    h = np.arange(168)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 2.9))
    # P2P price solid only in active pool hours (V_t > 1 kW); low-volume hours, where
    # the multiplier is one solver-selected dual, are greyed and dotted
    act = vol[a:b] > 1.0; pib = pi[a:b]
    ax.plot(h, np.where(act, pib, np.nan), color=NAVY, lw=1.6, label=r"P2P price $\pi$ (active hours)")
    ax.plot(h, np.where(~act, pib, np.nan), color="#b0b0b0", lw=0.8, ls=":", label=r"$\pi$ (low-volume dual)")
    ax.plot(h, tau[a:b], color=CORAL, lw=2.0, label=r"congestion price $\tau$")
    ax.fill_between(h, 0, tau[a:b], color=CORAL, alpha=.18)
    ax.set_xlabel("hour [h]"); ax.set_ylabel("price [€/kWh]")
    ax.set_title("Prices in the binding week", fontsize=8)
    _legend_outside(ax, ncol=3)
    wa = _annual_weights(len(tau))
    order = np.argsort(tau)[::-1]
    xh = np.cumsum(wa[order])
    ax2.plot(xh, tau[order], color=CORAL, lw=1.2)
    ax2.fill_between(xh, 0, tau[order], color=CORAL, alpha=.18)
    ax2.set_xlabel("annual hours [h]"); ax2.set_ylabel(r"$\tau$ [€/kWh]")
    ax2.set_title("Annual congestion-price duration", fontsize=8)
    _save_fig(fig, "F4_prices")


def fig_pareto(par):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for k, pid in enumerate(p for p in ORDER if p in par):
        pts = sorted([p for p in par[pid] if 0.0 <= p["SSR"] <= 1.0], key=lambda p: p["SSR"])
        if not pts:
            continue
        col = PAL[k % len(PAL)]
        ss = [p["SSR"] for p in pts]; cost = [p["cost_kEUR"] for p in pts]
        if len(pts) == 1:
            ax.plot(ss, cost, marker="o", ms=6, lw=0, color=col,
                    label=FIG_NAME.get(pid, pid))
            ax.annotate(f"{FIG_NAME.get(pid, pid)}, wall at optimum", (ss[0], cost[0]),
                        xytext=(7, -1), textcoords="offset points", fontsize=7,
                        color=col, va="center")
        else:
            ax.plot(ss, cost, marker="o", lw=1.6, color=col, label=FIG_NAME.get(pid, pid))
        ax.plot(ss[-1], cost[-1], marker="*", ms=12, mec="k", mew=0.5, color=col, zorder=5)
    ax.set_xlabel("self-sufficiency [-]")
    ax.set_ylabel("annual net cost [k\u20ac]")
    ax.legend(loc="lower right", frameon=False, ncol=1)
    _save_fig(fig, "F5_selfsuff_cost")


def fig_sensitivity(sens):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.0))

    H = sens["hosting"]; s = [r["scale"] for r in H]
    tot = [r["DE0"]["cost_kEUR"] - r["SP"]["cost_kEUR"] for r in H]
    res = [r["DE_M3"]["cost_kEUR"] - r["SP"]["cost_kEUR"] for r in H]
    a1.fill_between(s, res, tot, color=FILL_HI, alpha=.45, label="recovered at M3")
    a1.plot(s, tot, marker="o", color=NAVY, label="total gap (DE0 - SP)")
    a1.plot(s, res, marker="s", color=TEAL, label="residual at M3 (M3 - SP)")
    a1.set_xlabel("hosting limit / base [-]")
    a1.set_ylabel("coordination gap [k\u20ac]")
    _legend_outside(a1)

    Cb = sens["carbon"]; cp = [r["carbon"] * 1000 for r in Cb]
    a2.plot(cp, [r["DE0"]["curt_MWh"] for r in Cb], marker="o", color=NAVY, label="DE0")
    a2.plot(cp, [r["DE_M3"]["curt_MWh"] for r in Cb], marker="s", color=TEAL,
            label="M3 grid-aware")
    a2.set_xlabel("carbon price [\u20ac/t]")
    a2.set_ylabel("curtailment [MWh]")
    _legend_outside(a2, ncol=2)

    # M3 coincides with SP, so M3 is drawn dashed on top of a solid SP and both read
    St = sens["storage"]; ss = [r["scale"] for r in St]
    a3.plot(ss, [r["DE0"]["SSR"] for r in St], marker="o", color=NAVY, label="DE0")
    a3.plot(ss, [r["SP"]["SSR"] for r in St], marker="^", color=SLATE, label="SP")
    a3.plot(ss, [r["DE_M3"]["SSR"] for r in St], marker="s", color=TEAL, ls="--",
            label="M3 (coincides with SP)")
    a3.set_xlabel("storage capacity / base [-]")
    a3.set_ylabel("self-sufficiency [-]")
    _legend_outside(a3)

    _save_fig(fig, "F6_sensitivity")


def fig_sensitivity2(sens):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 3.0))
    R = sens["res"]; r = [x["ratio"] for x in R]
    a1.plot(r, [x["SP"]["SSR"] for x in R], marker="o", ms=3, color=NAVY, label="self-sufficiency")
    a1.plot(r, [x["SP"]["LPI_os"] for x in R], marker="s", ms=3, color=CORAL, label="oversupply")
    a1.set_xlabel("renewable penetration [-]"); a1.set_ylabel("fraction [-]")
    _legend_outside(a1, ncol=2)
    F = sens["flex"]; al = [x["alpha"] for x in F]
    a2.plot(al, [x["DE0"]["SSR"] for x in F], marker="o", ms=3, label="DE0")
    a2.plot(al, [x["DE_M3"]["SSR"] for x in F], marker="s", ms=3, label="M3")
    a2.plot(al, [x["SP"]["SSR"] for x in F], marker="^", ms=3, label="SP")
    a2.set_xlabel("demand-response depth [-]"); a2.set_ylabel("self-sufficiency [-]")
    _legend_outside(a2, ncol=3)
    Fi = sens["feedin"]; fr = [x["ratio"] for x in Fi]
    a3.plot(fr, [x["DE0"]["cost_kEUR"] - x["SP"]["cost_kEUR"] for x in Fi], marker="o", color=NAVY)
    a3.set_xlabel("feed-in / import price [-]"); a3.set_ylabel("coordination gap [k€]")
    _save_fig(fig, "F7_sensitivity2")


def fig_energy_balance(main):
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    x = np.arange(len(ORDER)); wbar = 0.36; ymax = 0.0
    for i, pid in enumerate(ORDER):
        sp = main[pid]["SP"]
        pv, imp, exp, curt = sp["pv_MWh"], sp["imp_MWh"], sp["exp_MWh"], sp["curt_MWh"]
        sc = max(pv - exp - curt, 0.0)
        ax.bar(i - 0.20, sc, wbar, color="#2a9d8f", zorder=3, label="self-consumed RES" if i == 0 else "")
        ax.bar(i - 0.20, exp, wbar, bottom=sc, color=TEAL, zorder=3, label="exported" if i == 0 else "")
        ax.bar(i - 0.20, curt, wbar, bottom=sc + exp, color=CORAL, zorder=3, label="curtailed" if i == 0 else "")
        ax.bar(i + 0.20, sc, wbar, color="#2a9d8f", zorder=3)
        ax.bar(i + 0.20, imp, wbar, bottom=sc, color="#e9a13b", zorder=3, label="imported" if i == 0 else "")
        ymax = max(ymax, sc + exp + curt, sc + imp)
    ax.set_xticks(x); ax.set_xticklabels([FIG_NAME[p] for p in ORDER], fontsize=8)
    ax.set_ylabel("annual energy [MWh]"); ax.set_ylim(0, ymax * 1.30); ax.margins(x=0.02)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    _save_fig(fig, "F8_energy_balance")


def fig_last20(main):
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    x = np.arange(len(ORDER))
    for i, p in enumerate(ORDER):
        sp = main[p]["SP"]
        deficit = sp["imp_MWh"] / max(sp["dem_MWh"], 1e-9)
        exp_f = sp["exp_MWh"] / max(sp["pv_MWh"], 1e-9)
        curt_f = sp["curt_MWh"] / max(sp["pv_MWh"], 1e-9)
        ax.bar(i, exp_f, 0.6, color=TEAL, zorder=3, label="exported" if i == 0 else "")
        ax.bar(i, curt_f, 0.6, bottom=exp_f, color=CORAL, zorder=3, label="curtailed" if i == 0 else "")
        ax.bar(i, -deficit, 0.6, color="#e9a13b", zorder=3, label="imported (deficit)" if i == 0 else "")
        ax.text(i, exp_f + curt_f + 0.01, f"{exp_f + curt_f:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(i, -deficit - 0.01, f"{deficit:.2f}", ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([FIG_NAME[p] for p in ORDER], fontsize=7.5)
    ax.set_ylabel("fraction [-]"); ax.set_ylim(-0.52, 0.60)
    ax.legend(fontsize=7, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    _save_fig(fig, "F9_last20")


def fig_seasonal(main):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
    for ax, pid in zip(axes, DEEP):
        s = main[pid]["series"]
        I = np.array(s["I"]); E = np.array(s["E"]); CT = np.array(s["CURT"])
        dem = np.array(s["dem"]); PV = np.array(s["PV"])
        n = len(I) // 4
        ss, os_ = [], []
        for k in range(4):
            sl = slice(k * n, (k + 1) * n)
            ss.append(1 - I[sl].sum() / max(dem[sl].sum(), 1e-9))
            os_.append((E[sl].sum() + CT[sl].sum()) / max(PV[sl].sum(), 1e-9))
        xk = np.arange(4)
        ax.bar(xk - 0.205, ss, 0.39, color=NAVY, label="self-sufficiency")
        ax.bar(xk + 0.205, os_, 0.39, color=CORAL, label="oversupply")
        ax.set_xticks(xk); ax.set_xticklabels(["W", "Sp", "Su", "Au"], fontsize=7)
        ax.set_title(FIG_NAME[pid], fontsize=8); ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction [-]"); axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    _save_fig(fig, "F10_seasonal")


def fig_uncertainty(u):
    draws = u["draws"]; st = u["stats"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.0))
    gaps = [d["gap_pct"] for d in draws]
    ax.hist(gaps, bins=12, color=TEAL, alpha=.8)
    ax.axvline(st["gap_pct"]["mean"], color=CORAL, lw=1.2, label=f"mean {st['gap_pct']['mean']:.2f}%")
    ax.set_xlabel("coordination gap [% of DE0 cost]"); ax.set_ylabel("count")
    ax.set_title("Gap distribution across seeds", fontsize=8); ax.legend(fontsize=7, frameon=False)
    labels = ["shared forecast", "diagnostic", "P2P market", "grid price"]
    keys = ["share_info", "share_foresight", "share_pool", "share_grid"]
    means = [st[k]["mean"] for k in keys]; sds = [st[k]["std"] for k in keys]
    ax2.bar(np.arange(4), means, 0.6, yerr=sds, capsize=3, color=["#8d99ae", "#6c8ead", TEAL, "#014f86"])
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("share of the DE0 to SP gap [-]")
    # every bar divides by the same DE0-to-SP gap, so the four sum to one. Table 8
    # states the two perfect-information shares against IND-to-SP instead.
    ax2.set_title("Stage contributions, share of the DE0 to SP gap", fontsize=8)
    _save_fig(fig, "F12_uncertainty")




# Bare-page treatment: land only, no frame, no graticule.
PLAIN = True

RASTERS = os.path.join(os.path.expanduser("~"), ".local", "share", "cartopy", "rasters")
CANDIDATES = [
    os.path.join(RASTERS, "NE1_HR_LC_SR_W_DR", "NE1_HR_LC_SR_W_DR.tif"),
    os.path.join(RASTERS, "NE1_50M_SR_W", "NE1_50M_SR_W.tif"),
]

# the raster window, cropped before projection. Deliberately wider than the displayed
# extent, because a conic projection of a lat-lon box is curved and the rectangular axes
# would otherwise show uncovered corners.
LON0, LON1, LAT0, LAT1 = -40.0, 70.0, 22.0, 80.0
VIEW = (-12.0, 40.0, 34.0, 66.0)

NAVY, CORAL = "#14507d", "#e67e22"
DARK = "#3a3a3a"
WHITEN = 0.18          # blend toward white so labels read over the relief
TARGET_DPI = 250
FIG_W_IN = 7.1         # full text width of the elsarticle layout

# the sites are what the reader has to find, so the markers carry the weight
S_DEEP, S_ILLUS = 260, 190
LABEL_PT = 10

SITES = [
    (-8.62, 52.66, "Limerick (IE)", "deep", (6, 4)),
    (12.48, 55.79, "Virum (DK)", "deep", (6, 4)),
    (16.55, 59.61, "Västerås (SE)", "deep", (6, 4)),
    (13.40, 52.52, "Berlin (DE)", "deep", (6, 4)),
    (15.44, 47.07, "Graz (AT)", "deep", (11, -15)),
    (26.10, 44.43, "Bucharest (RO)", "illus", (6, 4)),
    (32.49, 37.87, "Konya (TR)", "illus", (6, 4)),
]


def load_crop():
    """Read only the European window, downsampled to what the printed size needs."""
    path = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("no Natural Earth raster cached, run fetch_ne_raster.py first")
    with rasterio.open(path) as src:
        win = from_bounds(LON0, LAT0, LON1, LAT1, src.transform)
        # enough pixels for TARGET_DPI across the printed width, with headroom for the warp
        want_w = int(FIG_W_IN * TARGET_DPI * 1.25)
        full_w = int(win.width)
        scale = min(1.0, want_w / full_w)
        out_w = max(1, int(full_w * scale))
        out_h = max(1, int(win.height * scale))
        img = src.read(
            indexes=[1, 2, 3], window=win, out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear, boundless=True, fill_value=255,
        )
    arr = np.transpose(img, (1, 2, 0)).astype(np.float32) / 255.0
    # blend toward white, keeping the raster's own hues rather than desaturating
    arr = arr * (1.0 - WHITEN) + WHITEN
    # the water carries no information in this figure, so it goes white and the coastline
    # alone marks where land ends
    cov = water_coverage(out_h, out_w)[..., None]
    arr = arr * (1.0 - cov) + cov
    return os.path.basename(path), np.clip(arr, 0, 1), (out_w, out_h)


def water_coverage(out_h, out_w, ss=3):
    """Fractional water cover per raster pixel, from the 10 m ocean and lake polygons.

    Burned into the raster rather than drawn as a white vector overlay. The 10 m ocean
    polygon as vector geometry took the PDF to 6.6 MB, over the size ceiling, while burning
    it costs nothing. Rasterizing at ss times the output grid and averaging down gives
    fractional cover, so the coast is anti-aliased instead of stair-stepped.
    """
    tr = affine_from_bounds(LON0, LAT0, LON1, LAT1, out_w * ss, out_h * ss)
    geoms = [g
             for layer in ("ocean", "lakes")
             for g in shpreader.Reader(shpreader.natural_earth(
                 resolution="10m", category="physical", name=layer)).geometries()]
    burn = rasterize(geoms, out_shape=(out_h * ss, out_w * ss), transform=tr,
                     fill=0, default_value=1, dtype="uint8")
    return burn.reshape(out_h, ss, out_w, ss).mean(axis=(1, 3)).astype(np.float32)


def scale_bar(ax, proj, km=500, pad=0.06):
    """Alternating black and white bar, exact where it is drawn.

    Albers is equal-area, so distance along a parallel is not preserved. Drawing the bar at
    its nominal length in projection metres would be about 1.6 per cent short at this
    latitude, so the drawn length is corrected by the local scale factor measured on the
    ellipsoid at the bar's own position.
    """
    from cartopy.geodesic import Geodesic

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + (x1 - x0) * pad
    by = y0 + (y1 - y0) * pad

    pc = ccrs.PlateCarree()
    a = pc.transform_point(bx, by, proj)
    b = pc.transform_point(bx + km * 1000.0, by, proj)
    true_m = Geodesic().inverse(a, b)[0][0]
    bar_m = km * 1000.0 * (km * 1000.0 / true_m)   # stretch so the ground length is exact

    n = 4
    seg = bar_m / n
    h = (y1 - y0) * 0.010
    for i in range(n):
        ax.add_patch(Rectangle((bx + i * seg, by), seg, h,
                               facecolor=("black" if i % 2 == 0 else "white"),
                               edgecolor="black", linewidth=0.5, zorder=12))
    ax.text(bx + bar_m / 2, by + h * 1.9, f"{km} km", ha="center", va="bottom",
            fontsize=7, color="black", zorder=12,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])


def fig_site_map():
    name, arr, (w, h) = load_crop()
    src_crs = ccrs.PlateCarree()
    mapproj = ccrs.AlbersEqualArea(central_longitude=14.0, central_latitude=50.0,
                                   standard_parallels=(40.0, 60.0))

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_W_IN * 0.82),
                           subplot_kw={"projection": mapproj})
    ax.set_extent(list(VIEW), crs=src_crs)

    ax.imshow(arr, origin="upper", extent=[LON0, LON1, LAT0, LAT1],
              transform=src_crs, interpolation="bilinear", zorder=0)

    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m"),
                   facecolor="none", edgecolor=DARK, linewidth=0.35, zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "10m"),
        facecolor="none", edgecolor=DARK, linewidth=0.30, zorder=3)

    if not PLAIN:
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"},
                          linewidth=0.3, color="0.4", alpha=0.30, zorder=4)
        gl.xlocator = plt.matplotlib.ticker.FixedLocator(list(range(-10, 41, 10)))
        gl.ylocator = plt.matplotlib.ticker.FixedLocator(list(range(35, 66, 10)))
        gl.rotate_labels = False      # conic default tilts them, which reads badly
        gl.xlabel_style = {"size": 7, "color": DARK}
        gl.ylabel_style = {"size": 7, "color": DARK}

    ax.plot([13.40, 15.44], [52.52, 47.07], color=DARK, lw=0.8, ls="--",
            transform=src_crs, zorder=6)

    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    for lon, lat, label, tier, off in SITES:
        deep = tier == "deep"
        ax.scatter([lon], [lat], s=S_DEEP if deep else S_ILLUS,
                   marker="*" if deep else "o",
                   facecolor=NAVY if deep else "none",
                   edgecolor="white" if deep else CORAL,
                   linewidths=1.0 if deep else 2.2,
                   transform=src_crs, zorder=8)
        if deep:   # a second pass gives the star a thin white rim over the relief
            ax.scatter([lon], [lat], s=S_DEEP, marker="*", facecolor="none",
                       edgecolor="white", linewidths=1.0, transform=src_crs, zorder=9)
        ax.annotate(label, xy=(lon, lat), xycoords=src_crs._as_mpl_transform(ax),
                    xytext=off, textcoords="offset points", fontsize=LABEL_PT,
                    color="black", fontweight="medium", zorder=10, path_effects=halo)

    ax.scatter([], [], marker="*", s=S_DEEP, facecolor=NAVY, edgecolor="white",
               linewidths=1.0, label="Deep configuration")
    ax.scatter([], [], marker="o", s=S_ILLUS, facecolor="none", edgecolor=CORAL,
               linewidths=2.2, label="Illustrative configuration")
    leg = ax.legend(loc="upper left", fontsize=LABEL_PT, frameon=not PLAIN, framealpha=0.85,
                    edgecolor="0.6", facecolor="white")
    leg.set_zorder(11)

    scale_bar(ax, mapproj)

    if PLAIN:
        ax.patch.set_visible(False)       # no painted ground behind the land
        ax.spines["geo"].set_visible(False)
    else:
        for s in ax.spines.values():
            s.set_edgecolor("black")
            s.set_linewidth(0.6)

    base = os.path.join(FIGURES, "F0_testbed_map")
    fig.savefig(base + ".pdf", bbox_inches="tight", dpi=TARGET_DPI)
    fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    mb = os.path.getsize(base + ".pdf") / 1e6
    eff = w / FIG_W_IN
    print(f"base raster      : {name}")
    print(f"crop pixels      : {w} x {h}")
    print(f"effective dpi    : {eff:.0f} at {FIG_W_IN} in wide")
    print(f"PDF size         : {mb:.2f} MB  {'OK' if mb < 5 else 'OVER 5 MB'}")


def fig_residual_imbalance(main):
    """Figure 4. Shares of demand and renewable output, and absolute annual energy."""
    C_SELF, C_EXP, C_CURT, C_IMP = TEAL, "#8ecae6", CORAL, NAVY
    x = np.arange(len(ORDER))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.9))

    dmax = 0.0
    for i, p in enumerate(ORDER):
        sp = main[p]["SP"]
        deficit = sp["imp_MWh"] / max(sp["dem_MWh"], 1e-9)
        exp_f = sp["exp_MWh"] / max(sp["pv_MWh"], 1e-9)
        curt_f = sp["curt_MWh"] / max(sp["pv_MWh"], 1e-9)
        axL.bar(i, exp_f, 0.6, color=C_EXP, zorder=3)
        axL.bar(i, curt_f, 0.6, bottom=exp_f, color=C_CURT, zorder=3)
        axL.bar(i, -deficit, 0.6, color=C_IMP, zorder=3)
        axL.text(i, exp_f + curt_f + 0.012, f"{exp_f + curt_f:.2f}", ha="center",
                 va="bottom", fontsize=6.5)
        axL.text(i, -deficit - 0.012, f"{deficit:.2f}", ha="center", va="top", fontsize=6.5)
        dmax = max(dmax, deficit, exp_f + curt_f)
    axL.axhline(0, color="k", lw=0.7)
    axL.set_ylabel("fraction of demand or of renewable output [-]")
    axL.set_ylim(-dmax * 1.22, dmax * 1.22)

    wbar, ymax = 0.36, 0.0
    for i, pid in enumerate(ORDER):
        sp = main[pid]["SP"]
        pv, imp, exp, curt = sp["pv_MWh"], sp["imp_MWh"], sp["exp_MWh"], sp["curt_MWh"]
        sc = max(pv - exp - curt, 0.0)
        axR.bar(i - 0.20, sc, wbar, color=C_SELF, zorder=3)
        axR.bar(i - 0.20, exp, wbar, bottom=sc, color=C_EXP, zorder=3)
        axR.bar(i - 0.20, curt, wbar, bottom=sc + exp, color=C_CURT, zorder=3)
        axR.bar(i + 0.20, sc, wbar, color=C_SELF, zorder=3)
        axR.bar(i + 0.20, imp, wbar, bottom=sc, color=C_IMP, zorder=3)
        ymax = max(ymax, sc + exp + curt, sc + imp)
    axR.set_ylabel("annual energy [MWh]")
    axR.set_ylim(0, ymax * 1.12)
    axR.text(0.5, 0.97, "left bar supply, right bar demand", transform=axR.transAxes,
             ha="center", va="top", fontsize=7, color="0.35")

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels([FIG_NAME[p] for p in ORDER], fontsize=7.5, rotation=20,
                           ha="right")
        ax.margins(x=0.02)

    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               [(C_SELF, "self-consumed renewable"), (C_EXP, "exported"),
                (C_CURT, "curtailed"), (C_IMP, "imported (deficit)")]]
    fig.legend(handles=handles, fontsize=8, ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig, "F4_residual_imbalance")


def fig_thermal_mass():
    """Figure 10. Virum pre-heating sensitivity against the comfort band half-width."""
    d = _load("thermal_mass.json")
    loss = sorted([r for r in d["rows"] if r["variant"] == "pre loss"],
                  key=lambda r: r["half_K"])
    nol = sorted([r for r in d["rows"] if r["variant"] == "pre noloss"],
                 key=lambda r: r["half_K"])
    if not loss or not nol:
        raise SystemExit("expected the pre-heating variants in thermal_mass.json")
    x = [r["half_K"] for r in loss]
    C_SSR, C_OS, C_GAP, C_CURT = "#2a9d8f", "#c1121f", "#014f86", "#e9a13b"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axL.plot(x, [r["SSR"] for r in loss], "-o", color=C_SSR, ms=3.5,
             label="self-sufficiency")
    axL.plot(x, [r["SSR"] for r in nol], "--", color=C_SSR, lw=1.1)
    axL.plot(x, [r["LPI_os"] for r in loss], "-s", color=C_OS, ms=3.5, label="oversupply")
    axL.plot(x, [r["LPI_os"] for r in nol], "--", color=C_OS, lw=1.1)
    axL.set_xlabel("comfort band half-width [K]")
    axL.set_ylabel("fraction [-]")
    axL.legend(fontsize=7.5, frameon=False, loc="center right")
    axL.spines["right"].set_visible(False)

    axR.plot(x, [r["gap_kEUR"] for r in loss], "-o", color=C_GAP, ms=3.5,
             label="coordination gap")
    axR.plot(x, [r["gap_kEUR"] for r in nol], "--", color=C_GAP, lw=1.1)
    axR.set_xlabel("comfort band half-width [K]")
    axR.set_ylabel("coordination gap [k\u20ac]", color=C_GAP)
    axR.tick_params(axis="y", labelcolor=C_GAP)

    ax2 = axR.twinx()
    ax2.plot(x, [r["curt_MWh"] for r in loss], "-^", color=C_CURT, ms=3.5,
             label="curtailment")
    ax2.plot(x, [r["curt_MWh"] for r in nol], "--", color=C_CURT, lw=1.1)
    ax2.set_ylabel("curtailment [MWh]", color=C_CURT)
    ax2.tick_params(axis="y", labelcolor=C_CURT)
    ax2.spines["top"].set_visible(False)

    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="center right")

    handles = [matplotlib.lines.Line2D([], [], color="0.3", ls="-", label="with fabric loss"),
               matplotlib.lines.Line2D([], [], color="0.3", ls="--",
                                       label="no loss, upper bound")]
    fig.legend(handles=handles, fontsize=8, ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, "F13_thermal_mass")


def generate_figures():
    main = _load("main_results.json")
    fig_concept(main)
    fig_waterfall(main)
    fig_prices(main)
    fig_last20(main)
    fig_seasonal(main)
    fig_energy_balance(main)
    fig_pareto(_load("pareto.json"))
    fig_sensitivity(_load("sensitivity_virum.json"))
    fig_sensitivity2(_load("sensitivity_virum.json"))
    fig_uncertainty(_load("uncertainty.json"))
    fig_residual_imbalance(main)
    fig_thermal_mass()
    if not HAVE_MAP:
        print("  site map skipped, cartopy and rasterio are not installed")
    else:
        try:
            fig_site_map()
        except Exception as e:                                   # noqa: BLE001
            print(f"  site map skipped: {e}")


def fetch_basemap():
    """Cache the Natural Earth shaded-relief raster (public domain, ~700 MB)."""
    import io
    import urllib.request
    import zipfile

    out = os.path.join(os.path.expanduser("~"), ".local", "share", "cartopy", "rasters")
    os.makedirs(out, exist_ok=True)
    for name, url in [
        ("NE1_HR_LC_SR_W_DR",
         "https://naciscdn.org/naturalearth/10m/raster/NE1_HR_LC_SR_W_DR.zip"),
        ("NE1_50M_SR_W",
         "https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip"),
    ]:
        tif = os.path.join(out, name, name + ".tif")
        if os.path.exists(tif):
            print(f"  already cached: {name}")
            return tif
        print(f"  downloading {name} from {url}")
        try:
            with urllib.request.urlopen(url) as r:
                blob = r.read()
            zipfile.ZipFile(io.BytesIO(blob)).extractall(os.path.join(out, name))
            print(f"  cached at {tif}")
            return tif
        except Exception as e:                                   # noqa: BLE001
            print(f"  failed: {e}")
    raise SystemExit("could not obtain a Natural Earth raster")


def main_cli(argv):
    if len(argv) < 2 or argv[1] not in ("figures", "fetch-basemap"):
        print(__doc__)
        return 1
    if argv[1] == "fetch-basemap":
        fetch_basemap()
        return 0
    os.makedirs(FIGURES, exist_ok=True)
    generate_figures()
    print(f"\nFigures written to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli(sys.argv))
