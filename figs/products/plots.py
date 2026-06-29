"""SPC-style plotting: filled probability contours, categorical-risk fills, and
significant-severe hatching on a CONUS Lambert map.

Mirrors the SPC convective-outlook look. Uses the FIGS grid lat/lon and cartopy;
all functions render to a PNG and return the path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CATEGORY_NAMES, PIB_COLORS, PIB_LABELS, PRODUCTS, SPC_PROB_LEVELS
from ..data import grid

# SPC probabilistic-outlook fill colors, per hazard, aligned to config.SPC_PROB_LEVELS.
# The lowest level is a FIGS custom addition below the SPC levels, drawn a fainter
# pastel of the next color (tor 1% pastel green; wind/hail 2% pastel brown).
#   tornado : 1/2/5/10/15/30/45/60 %  -> pastelGreen/green/brown/gold/red/magenta/purple/blue
#   wind    : 2/5/15/30/45/60/75/90 % -> pastelBrown/brown/gold/red/magenta/purple/blue/indigo
#   hail    : 2/5/15/30/45/60 %       -> pastelBrown/brown/gold/red/magenta/purple
PROB_COLORS = {
    "tor": ["#a6dba6", "#008b00", "#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee", "#104e8b"],
    "wind": ["#d6b48c", "#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee", "#104e8b", "#4b0082"],
    "hail": ["#d6b48c", "#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee"],
}

# SPC categorical-risk fill colors. 0=TSTM uses the official SPC light green; 1..5 =
# MRGL/SLGT/ENH/MDT/HIGH.
CAT_COLORS = {0: "#c1e9c1", 1: "#66a366", 2: "#ffe066", 3: "#ffa366", 4: "#e06666", 5: "#ee99ee"}

# EF-scale (tornado intensity) discrete palette, EF0 -> EF4+ in bin order.
# EF4+ uses the EF4 colour (the EF5/cat5 colour A188FC is intentionally unused).
EF_COLORS = ["#4DFFFF", "#FFFFD9", "#FFD98C", "#FF9E59", "#FF738A"]

# CIG / intensity hatching: CIG1 dotted, CIG2 solid diagonal, CIG3 solid crosshatch.
CIG_HATCH = {1: "...", 2: "//", 3: "xx"}


def overlay_cig(ax, lon, lat, cig):
    """Overlay CIG intensity hatching on a cartopy ``ax``. ``cig`` is an (ny, nx)
    int array (0 none .. 3 CIG3); each band is hatched by its exact level so the
    nested CIG1/2/3 areas read as concentric dotted / diagonal / crosshatch, and a
    black line outlines the edge of each CIG region (>=1, >=2, >=3), matching the
    way the SPC outlook contours are bounded."""
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    for lvl in (1, 2, 3):                       # hatch each exact band (concentric)
        band = (cig == lvl) if lvl < 3 else (cig >= 3)
        if band.any():
            ax.contourf(lon, lat, band.astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=[CIG_HATCH[lvl]], transform=pc)
    for lvl in (1, 2, 3):                       # outline each CIG region edge (nested)
        region = cig >= lvl
        if region.any():
            ax.contour(lon, lat, region.astype(float), levels=[0.5], colors="black",
                       linewidths=0.6, transform=pc)


# Plan-view map extent [lon_min, lon_max, lat_min, lat_max] for ALL product
# plots. Override in one place via ``set_extent(...)`` (e.g. from a notebook).
MAP_EXTENT = [-120, -74, 23, 50]

# Module-level run context — set automatically by predict_or_load / predict_forecast
# so every plot function can annotate "init … f##–##" without caller changes.
_RUN_CONTEXT: dict = {}   # keys: "run" (datetime), "fxx_list" (list[int])


def set_extent(extent) -> None:
    """Set the plan-view extent used by every plot here (lon_min, lon_max,
    lat_min, lat_max). Pass ``None`` to let cartopy auto-fit the data."""
    global MAP_EXTENT
    MAP_EXTENT = list(extent) if extent is not None else None


def set_run_context(run, fxx_list) -> None:
    """Record the current HRRR run + fxx list so all subsequent plot calls can
    annotate figures automatically. Called by predict_or_load / predict_forecast."""
    _RUN_CONTEXT["run"] = run
    _RUN_CONTEXT["fxx_list"] = sorted(int(f) for f in fxx_list)


def _run_tag(fxx=None) -> str:
    """Build the run-context tag string, or '' if no context is set."""
    run = _RUN_CONTEXT.get("run")
    if run is None:
        return ""
    fxx_list = _RUN_CONTEXT.get("fxx_list", [])
    if fxx is not None:
        fxx_part = f"f{int(fxx):02d}"
    elif fxx_list:
        fxx_part = f"f{min(fxx_list):02d}–{max(fxx_list):02d}"
    else:
        fxx_part = ""
    return f"init {run:%Y-%m-%d %HZ}  {fxx_part}".strip()


def _with_run(title: str, fxx=None) -> str:
    """Append 'init … f##' as a second title line, or return title unchanged."""
    tag = _run_tag(fxx)
    return f"{title}\n{tag}" if tag else title


def _base_ax(figsize=(10, 6)):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.4")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    if MAP_EXTENT is not None:
        ax.set_extent(MAP_EXTENT)
    return fig, ax


def plot_probability(prob: np.ndarray, hazard: str, title: str, out_path: str | Path | None = None,
                     sig: np.ndarray | None = None, cig: np.ndarray | None = None,
                     fxx: int | None = None) -> str:
    """SPC-style map: filled probability contours (prob in 0..1) + CIG intensity
    hatching. ``cig`` is an (ny, nx) int array (0 none .. 3 CIG3) -> dotted /
    diagonal / crosshatch. ``sig`` (legacy) hatches a significant field at 10%."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    lat, lon = grid.figs_latlon()
    prob_levels = SPC_PROB_LEVELS[hazard]
    levels = list(prob_levels) + [1.0]
    fig, ax = _base_ax()
    cf = ax.contourf(lon, lat, prob, levels=levels, colors=PROB_COLORS[hazard],
                     transform=ccrs.PlateCarree(), extend="neither")
    if cig is not None:
        # CIG is defined per the conditional-intensity dist everywhere; only hatch
        # it inside the drawn probability area (>= the lowest plotted contour).
        overlay_cig(ax, lon, lat, np.where(prob >= prob_levels[0], cig, 0))
    elif sig is not None:
        ax.contourf(lon, lat, sig, levels=[0.10, 1.0], colors="none", hatches=["xx"],
                    transform=ccrs.PlateCarree())
        ax.contour(lon, lat, sig, levels=[0.10], colors="black", linewidths=0.8,
                   transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=prob_levels)
    cbar.ax.set_xticklabels([f"{int(p*100)}%" for p in prob_levels])
    ax.set_title(_with_run(title, fxx))
    out_path = out_path or (PRODUCTS / f"prob_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_categorical(category: np.ndarray, hazard: str, title: str,
                     out_path: str | Path | None = None, fxx: int | None = None) -> str:
    """Filled SPC categorical risk (integer levels 0..5; 0=TSTM drawn, <0 = no
    risk, not drawn)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = grid.figs_latlon()
    cmap = ListedColormap([CAT_COLORS[i] for i in range(0, 6)])      # 0=TSTM .. 5=HIGH
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    masked = np.ma.masked_where(category < 0, category)             # only no-risk hidden
    fig, ax = _base_ax()
    pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_xticklabels([CATEGORY_NAMES[i] for i in range(0, 6)])
    ax.set_title(_with_run(title, fxx))
    out_path = out_path or (PRODUCTS / f"cat_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_intensity(median_bin: np.ndarray, hazard: str, title: str,
                   out_path: str | Path | None = None, fxx: int | None = None) -> str:
    """Filled median conditional-intensity bin index (masked where < 0)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from ..config import INTENSITY_BINS

    spec = INTENSITY_BINS[hazard]
    labels = spec["labels"]
    n = len(labels)
    lat, lon = grid.figs_latlon()
    masked = np.ma.masked_where(median_bin < 0, median_bin)
    fig, ax = _base_ax()
    if spec["kind"] == "ef":  # tornado EF scale -> fixed discrete palette
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap(EF_COLORS[:n])
        norm = BoundaryNorm(np.arange(-0.5, n, 1.0), cmap.N)
        pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), shading="auto")
    else:
        pm = ax.pcolormesh(lon, lat, masked, cmap="plasma", vmin=0, vmax=n - 1,
                           transform=ccrs.PlateCarree(), shading="auto")
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=range(n))
    cbar.ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_title(_with_run(title, fxx))
    out_path = out_path or (PRODUCTS / f"intensity_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_pib(prob: np.ndarray, pib: np.ndarray, hazard: str, title: str,
             out_path: str | Path | None = None, fxx: int | None = None) -> str:
    """Probability / Peak-Intensity-Bin overlap map (the PIB analog of the
    probability+CIG plot): the hazard probability is the filled contour (standard SPC
    prob colors), and the **PIB** is overlaid as FILLED colored bands using the PIB
    color table (``config.PIB_COLORS``) — NO hatching. The PIB fill is drawn only
    inside the plotted probability area and at reduced opacity so the probability
    contours read through. ``pib`` is an (ny, nx) int field (0..6 == PIB1..PIB7; <0
    none, not drawn)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = grid.figs_latlon()
    prob_levels = SPC_PROB_LEVELS[hazard]
    fig, ax = _base_ax()
    # probability as line contours (so the PIB color fill below is the dominant fill)
    cs = ax.contour(lon, lat, prob, levels=prob_levels, colors="0.25", linewidths=0.7,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt=lambda v: f"{int(round(v*100))}%", fontsize=6, inline=True)
    # PIB filled bands, masked to the drawn probability area (>= lowest contour)
    n = len(PIB_LABELS)
    pib = np.where(np.asarray(prob) >= prob_levels[0], np.asarray(pib), -1)
    masked = np.ma.masked_where(pib < 0, pib)
    cmap = ListedColormap(list(PIB_COLORS[:n]))
    norm = BoundaryNorm(np.arange(-0.5, n, 1.0), cmap.N)
    pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm, alpha=0.85,
                       transform=ccrs.PlateCarree(), shading="auto", zorder=0.5)
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=range(n))
    cbar.ax.set_xticklabels(PIB_LABELS, fontsize=8)
    ax.set_title(_with_run(title, fxx))
    out_path = out_path or (PRODUCTS / f"pib_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
