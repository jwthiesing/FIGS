"""SPC-style plotting: filled probability contours, categorical-risk fills, and
significant-severe hatching on a CONUS Lambert map.

Mirrors the SPC convective-outlook look. Uses the FIGS grid lat/lon and cartopy;
all functions render to a PNG and return the path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import CATEGORY_NAMES, PRODUCTS, SPC_PROB_LEVELS
from ..data import grid

# Official SPC probabilistic-outlook fill colors, per hazard, aligned to the
# levels in config.SPC_PROB_LEVELS.
#   tornado : 2/5/10/15/30/45/60 %   -> green/brown/gold/red/magenta/purple/blue
#   wind    : 5/15/30/45/60/75/90 %  -> brown/gold/red/magenta/purple/blue/indigo
#   hail    : 5/15/30/45/60 %        -> brown/gold/red/magenta/purple
PROB_COLORS = {
    "tor": ["#008b00", "#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee", "#104e8b"],
    "wind": ["#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee", "#104e8b", "#4b0082"],
    "hail": ["#8b4726", "#ffc800", "#ff0000", "#ff00ff", "#912cee"],
}

# Official SPC categorical-risk fill colors for levels 1..5 (MRGL/SLGT/ENH/MDT/HIGH).
CAT_COLORS = {1: "#66a366", 2: "#ffe066", 3: "#ffa366", 4: "#e06666", 5: "#ee99ee"}

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


def set_extent(extent) -> None:
    """Set the plan-view extent used by every plot here (lon_min, lon_max,
    lat_min, lat_max). Pass ``None`` to let cartopy auto-fit the data."""
    global MAP_EXTENT
    MAP_EXTENT = list(extent) if extent is not None else None


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
                     sig: np.ndarray | None = None, cig: np.ndarray | None = None) -> str:
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
    ax.set_title(title)
    out_path = out_path or (PRODUCTS / f"prob_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_categorical(category: np.ndarray, hazard: str, title: str,
                     out_path: str | Path | None = None) -> str:
    """Filled SPC categorical risk (integer levels 0..5; 0 not drawn)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = grid.figs_latlon()
    cmap = ListedColormap([CAT_COLORS[i] for i in range(1, 6)])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    masked = np.ma.masked_where(category < 1, category)
    fig, ax = _base_ax()
    pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_xticklabels([CATEGORY_NAMES[i] for i in range(1, 6)])
    ax.set_title(title)
    out_path = out_path or (PRODUCTS / f"cat_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_intensity(median_bin: np.ndarray, hazard: str, title: str,
                   out_path: str | Path | None = None) -> str:
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
    ax.set_title(title)
    out_path = out_path or (PRODUCTS / f"intensity_{hazard}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
