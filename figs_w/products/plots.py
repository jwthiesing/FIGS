"""SPC fire-weather-style plotting for FIGS-W.

Reuses the FIGS cartopy base axis / extent control (``figs.products.plots``) and
grid; only the color tables + level/label sourcing are wildfire-specific. The
categorical fill uses the SPC fire-weather palette (ELEVATED tan, CRITICAL red,
EXTREME magenta); NONE / below-threshold is not drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from figs.data import grid
from figs.products import plots as P   # reuse _base_ax, set_extent, overlay_cig, MAP_EXTENT

from .. import config as C

set_extent = P.set_extent

# Probability fill ramp: tan → orange → red → magenta → purple → deep purple (7 levels).
PROB_COLORS = {"wildfire": ["#ffe7b3", "#ffcc99", "#ff9966", "#ff5a5a", "#ff00ff",
                             "#9900cc", "#4b0082"]}

# Categorical fire-weather colors: 1 ELEVATED (tan), 2 CRITICAL (red), 3 EXTREME (magenta).
# 0 NONE and below-threshold (-1) are not drawn (SPC fire outlooks fill only 1–3).
CAT_COLORS = {1: "#f6c87a", 2: "#ff5a5a", 3: "#ff33ff"}

# size-bin (intensity) palette for 7 bins (pale yellow → dark red), small → very large fire.
SIZE_COLORS = ["#ffffb2", "#fee391", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026", "#67000d"]


def plot_probability(prob, title, out_path=None, cig=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    lat, lon = grid.figs_latlon()
    levels = list(C.SPC_PROB_LEVELS["wildfire"]) + [1.0]
    fig, ax = P._base_ax()
    cf = ax.contourf(lon, lat, prob, levels=levels, colors=PROB_COLORS["wildfire"],
                     transform=ccrs.PlateCarree(), extend="neither")
    if cig is not None:
        P.overlay_cig(ax, lon, lat, np.where(prob >= C.SPC_PROB_LEVELS["wildfire"][0], cig, 0))
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=C.SPC_PROB_LEVELS["wildfire"])
    cbar.ax.set_xticklabels([f"{int(p*100)}%" for p in C.SPC_PROB_LEVELS["wildfire"]])
    ax.set_title(title)
    out_path = out_path or (C.PRODUCTS / "fire_prob.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return str(out_path)


def plot_categorical(category, title, out_path=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = grid.figs_latlon()
    cmap = ListedColormap([CAT_COLORS[i] for i in (1, 2, 3)])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    masked = np.ma.masked_where(np.asarray(category) < 1, category)   # NONE / no-risk hidden
    fig, ax = P._base_ax()
    pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8,
                        ticks=[1, 2, 3])
    cbar.ax.set_xticklabels([C.CATEGORY_NAMES[i] for i in (1, 2, 3)])
    ax.set_title(title)
    out_path = out_path or (C.PRODUCTS / "fire_categorical.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return str(out_path)


def plot_intensity(median_bin, title, out_path=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    labels = C.INTENSITY_BINS["wildfire"]["labels"]
    n = len(labels)
    lat, lon = grid.figs_latlon()
    masked = np.ma.masked_where(np.asarray(median_bin) < 0, median_bin)
    cmap = ListedColormap(SIZE_COLORS[:n])
    norm = BoundaryNorm(np.arange(-0.5, n, 1.0), cmap.N)
    fig, ax = P._base_ax()
    pm = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8, ticks=range(n))
    cbar.ax.set_xticklabels([f"{l} ac" for l in labels], rotation=30, fontsize=8)
    ax.set_title(title)
    out_path = out_path or (C.PRODUCTS / "fire_size.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return str(out_path)
