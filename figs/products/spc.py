"""Fetch and plot **actual SPC outlook contours** from the Iowa Environmental
Mesonet (IEM) GIS archive, for side-by-side comparison with FIGS probabilities.

IEM serves the archived SPC convective outlooks as a shapefile (per UTC window),
with one polygon per (issuance, hazard, threshold). We pull the probabilistic
tornado/wind/hail contours for a convective day and draw them as filled geographic
contours on the same cartopy projection as the FIGS panels — replacing the old
raster-GIF screenshots with true vector contours (approach from Reference-Stormigami).
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import PRODUCTS, SPC_PROB_LEVELS

GIS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/spc_outlooks.py"
HAZARD_CATEGORY = {"tor": "TORNADO", "wind": "WIND", "hail": "HAIL"}


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def fetch_spc_outlooks(date: datetime, *, day: int = 1, cache_dir: str | Path | None = None):
    """Return a GeoDataFrame of SPC day-``day`` outlook polygons covering the UTC
    convective day of ``date`` (00Z that date through 12Z the next day, so the
    1300Z/1630Z/2000Z issuances are all included). Columns include CATEGORY,
    THRESHOLD, PRODISS/ISSUE/EXPIRE, geometry (EPSG:4269). Cached as GeoPackage."""
    import geopandas as gpd

    d0 = date.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    sts, ets = d0, d0 + timedelta(days=1, hours=12)
    cache_dir = Path(cache_dir) if cache_dir else (PRODUCTS / "spc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"spc_day{day}_{d0:%Y%m%d}.gpkg"
    if cache.exists():
        return gpd.read_file(cache)

    import requests

    r = requests.get(GIS_URL, params={"d": day, "type": "C", "sts": _ts(sts), "ets": _ts(ets)},
                     timeout=120)
    r.raise_for_status()
    with tempfile.TemporaryDirectory() as td:
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(td)
        shp = next(p for p in Path(td).iterdir() if p.suffix == ".shp")
        g = gpd.read_file(shp)
    g = g[g.geometry.notna()].copy()
    if len(g):
        g.to_file(cache, driver="GPKG")
    return g


def select_issuance(gdf, target: datetime):
    """Keep only the single outlook issuance whose ISSUE time is closest to
    ``target`` (e.g. the model run hour) — so we compare against the outlook SPC
    actually had out for that cycle, not a blend of all daily updates."""
    if gdf is None or len(gdf) == 0 or "ISSUE" not in gdf.columns:
        return gdf
    tgt = target.astimezone(timezone.utc).strftime("%Y%m%d%H%M")

    def _key(s):
        return abs(int(s) - int(tgt))

    issues = sorted(gdf["ISSUE"].dropna().unique(), key=_key)
    if not issues:
        return gdf
    return gdf[gdf["ISSUE"] == issues[0]].copy()


def plot_spc_outlook(ax, gdf, hazard: str, colors, *, sign_hatch: bool = True):
    """Draw the SPC probabilistic outlook for ``hazard`` (filled by THRESHOLD using
    the matching FIGS ``colors``) on a cartopy ``ax``. Higher probabilities draw on
    top; the significant-severe / conditional-intensity area is hatched.

    Significant-severe encoding differs by SPC system:
      * legacy (pre-2026-03-03): a single ``THRESHOLD == "SIGN"`` area → '//' hatch;
      * new system (2026+): ``THRESHOLD`` ∈ {"CIG1","CIG2","CIG3"} tiers → hatched
        with the FIGS ``CIG_HATCH`` patterns (dots / diagonal / crosshatch), so the
        SPC panel reads the same way as the FIGS CIG overlay.

    Returns the # of probability polygons drawn (0 → nothing issued for that hazard)."""
    import cartopy.crs as ccrs

    from .plots import CIG_HATCH

    if gdf is None or len(gdf) == 0:
        return 0
    sub = gdf[gdf["CATEGORY"] == HAZARD_CATEGORY[hazard]]
    levels = SPC_PROB_LEVELS[hazard]
    pc = ccrs.PlateCarree()
    drawn = 0
    for thr, color in zip(levels, colors):                 # low -> high (higher on top)
        layer = sub[sub["THRESHOLD"] == f"{thr:.2f}"]
        if len(layer):
            ax.add_geometries(layer.geometry, crs=pc, facecolor=color,
                              edgecolor="0.3", linewidth=0.4, alpha=0.85)
            drawn += len(layer)
    if sign_hatch:
        # new-system CIG tiers (FIGS hatch patterns) + legacy single SIGN area
        for label, hatch in (("CIG1", CIG_HATCH[1]), ("CIG2", CIG_HATCH[2]),
                             ("CIG3", CIG_HATCH[3]), ("SIGN", "//")):
            layer = sub[sub["THRESHOLD"] == label]
            if len(layer):
                ax.add_geometries(layer.geometry, crs=pc, facecolor="none",
                                  edgecolor="black", linewidth=0.6, hatch=hatch)
    return drawn
