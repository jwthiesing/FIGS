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


def _convective_day_start(t: datetime) -> datetime:
    """12Z that begins the SPC convective day (12Z–12Z) containing ``t``."""
    t = t.astimezone(timezone.utc)
    base = t if t.hour >= 12 else t - timedelta(days=1)
    return base.replace(hour=12, minute=0, second=0, microsecond=0)


def outlook_day_for(run: datetime, valid_time: datetime) -> int:
    """SPC outlook day number for a forecast valid at ``valid_time`` from a cycle
    issued at ``run`` — 1 = SPC's Day 1, 2 = Day 2, etc. Clamped to the 1–3
    probabilistic archive range.

    SPC labels as "Day 1" the convective day (12Z–12Z) beginning at **12Z on the
    run's calendar date**, for both morning and afternoon cycles (e.g. a 06Z cycle's
    Day 1 is the upcoming 12Z day, not the one already ending). The valid time is
    placed in its own 12Z–12Z convective day; the difference in days + 1 is the
    SPC day number."""
    r = run.astimezone(timezone.utc)
    day1_start = r.replace(hour=12, minute=0, second=0, microsecond=0)
    vd = _convective_day_start(valid_time)
    n = round((vd - day1_start).total_seconds() / 86400) + 1
    return max(1, min(3, n))


def fetch_spc_outlooks(date: datetime, *, day: int = 1, cache_dir: str | Path | None = None):
    """Return a GeoDataFrame of SPC day-``day`` outlook polygons VALID for the
    convective day (12Z–12Z) containing ``date``. Day-N outlooks are issued ~(N-1)
    days before they're valid, so the request window reaches back that far and
    ``select_issuance`` then narrows to the issuance closest to the model run.
    Columns include CATEGORY, THRESHOLD, PRODISS/ISSUE/EXPIRE, geometry
    (EPSG:4269). Cached as GeoPackage keyed by valid convective day + day."""
    import geopandas as gpd

    cstart = _convective_day_start(date)                # 12Z start of the valid conv day
    sts = cstart - timedelta(hours=12, days=day - 1)    # reach back to the Day-N issuances
    ets = cstart + timedelta(hours=24)                  # end of the valid convective day
    # If the convective day hasn't ended yet, SPC is still issuing updates for it, so a
    # cached copy may be stale (e.g. fetched before today's Day 1 was out) — bypass it.
    in_progress = ets > datetime.now(timezone.utc)
    cache_dir = Path(cache_dir) if cache_dir else (PRODUCTS / "spc_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"spc_day{day}_{cstart:%Y%m%d}.gpkg"
    if cache.exists() and not in_progress:
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
    if len(g) and not in_progress:      # don't persist a still-updating day's partial set
        g.to_file(cache, driver="GPKG")
    return g


def select_issuance(gdf, target: datetime, valid_time: datetime | None = None):
    """Keep only the single outlook issuance closest to ``target`` (the model run).

    When ``valid_time`` is given, FIRST restrict to the outlooks actually valid for
    that forecast's convective day (identified by ``EXPIRE`` = 12Z at the day's
    end). Without this, an early-morning run snaps to the *previous* day's tail
    update (the 01Z/06Z outlook) merely because it is closest in time — that
    outlook is for the prior convective day, not the one being forecast."""
    if gdf is None or len(gdf) == 0 or "ISSUE" not in gdf.columns:
        return gdf

    g = gdf
    if valid_time is not None and "EXPIRE" in gdf.columns:
        import pandas as pd

        cend = _convective_day_start(valid_time) + timedelta(hours=24)  # 12Z conv-day end
        exp = pd.to_datetime(gdf["EXPIRE"].astype(str), format="%Y%m%d%H%M",
                             errors="coerce", utc=True)
        if exp.notna().any():                   # EXPIRE parsed -> trust the valid-day filter
            # Keep ONLY the outlooks valid for the forecast's convective day. If that
            # leaves nothing (e.g. SPC hasn't issued today's Day 1 yet), return empty —
            # a blank panel is correct, where snapping to the previous day's outlook is not.
            g = gdf[exp.dt.strftime("%Y%m%d") == cend.strftime("%Y%m%d")]

    tgt = target.astimezone(timezone.utc).strftime("%Y%m%d%H%M")

    def _key(s):
        return abs(int(s) - int(tgt))

    issues = sorted(g["ISSUE"].dropna().unique(), key=_key)
    if not issues:
        return g
    return g[g["ISSUE"] == issues[0]].copy()


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
            # opaque (alpha=1) to match the FIGS panel's contourf fill, and zorder=1 so
            # the state borders (cartopy STATES default zorder 1.5) draw OVER the fill —
            # add_geometries defaults to 1.5 (== STATES) and would otherwise hide them,
            # while the FIGS contourf is zorder 1 and lets the borders bleed through.
            ax.add_geometries(layer.geometry, crs=pc, facecolor=color,
                              edgecolor="0.3", linewidth=0.4, alpha=1.0, zorder=1)
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
