"""Combined storm-report database for FIGS labels.

Thin wrapper over the battle-tested fetcher in ``Reference-ReportDB``
(``radar_warning_game.data.reports.fetch_reports``), which merges IEM LSRs +
SPC daily CSV + SVRGIS (post-survey EF / casualty backfill). We normalize the
returned reports into a FIGS DataFrame (hazard codes, knots for wind) and cache
per-UTC-day as CSV under ``Data/reports`` so repeated label builds are cheap.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NEIGHBORHOOD_RADIUS_MI, REPORTS_CACHE

MPH_TO_KT = 1.0 / 1.15077945
_CATEGORY_MAP = {"tornado": "tor", "wind": "wind", "hail": "hail"}

# DAT events are timed by association to the nearest same-day reliable event
# (SVRGIS track / LSR report) within this radius; unmatched DAT events are dropped.
DAT_ASSOC_MI = NEIGHBORHOOD_RADIUS_MI

# FIGS-normalized report columns. ``wind_mph`` carries the peak wind in MPH for the
# PIB (peak-intensity) scale: native mph for wind reports, the DAT damage-wind for
# DAT-sourced rows, NaN otherwise (tornado EF carries no mph — see labels/PIB).
REPORT_COLUMNS = ["time", "lat", "lon", "hazard", "magnitude", "ef", "source", "wind_mph"]


def _haversine_mi(lat1, lon1, lat2, lon2) -> float:
    import numpy as np
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


@lru_cache(maxsize=1)
def _reference_reports_module():
    """Import ``radar_warning_game.data.reports`` from the sibling reference
    project, adding it to sys.path on first use."""
    ref = Path(__file__).resolve().parents[2] / "Reference-ReportDB"
    if str(ref) not in sys.path:
        sys.path.insert(0, str(ref))
    from radar_warning_game.data import reports as _rep  # type: ignore

    # FIGS does NOT count damage-only / unknown-magnitude wind reports as a
    # default 60 mph (the reference's convention). Patch the sentinel to NaN so
    # such reports arrive with magnitude NaN and are dropped in _normalize,
    # without falsely discarding genuine measured/estimated 60 mph reports.
    _rep.UNKNOWN_WIND_DEFAULT_MPH = float("nan")
    return _rep


def _normalize(reports) -> pd.DataFrame:
    """Convert a list of reference ``Report`` objects to the FIGS DataFrame.

    magnitude is stored in the hazard's native intensity unit used for binning:
    knots for wind, inches for hail, EF integer for tornado. ``ef`` carries the
    tornado EF (or -1 unknown) separately for convenience.
    """
    rows = []
    for r in reports:
        hazard = _CATEGORY_MAP.get(r.category)
        if hazard is None:
            continue
        wind_mph = float("nan")
        if hazard == "wind":
            # Drop unknown-magnitude wind reports (NaN after the sentinel patch).
            if r.magnitude is None or not np.isfinite(r.magnitude):
                continue
            wind_mph = float(r.magnitude)          # native mph (pre-knots) → PIB
            mag = wind_mph * MPH_TO_KT
            ef = -1
        elif hazard == "hail":
            mag = float(r.magnitude)
            ef = -1
        else:  # tornado
            ef = int(r.magnitude) if r.magnitude is not None and r.magnitude >= 0 else -1
            mag = float(ef)
        t = r.time if r.time.tzinfo else r.time.replace(tzinfo=timezone.utc)
        rows.append((t, float(r.lat), float(r.lon), hazard, mag, ef, r.source, wind_mph))
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def _day_cache_path(day_utc: datetime) -> Path:
    return REPORTS_CACHE / f"reports_{day_utc:%Y%m%d}.csv"


# A day's reports keep arriving while it's in progress and finalize over the
# following hours; don't trust a cached CSV until this long AFTER the UTC day ends
# (so Day-1 / "today" validation always re-fetches the latest LSRs).
REPORTS_REFRESH_GRACE = timedelta(hours=12)


def reports_for_day(day_utc: datetime, *, refresh: bool = False) -> pd.DataFrame:
    """Normalized FIGS reports for a single UTC calendar day (cached as CSV).

    A cached CSV is trusted ONLY if it was WRITTEN after the day finalized
    (day end + ``REPORTS_REFRESH_GRACE``). This matters because a window/forecast
    can fetch a day while it's still in progress — or even before it starts (a
    forecast valid into a future day) — freezing a partial/empty report set; gating
    on the cache's mtime (not just ``now``) means such a premature cache is always
    re-fetched once the day is actually over, instead of being served as final.
    ``refresh`` forces a re-fetch regardless."""
    import os

    if day_utc.tzinfo is None:
        day_utc = day_utc.replace(tzinfo=timezone.utc)
    day0 = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    path = _day_cache_path(day0)
    final_after = day0 + timedelta(days=1) + REPORTS_REFRESH_GRACE   # day is "done" after this
    cache_is_final = False
    if path.exists():
        written = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        cache_is_final = written >= final_after        # written when the day was already complete
    if cache_is_final and not refresh:
        df = pd.read_csv(path, parse_dates=["time"])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        if "wind_mph" not in df.columns:          # back-compat for pre-PIB caches
            df["wind_mph"] = float("nan")
        # Pre-PIB caches lack wind_mph entirely → wind PIB would be empty. Recover it
        # for wind rows from the stored knots magnitude (kt → mph; exact, since
        # _normalize stored mph*MPH_TO_KT) so measured wind reports keep their PIB.
        if not df.empty:
            m = (df["hazard"] == "wind") & df["wind_mph"].isna()
            if m.any():
                df.loc[m, "wind_mph"] = df.loc[m, "magnitude"] / MPH_TO_KT
        return df
    # (re)fetch: the cache is missing, premature (written before the day finalized),
    # or a forced refresh. The reference fetcher has its OWN per-day raw-IEM cache and
    # only re-downloads when that file is missing — so a premature/partial copy would
    # defeat the re-fetch. Drop it (and the boundary day the window touches) to force
    # a fresh pull whenever we don't already hold a finalized cache.
    rep = _reference_reports_module()
    cache_path = getattr(rep, "_iem_lsr_cache_path", None)
    if cache_path is not None:
        for dd in (day0, day0 + timedelta(days=1)):
            try:
                p = cache_path(dd)
                if p.exists():
                    p.unlink()
            except Exception:  # noqa: BLE001 - best-effort cache invalidation
                pass
    raw = rep.fetch_reports(day0, day0 + timedelta(days=1))
    df = _normalize(raw)
    df.to_csv(path, index=False)            # always rewrite: keeps the freshest copy on disk
    return df


@lru_cache(maxsize=1)
def _svrgis_module():
    """Import the reference SVRGIS loader (adds the path on first use)."""
    ref = Path(__file__).resolve().parents[2] / "Reference-ReportDB"
    if str(ref) not in sys.path:
        sys.path.insert(0, str(ref))
    from radar_warning_game.data import spc_svrgis  # type: ignore

    return spc_svrgis


# SVRGIS tornado-track columns (start + end points define the damage path).
TRACK_COLUMNS = ["time", "slat", "slon", "elat", "elon", "ef"]


def svrgis_tracks_in_window(center: datetime, half_minutes: float) -> pd.DataFrame:
    """SVRGIS tornado **tracks** whose start time is within ``±half_minutes`` of
    ``center`` (UTC). Returns columns ``time, slat, slon, elat, elon, ef`` — the
    full start→end damage path and post-survey EF (``-1`` if unrated). Tornadoes
    with no recorded end point (``elat``/``elon`` == 0) get end := start.

    SVRGIS covers 1950–2023 (≈6-month lag), so tracks are available for the
    HRRRv4 training span; recent events fall back to point reports.
    """
    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)
    try:
        df = _svrgis_module().load_svrgis()
    except Exception:  # noqa: BLE001 - SVRGIS unavailable -> no tracks
        return pd.DataFrame(columns=TRACK_COLUMNS)
    lo = pd.Timestamp((center - timedelta(minutes=half_minutes)).replace(tzinfo=None))
    hi = pd.Timestamp((center + timedelta(minutes=half_minutes)).replace(tzinfo=None))
    band = df[(df["utc_dt"] >= lo) & (df["utc_dt"] <= hi)]
    if band.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    rows = []
    for _, r in band.iterrows():
        slat, slon = float(r["slat"]), float(r["slon"])
        elat, elon = float(r.get("elat", 0) or 0), float(r.get("elon", 0) or 0)
        if elat == 0.0 or elon == 0.0:  # no recorded end -> treat as point
            elat, elon = slat, slon
        mag = r.get("mag", -9)
        ef = int(mag) if (mag is not None and 0 <= mag <= 5) else -1
        t = r["utc_dt"].to_pydatetime().replace(tzinfo=timezone.utc)
        rows.append((t, slat, slon, elat, elon, ef))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _utc_days_in_window(center: datetime, half_minutes: float) -> list[datetime]:
    lo = center - timedelta(minutes=half_minutes)
    hi = center + timedelta(minutes=half_minutes)
    days = sorted({lo.date(), hi.date()})
    return [datetime(d.year, d.month, d.day, tzinfo=timezone.utc) for d in days]


def _tornado_time_anchors(day0: datetime) -> list[tuple[datetime, float, float]]:
    """Reliable (time, lat, lon) anchors for a UTC day from SVRGIS track starts +
    LSR tornado reports — used to give DAT tornado tracks a trustworthy time."""
    anchors: list[tuple[datetime, float, float]] = []
    try:
        df = _svrgis_module().load_svrgis()
        d0 = pd.Timestamp(day0.replace(tzinfo=None))
        band = df[(df["utc_dt"] >= d0) & (df["utc_dt"] < d0 + pd.Timedelta(days=1))]
        for _, r in band.iterrows():
            anchors.append((r["utc_dt"].to_pydatetime().replace(tzinfo=timezone.utc),
                            float(r["slat"]), float(r["slon"])))
    except Exception:  # noqa: BLE001
        pass
    rep = reports_for_day(day0)
    tor = rep[rep["hazard"] == "tor"] if not rep.empty else rep
    for _, r in tor.iterrows():
        anchors.append((r["time"].to_pydatetime(), float(r["lat"]), float(r["lon"])))
    return anchors


def _nearest_anchor_time(lat, lon, anchors, radius_mi):
    """Time of the nearest anchor within ``radius_mi`` of (lat, lon), else None."""
    best_t, best_d = None, radius_mi
    for (t, alat, alon) in anchors:
        d = _haversine_mi(lat, lon, alat, alon)
        if d <= best_d:
            best_t, best_d = t, d
    return best_t


# DAT tornado track columns: full damage PATH + DAT max wind (mph) + EF + associated time.
DAT_TRACK_COLUMNS = ["time", "path", "maxwind_mph", "ef"]


def dat_tornado_tracks_in_window(center: datetime, half_minutes: float) -> pd.DataFrame:
    """DAT tornado damage **tracks** (full polyline path + ``maxwind`` mph) whose
    ASSOCIATED time (nearest same-day SVRGIS/LSR tornado anchor within 25 mi) falls
    within ``±half_minutes`` of ``center``. DAT times are unreliable, so the time is
    borrowed from the matched reliable event; unmatched tracks are dropped.

    ``path`` is a list of ``[lat, lon]`` vertices (stamped along its length, like the
    SVRGIS tracks). ``maxwind_mph`` is ``-1`` if DAT gave no speed (excluded from PIB)."""
    from . import dat

    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)
    lo, hi = center - timedelta(minutes=half_minutes), center + timedelta(minutes=half_minutes)
    rows = []
    for day0 in _utc_days_in_window(center, half_minutes):
        tracks = dat.fetch_dat_day(day0).get("tracks", [])
        if not tracks:
            continue
        anchors = _tornado_time_anchors(day0)
        if not anchors:
            continue
        for tr in tracks:
            path = tr.get("path") or []
            if not path:
                continue
            slat, slon = path[0]
            t = _nearest_anchor_time(slat, slon, anchors, DAT_ASSOC_MI)
            if t is None or not (lo <= t <= hi):
                continue
            rows.append((t, path, float(tr.get("maxwind", -1.0)), int(tr.get("ef", -1))))
    return pd.DataFrame(rows, columns=DAT_TRACK_COLUMNS)


def dat_wind_reports_in_window(center: datetime, half_minutes: float) -> pd.DataFrame:
    """DAT straight-line **wind** damage points (with ``windspeed`` mph) timed by the
    nearest same-day wind LSR within 25 mi, returned as FIGS report rows (so they
    backfill wind presence + PIB magnitude). Unmatched points are dropped."""
    from . import dat

    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)
    lo, hi = center - timedelta(minutes=half_minutes), center + timedelta(minutes=half_minutes)
    rows = []
    for day0 in _utc_days_in_window(center, half_minutes):
        pts = [p for p in dat.fetch_dat_day(day0).get("points", []) if p["hazard"] == "wind"]
        if not pts:
            continue
        rep = reports_for_day(day0)
        wind = rep[rep["hazard"] == "wind"] if not rep.empty else rep
        anchors = [(r["time"].to_pydatetime(), float(r["lat"]), float(r["lon"]))
                   for _, r in wind.iterrows()]
        if not anchors:
            continue
        for p in pts:
            if p["mph"] <= 0:                      # no DAT speed → no PIB value
                continue
            t = _nearest_anchor_time(p["lat"], p["lon"], anchors, DAT_ASSOC_MI)
            if t is None or not (lo <= t <= hi):
                continue
            rows.append((t, p["lat"], p["lon"], "wind", p["mph"] * MPH_TO_KT, -1, "DAT", p["mph"]))
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def reports_in_window(center: datetime, half_minutes: float) -> pd.DataFrame:
    """All reports within ``±half_minutes`` of ``center`` (UTC), spanning a day
    boundary if needed."""
    if center.tzinfo is None:
        center = center.replace(tzinfo=timezone.utc)
    lo = center - timedelta(minutes=half_minutes)
    hi = center + timedelta(minutes=half_minutes)
    days = {lo.date(), hi.date()}
    frames = [reports_for_day(datetime(d.year, d.month, d.day, tzinfo=timezone.utc)) for d in days]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REPORT_COLUMNS)
    if df.empty:
        return df
    mask = (df["time"] >= lo) & (df["time"] <= hi)
    return df.loc[mask].reset_index(drop=True)
