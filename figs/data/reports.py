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

from ..config import REPORTS_CACHE

MPH_TO_KT = 1.0 / 1.15077945
_CATEGORY_MAP = {"tornado": "tor", "wind": "wind", "hail": "hail"}

# FIGS-normalized report columns.
REPORT_COLUMNS = ["time", "lat", "lon", "hazard", "magnitude", "ef", "source"]


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
        if hazard == "wind":
            # Drop unknown-magnitude wind reports (NaN after the sentinel patch).
            if r.magnitude is None or not np.isfinite(r.magnitude):
                continue
            mag = float(r.magnitude) * MPH_TO_KT
            ef = -1
        elif hazard == "hail":
            mag = float(r.magnitude)
            ef = -1
        else:  # tornado
            ef = int(r.magnitude) if r.magnitude is not None and r.magnitude >= 0 else -1
            mag = float(ef)
        t = r.time if r.time.tzinfo else r.time.replace(tzinfo=timezone.utc)
        rows.append((t, float(r.lat), float(r.lon), hazard, mag, ef, r.source))
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def _day_cache_path(day_utc: datetime) -> Path:
    return REPORTS_CACHE / f"reports_{day_utc:%Y%m%d}.csv"


def reports_for_day(day_utc: datetime, *, refresh: bool = False) -> pd.DataFrame:
    """Normalized FIGS reports for a single UTC calendar day (cached as CSV)."""
    if day_utc.tzinfo is None:
        day_utc = day_utc.replace(tzinfo=timezone.utc)
    day0 = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    path = _day_cache_path(day0)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["time"])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    rep = _reference_reports_module()
    raw = rep.fetch_reports(day0, day0 + timedelta(days=1))
    df = _normalize(raw)
    df.to_csv(path, index=False)
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
