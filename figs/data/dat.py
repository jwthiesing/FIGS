"""NWS Damage Assessment Toolkit (DAT) client — per-UTC-day fetch + cache.

DAT post-event survey data on the ArcGIS FeatureServer:
  * **Layer 0 "Damage Points"** — point damage indicators with an estimated
    ``windspeed`` (mph) and ``efscale`` per point (tornado EF0-5, or TSTM/Wind
    straight-line). Used to backfill wind / tornado damage-wind magnitudes.
  * **Layer 1 "Damage Lines"** — tornado damage **tracks** (polylines) with
    ``maxwind`` (mph) and ``efnum``. Used for full tornado-PATH integration (the
    path is stamped like the SVRGIS tracks), with ``maxwind`` driving tornado PIB.

We fetch by **UTC day** (DAT timestamps are often inaccurate, so a narrow time
window is unreliable — callers associate a DAT event with the nearest same-day
SVRGIS track / LSR report to borrow a trustworthy time; see ``data/reports.py``).

IMPORTANT: there is intentionally **no EF→mph fallback**. If ``windspeed`` /
``maxwind`` is blank, the speed is unknown (``-1.0``) and the event is excluded
from the PIB (peak-intensity) magnitude — we never synthesize a speed from EF.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import REPORTS_CACHE

DAT_BASE = ("https://services.dat.noaa.gov/arcgis/rest/services/"
            "nws_damageassessmenttoolkit/DamageViewer/FeatureServer")
_POINTS_URL = f"{DAT_BASE}/0/query"
_LINES_URL = f"{DAT_BASE}/1/query"
_PAGE = 2000

_DAT_CACHE = REPORTS_CACHE / "dat"

# DAT publishes with a survey lag and keeps editing recent events; don't trust a
# cached day until this long after it ends (mirrors reports.REPORTS_REFRESH_GRACE).
_REFRESH_GRACE = timedelta(days=2)


def _parse_ef(efscale) -> int:
    if not efscale:
        return -1
    m = re.search(r"EF\s*([0-5])", str(efscale), re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _hazard_from_ef(efscale, ef: int) -> str:
    """EF0-5 → tornado; TSTM / Wind survey → wind; else '' (skipped)."""
    if ef >= 0:
        return "tor"
    s = str(efscale or "").upper()
    if "TSTM" in s or "WIND" in s:
        return "wind"
    return ""


def _speed_mph(val) -> float:
    """Parse a DAT speed (string '110 mph' or numeric) → mph; -1.0 if blank/unknown.
    No EF fallback — a blank speed stays unknown."""
    if val is None:
        return -1.0
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    return float(m.group(1)) if m else -1.0


def _ms_day_bounds(day0: datetime) -> tuple[str, str]:
    nxt = day0 + timedelta(days=1)
    return f"DATE '{day0:%Y-%m-%d}'", f"DATE '{nxt:%Y-%m-%d}'"


def _query_all(url: str, where: str, *, geometry: bool, out_fields: str, timeout: float):
    import requests

    params = {"where": where, "outFields": out_fields,
              "returnGeometry": "true" if geometry else "false",
              "outSR": "4326", "f": "json", "resultRecordCount": _PAGE,
              "returnExceededLimitFeatures": "true"}
    feats, offset = [], 0
    while True:
        params["resultOffset"] = offset
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(f"DAT query error: {payload['error']}")
        batch = payload.get("features", [])
        feats.extend(batch)
        if len(batch) < _PAGE or not payload.get("exceededTransferLimit", False):
            return feats
        offset += _PAGE


def _fetch_day_raw(day0: datetime, timeout: float) -> dict:
    """Raw DAT features for one UTC day: {'points': [...], 'tracks': [...]}."""
    lo, hi = _ms_day_bounds(day0)
    where = f"stormdate >= {lo} AND stormdate < {hi}"
    pts, trks = [], []
    try:
        pt_feats = _query_all(_POINTS_URL, where, geometry=True,
                              out_fields="stormdate,windspeed,efscale,dod", timeout=timeout)
        for ft in pt_feats:
            a = ft.get("attributes", {}) or {}
            g = ft.get("geometry") or {}
            if g.get("x") is None or g.get("y") is None:
                continue
            ef = _parse_ef(a.get("efscale"))
            hz = _hazard_from_ef(a.get("efscale"), ef)
            if not hz:
                continue
            pts.append({"lat": float(g["y"]), "lon": float(g["x"]),
                        "mph": _speed_mph(a.get("windspeed")), "ef": ef, "hazard": hz})
    except Exception as e:  # noqa: BLE001
        print(f"[warn] DAT points fetch failed {day0:%Y-%m-%d} ({str(e)[:70]})", file=sys.stderr)
    try:
        ln_feats = _query_all(_LINES_URL, where, geometry=True,
                              out_fields="stormdate,efscale,efnum,maxwind", timeout=timeout)
        for ft in ln_feats:
            a = ft.get("attributes", {}) or {}
            paths = (ft.get("geometry") or {}).get("paths") or []
            verts = [pt for path in paths for pt in path]   # flatten multi-part
            if len(verts) < 1:
                continue
            mw = a.get("maxwind")
            maxwind = float(mw) if mw not in (None, "") and float(mw) > 0 else -1.0
            ef = a.get("efnum")
            ef = int(ef) if ef not in (None, "") and int(ef) >= 0 else _parse_ef(a.get("efscale"))
            # path stored as [[lat, lon], ...] (verts come as [lon, lat])
            trks.append({"path": [[float(v[1]), float(v[0])] for v in verts],
                         "maxwind": maxwind, "ef": ef})
    except Exception as e:  # noqa: BLE001
        print(f"[warn] DAT lines fetch failed {day0:%Y-%m-%d} ({str(e)[:70]})", file=sys.stderr)
    return {"points": pts, "tracks": trks}


def fetch_dat_day(day_utc: datetime, *, refresh: bool = False, timeout: float = 90.0) -> dict:
    """DAT points + tornado tracks for one UTC calendar day, cached as JSON.

    Returns ``{'points': [{lat, lon, mph, ef, hazard}, ...],
               'tracks': [{path: [[lat,lon],...], maxwind, ef}, ...]}``.
    ``mph`` / ``maxwind`` are ``-1.0`` when DAT gives no speed (no EF fallback)."""
    if day_utc.tzinfo is None:
        day_utc = day_utc.replace(tzinfo=timezone.utc)
    day0 = day_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    _DAT_CACHE.mkdir(parents=True, exist_ok=True)
    path = _DAT_CACHE / f"dat_{day0:%Y%m%d}.json"
    still_updating = datetime.now(timezone.utc) < day0 + timedelta(days=1) + _REFRESH_GRACE
    if path.exists() and not refresh and not still_updating:
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - corrupt cache → refetch
            pass
    data = _fetch_day_raw(day0, timeout)
    try:
        path.write_text(json.dumps(data))
    except Exception:  # noqa: BLE001
        pass
    return data
