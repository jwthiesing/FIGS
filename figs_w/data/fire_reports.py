"""Combined wildfire catalog for FIGS-W labels — NIFC (authoritative) + IEM (fill).

A wildfire is an **interval**, not a point: large fires burn across many HRRR
cycles, so each fire is represented as an ACTIVE WINDOW ``[start, end]`` with a
time-ordered set of footprint sample points (from the **fire-progression**
perimeters), plus an authoritative **final size** / fatalities / structures.

Sources, merged authoritatively:
  1. **NIFC incident locations** — IRWIN id, discovery + containment time, point,
     fatalities, structures, (preliminary) size.
  2. **NIFC perimeters / NIFS archive** — final acres (joined to the incident by
     IRWIN id; authoritative size, overrides the preliminary incident size).
  3. **NIFC fire progression** — time-stamped perimeters → the fire's footprint
     over time (so a multi-day fire is stamped at every valid hour it's active,
     using its perimeter as of that hour).
  4. **IEM wildfire LSRs** — near-real-time fill; **deduplicated** against NIFC by
     space + time and only kept when NIFC has no matching fire (then occurrence-only,
     unknown size).

NOTE — the ArcGIS service URLs + attribute field names below are configurable and
should be confirmed against the linked NIFC datasets. The query/merge logic is
source-agnostic given those.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from figs.data.grid import MI_TO_KM

from .. import config as C

# --- NIFC ArcGIS feature services (CONFIRM against the linked datasets) -------- #
NIFC_INCIDENT_SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations/FeatureServer/0/query")
NIFC_PERIMETER_SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query")
NIFC_PROGRESSION_SERVICE = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Fire_Progression/FeatureServer/0/query")            # time-stamped perimeters
IEM_LSR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py"

INCIDENT_FIELDS = dict(
    irwin="IrwinID", time="FireDiscoveryDateTime", contain="ContainmentDateTime",
    out="FireOutDateTime", size="IncidentSize", fatalities="Fatalities",
    structures="StructuresDestroyed", lat="InitialLatitude", lon="InitialLongitude")
PERIMETER_FIELDS = dict(irwin="poly_IRWINID", acres="poly_GISAcres")
PROGRESSION_FIELDS = dict(irwin="IRWINID", time="CreateDate")    # + geometry

# IEM↔NIFC dedup tolerances and the default IEM-only / no-containment durations.
_DEDUP_MI = 10.0
_DEDUP_HR = 24.0
_IEM_ACTIVE_HR = 3.0            # IEM-only fire treated active ± this around the LSR
_DEFAULT_DURATION_HR = 24.0    # fallback active span when containment is unknown
_MAX_FOOTPRINT_PTS = 24        # perimeter vertices sampled per progression step


@dataclass
class FireRecord:
    irwin: str
    start: datetime
    end: datetime
    final_size: float                       # acres; NaN if unknown (IEM-only)
    deadly: bool
    # footprint over time: list of (time, points) where points is (M,2) lat/lon.
    footprint: list = field(default_factory=list)
    static_pt: tuple | None = None          # (lat, lon) fallback when no progression


# --------------------------------------------------------------------------- #
# ArcGIS / geometry helpers
# --------------------------------------------------------------------------- #
def _arcgis_query(url: str, where: str, out_fields: str, *, geometry: bool = False,
                  page: int = 2000) -> list[dict]:
    import requests

    feats, offset = [], 0
    while True:
        params = {"where": where, "outFields": out_fields, "f": "geojson",
                  "returnGeometry": str(geometry).lower(), "outSR": 4326,
                  "resultOffset": offset, "resultRecordCount": page}
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        batch = r.json().get("features", [])
        feats.extend(batch)
        if len(batch) < page:
            return feats
        offset += page


def _epoch(dtval) -> datetime | None:
    if dtval is None:
        return None
    if isinstance(dtval, (int, float)):
        return datetime.fromtimestamp(dtval / 1000, tz=timezone.utc)
    try:
        return pd.to_datetime(dtval, utc=True).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _sample_polygon(geom: dict) -> np.ndarray:
    """Representative (M,2) lat/lon points for a GeoJSON Polygon/MultiPolygon:
    the centroid + a downsampled set of exterior-ring vertices (so a big fire's
    whole perimeter — not just its center — seeds the 25 mi neighborhood)."""
    if not geom:
        return np.empty((0, 2))
    rings = []
    if geom.get("type") == "Polygon":
        rings = geom.get("coordinates", [])[:1]
    elif geom.get("type") == "MultiPolygon":
        rings = [poly[0] for poly in geom.get("coordinates", []) if poly]
    pts = []
    for ring in rings:
        arr = np.asarray(ring, dtype=float)        # (N,2) lon,lat
        if arr.size == 0:
            continue
        step = max(1, len(arr) // _MAX_FOOTPRINT_PTS)
        verts = arr[::step]
        pts.append(verts)
        pts.append(arr.mean(axis=0, keepdims=True))   # centroid
    if not pts:
        return np.empty((0, 2))
    lonlat = np.vstack(pts)
    return np.column_stack([lonlat[:, 1], lonlat[:, 0]])   # -> (M,2) lat,lon


# --------------------------------------------------------------------------- #
# Catalog assembly (cached per month)
# --------------------------------------------------------------------------- #
def _fetch_incidents(start: datetime, end: datetime) -> pd.DataFrame:
    f = INCIDENT_FIELDS
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    where = f"{f['time']} <= {e_ms} AND ({f['contain']} >= {s_ms} OR {f['contain']} IS NULL)"
    try:
        feats = _arcgis_query(NIFC_INCIDENT_SERVICE, where, ",".join(f.values()))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] NIFC incidents fetch failed ({str(e)[:90]})", file=sys.stderr)
        return pd.DataFrame()
    rows = []
    for ft in feats:
        a = ft.get("properties", {}) or {}
        geom = (ft.get("geometry") or {}).get("coordinates") or [None, None]
        lon, lat = (list(geom) + [None, None])[:2]
        rows.append(dict(
            irwin=a.get(f["irwin"]), start=_epoch(a.get(f["time"])),
            contain=_epoch(a.get(f["contain"])) or _epoch(a.get(f["out"])),
            size=float(a.get(f["size"]) or np.nan),
            fatalities=int(a.get(f["fatalities"]) or 0),
            structures=int(a.get(f["structures"]) or 0),
            lat=a.get(f["lat"], lat), lon=a.get(f["lon"], lon)))
    df = pd.DataFrame(rows)
    return df[df["irwin"].notna() & df["start"].notna()] if len(df) else df


def _fetch_final_acres(irwins: list[str]) -> dict[str, float]:
    """Authoritative final acres per IRWIN id from the perimeter/NIFS layer."""
    f = PERIMETER_FIELDS
    if not irwins:
        return {}
    out: dict[str, float] = {}
    for i in range(0, len(irwins), 500):                 # chunk the IN() clause
        chunk = irwins[i:i + 500]
        where = f["irwin"] + " IN (" + ",".join(f"'{x}'" for x in chunk) + ")"
        try:
            for ft in _arcgis_query(NIFC_PERIMETER_SERVICE, where, ",".join(f.values())):
                a = ft.get("properties", {}) or {}
                irw, ac = a.get(f["irwin"]), a.get(f["acres"])
                if irw is not None and ac is not None:
                    out[irw] = max(out.get(irw, 0.0), float(ac))   # largest perimeter = final
        except Exception as e:  # noqa: BLE001
            print(f"[warn] NIFC perimeters fetch failed ({str(e)[:90]})", file=sys.stderr)
            break
    return out


def _fetch_progression(start: datetime, end: datetime) -> dict[str, list]:
    """Time-stamped footprint samples per IRWIN id from the fire-progression layer:
    {irwin: [(time, points(M,2 lat/lon)), ...]} sorted by time."""
    f = PROGRESSION_FIELDS
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    where = f"{f['time']} >= {s_ms} AND {f['time']} <= {e_ms}"
    prog: dict[str, list] = {}
    try:
        feats = _arcgis_query(NIFC_PROGRESSION_SERVICE, where, ",".join(f.values()),
                              geometry=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] NIFC progression fetch failed ({str(e)[:90]})", file=sys.stderr)
        return prog
    for ft in feats:
        a = ft.get("properties", {}) or {}
        irw, t = a.get(f["irwin"]), _epoch(a.get(f["time"]))
        pts = _sample_polygon(ft.get("geometry") or {})
        if irw is None or t is None or len(pts) == 0:
            continue
        prog.setdefault(irw, []).append((t, pts))
    for irw in prog:
        prog[irw].sort(key=lambda tp: tp[0])
    return prog


def _build_catalog(start: datetime, end: datetime) -> list[FireRecord]:
    """All fires ACTIVE at any point in [start, end] as merged FireRecords."""
    inc = _fetch_incidents(start, end)
    if inc.empty:
        return _iem_only_records(start, end, nifc=[])
    final_ac = _fetch_final_acres([str(x) for x in inc["irwin"].tolist()])
    prog = _fetch_progression(start - timedelta(days=2), end + timedelta(days=2))

    recs: list[FireRecord] = []
    for _, r in inc.iterrows():
        irw = str(r["irwin"])
        endt = r["contain"] or (r["start"] + timedelta(hours=_DEFAULT_DURATION_HR))
        size = final_ac.get(irw, r["size"])              # perimeter overrides incident size
        deadly = (int(r["fatalities"]) > 0
                  or int(r["structures"]) >= C.DEADLY_STRUCTURES_THRESHOLD)
        fp = prog.get(irw, [])
        static = (float(r["lat"]), float(r["lon"])) if pd.notna(r["lat"]) and pd.notna(r["lon"]) else None
        recs.append(FireRecord(irwin=irw, start=r["start"], end=endt,
                               final_size=float(size) if pd.notna(size) else np.nan,
                               deadly=bool(deadly), footprint=fp, static_pt=static))
    recs += _iem_only_records(start, end, nifc=recs)
    return recs


def _iem_only_records(start, end, nifc: list[FireRecord]) -> list[FireRecord]:
    """IEM wildfire LSRs not already covered by a NIFC fire (space+time dedup) →
    occurrence-only records (unknown size)."""
    df = _fetch_iem_fire(start, end)
    if df.empty:
        return []
    # NIFC anchor points (start point of each fire) for dedup
    anchors = [(rec.start, rec.static_pt) for rec in nifc if rec.static_pt]
    out = []
    for _, r in df.iterrows():
        t, la, lo = r["time"].to_pydatetime(), float(r["lat"]), float(r["lon"])
        dup = False
        for (st, (alat, alon)) in anchors:
            if abs((t - st).total_seconds()) <= _DEDUP_HR * 3600 and \
               _haversine_mi(la, lo, alat, alon) <= _DEDUP_MI:
                dup = True
                break
        if dup:
            continue
        out.append(FireRecord(irwin=f"IEM-{t:%Y%m%d%H%M}-{la:.2f}-{lo:.2f}",
                              start=t - timedelta(hours=_IEM_ACTIVE_HR),
                              end=t + timedelta(hours=_IEM_ACTIVE_HR),
                              final_size=np.nan, deadly=False, footprint=[],
                              static_pt=(la, lo)))
    return out


def _fetch_iem_fire(start: datetime, end: datetime) -> pd.DataFrame:
    import io

    import requests
    params = {"sts": start.strftime("%Y-%m-%dT%H:%MZ"), "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
              "fmt": "csv", "type": "WILDFIRE"}
    try:
        r = requests.get(IEM_LSR_URL, params=params, timeout=60); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] IEM fire LSR fetch failed ({str(e)[:90]})", file=sys.stderr)
        return pd.DataFrame()
    if df.empty or "VALID" not in df:
        return pd.DataFrame()
    t = pd.to_datetime(df["VALID"], format="%Y%m%d%H%M", utc=True, errors="coerce")
    out = pd.DataFrame({"time": t, "lat": pd.to_numeric(df.get("LAT"), errors="coerce"),
                        "lon": pd.to_numeric(df.get("LON"), errors="coerce")})
    return out.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)


def _haversine_mi(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)) / MI_TO_KM)


def _catalog_pkl(year: int, month: int):
    return C.REPORTS_CACHE / f"catalog_{year:04d}{month:02d}.pkl"


@lru_cache(maxsize=36)
def _month_catalog(year: int, month: int) -> tuple:
    """Catalog for one calendar month (fires active any time that month), with a
    ±2-day pad. Cached **on disk** (pickle) as well as in-memory, so parallel build
    workers share one NIFC fetch per month instead of each re-querying it. (Months
    are historical for training, so no freshness check; delete the pkl to refresh.)"""
    import pickle

    pkl = _catalog_pkl(year, month)
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                return tuple(pickle.load(f))
        except Exception:  # noqa: BLE001 - corrupt cache -> rebuild
            pass
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = (datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
           - timedelta(seconds=1))
    recs = _build_catalog(start - timedelta(days=2), end + timedelta(days=2))
    try:
        with open(pkl, "wb") as f:
            pickle.dump(recs, f)
    except Exception:  # noqa: BLE001
        pass
    return tuple(recs)


def active_fires(valid_time: datetime):
    """Fires active at ``valid_time``, each as ``(points (M,2 lat/lon), final_size,
    deadly)`` using the footprint as-of that time (latest progression perimeter ≤
    valid_time, else the static incident point)."""
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=timezone.utc)
    # current + previous month catch fires that started before this month
    cats = list(_month_catalog(valid_time.year, valid_time.month))
    pm = (valid_time.year - (valid_time.month == 1), (valid_time.month - 2) % 12 + 1)
    cats += list(_month_catalog(*pm))
    seen, out = set(), []
    for rec in cats:
        if rec.irwin in seen or not (rec.start <= valid_time <= rec.end):
            continue
        seen.add(rec.irwin)
        pts = None
        prior = [tp for tp in rec.footprint if tp[0] <= valid_time]
        if prior:
            pts = prior[-1][1]                       # footprint as of valid_time
        elif rec.static_pt is not None:
            pts = np.array([rec.static_pt])
        if pts is None or len(pts) == 0:
            continue
        out.append((pts, rec.final_size, rec.deadly))
    return out


def prime_catalog(start: datetime, end: datetime) -> None:
    """Warm the month caches spanning [start, end] (call once before a build)."""
    d = start.replace(day=1)
    while d <= end:
        _month_catalog(d.year, d.month)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def active_fire_hours(start: datetime, end: datetime, min_fires: int = 1) -> list[datetime]:
    """UTC hours in [start, end] with ≥ ``min_fires`` active fires (label-bearing
    hours to build — includes ongoing multi-day fires, not just ignitions)."""
    hours: dict[datetime, int] = {}
    h = start.replace(minute=0, second=0, microsecond=0)
    while h <= end:
        n = len(active_fires(h))
        if n >= min_fires:
            hours[h] = n
        h += timedelta(hours=1)
    return sorted(hours)
