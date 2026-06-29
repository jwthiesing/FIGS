"""Label construction for FIGS training.

For a given valid time we produce, on the FIGS ~15 km grid:
  * ``{hazard}``        : 1 if any severe report of that hazard falls within the
                          neighborhood (25 mi, ±30 min) of the cell.
  * ``{hazard}_sig``    : 1 if a *significant*-severe report does.
  * ``{hazard}_bin``    : conditional-intensity bin index of the strongest
                          nearby report (-1 if none, or tornado EF unknown).
  * ``{hazard}_pib``    : Peak Intensity Bin (0..6 == PIB1..PIB7) of the strongest
                          nearby report by RATED WIND SPEED (mph) / hail size; -1 if
                          none. Tornado PIB comes only from DAT damage-wind (mph) —
                          its tracks are stamped along their full path, like SVRGIS;
                          wind PIB from native LSR mph + DAT damage-wind backfill;
                          hail PIB from diameter.

The hazard flags train the p(hazard) models; the bin labels train the
conditional-intensity models; the PIB labels train the PIB subsystem (on positive
cells only).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..config import (
    FIGS_NX,
    FIGS_NY,
    HAZARDS,
    INTENSITY_BINS,
    NEIGHBORHOOD_RADIUS_MI,
    NEIGHBORHOOD_TIME_MIN,
    PIB_BINS,
    SEVERE_THRESHOLDS,
    SIGNIFICANT_THRESHOLDS,
)
from . import grid, reports


def pib_bin(hazard: str, value: float) -> int:
    """Peak Intensity Bin index (0..6 == PIB1..PIB7) for a rated wind speed (mph for
    tor/wind) or hail diameter (in), using the lower-bound edges in ``PIB_BINS``;
    -1 if the value is missing / non-positive (no PIB assigned)."""
    if value is None or not np.isfinite(value) or value <= 0:
        return -1
    edges = PIB_BINS[hazard]["edges"]
    return int(np.searchsorted(edges, value, side="right"))   # below edges[0] -> 0 (PIB1)


def intensity_bin(hazard: str, value: float) -> int:
    """Map an intensity ``value`` (knots/inches/EF, matching the report's stored
    unit) to its conditional-intensity bin index, or -1 if below severe / unknown."""
    spec = INTENSITY_BINS[hazard]
    edges = spec["edges"]
    if spec["kind"] == "ef":
        v = int(value)
        # EFU / unknown-EF tornadoes (v < 0) are NOT counted as EF0 — they return
        # -1 and are excluded from the conditional EF distribution. (They still
        # count as p(tor) positives; see build_labels.)
        if v < 0:
            return -1
        return min(v, len(spec["labels"]) - 1)
    if value < edges[0]:
        return -1
    return int(np.searchsorted(edges, value, side="right") - 1)


def _severe_value_threshold(hazard: str, thresholds: dict) -> float:
    return {
        "tor": float(thresholds["tor_ef"]),
        "wind": float(thresholds["wind_kt"]),
        "hail": float(thresholds["hail_in"]),
    }[hazard]


def _report_cell_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest FIGS (iy, ix) for each report; ``valid`` masks in-domain rows."""
    xc, yc = grid.figs_xy()
    x, y = grid.lcc_forward(df["lon"].to_numpy(), df["lat"].to_numpy())
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    ix = np.round((x - xc[0]) / dx).astype(int)
    iy = np.round((y - yc[0]) / dy).astype(int)
    valid = (ix >= 0) & (ix < FIGS_NX) & (iy >= 0) & (iy < FIGS_NY)
    return iy, ix, valid


def build_labels(valid_time: datetime) -> dict[str, np.ndarray]:
    """Per-cell label arrays for ``valid_time`` (shape (FIGS_NY, FIGS_NX))."""
    out: dict[str, np.ndarray] = {}
    for h in HAZARDS:
        out[h] = np.zeros((FIGS_NY, FIGS_NX), dtype=np.int8)
        out[f"{h}_sig"] = np.zeros((FIGS_NY, FIGS_NX), dtype=np.int8)
        out[f"{h}_bin"] = np.full((FIGS_NY, FIGS_NX), -1, dtype=np.int8)
        out[f"{h}_pib"] = np.full((FIGS_NY, FIGS_NX), -1, dtype=np.int8)

    off = grid.stencil(NEIGHBORHOOD_RADIUS_MI)  # (K, 2) dy, dx offsets

    df = reports.reports_in_window(valid_time, NEIGHBORHOOD_TIME_MIN)
    # DAT straight-line wind damage points (windspeed mph), timed by the nearest
    # same-day wind LSR — backfills wind presence + PIB magnitude.
    dat_wind = reports.dat_wind_reports_in_window(valid_time, NEIGHBORHOOD_TIME_MIN)
    if not dat_wind.empty:
        df = dat_wind if df.empty else pd.concat([df, dat_wind], ignore_index=True)
    if not df.empty:
        iy_all, ix_all, valid = _report_cell_indices(df)
        wind_mph = df["wind_mph"].to_numpy() if "wind_mph" in df.columns \
            else np.full(len(df), np.nan)
        for h in HAZARDS:
            sev_thr = _severe_value_threshold(h, SEVERE_THRESHOLDS)
            sig_thr = _severe_value_threshold(h, SIGNIFICANT_THRESHOLDS)
            sub = df["hazard"].to_numpy() == h
            rows = np.where(sub & valid)[0]
            for ri in rows:
                mag = float(df["magnitude"].iloc[ri])
                # tornadoes always count as severe (EF may be unknown / -1)
                is_severe = (h == "tor") or (mag >= sev_thr)
                if not is_severe:
                    continue
                is_sig = mag >= sig_thr if not (h == "tor" and df["ef"].iloc[ri] < 0) else False
                b = intensity_bin(h, mag)
                # PIB magnitude: wind/tornado use rated MPH, hail uses inches. Tornado
                # point reports carry no mph (EF only) -> no PIB here (DAT tracks below).
                pib_val = wind_mph[ri] if h == "wind" else (mag if h == "hail" else np.nan)
                pb = pib_bin(h, float(pib_val)) if np.isfinite(pib_val) else -1
                cy = iy_all[ri] + off[:, 0]
                cx = ix_all[ri] + off[:, 1]
                m = (cy >= 0) & (cy < FIGS_NY) & (cx >= 0) & (cx < FIGS_NX)
                cy, cx = cy[m], cx[m]
                out[h][cy, cx] = 1
                if is_sig:
                    out[f"{h}_sig"][cy, cx] = 1
                # b < 0 (e.g. EFU tornado / below-severe) does not enter the intensity
                # distribution, so EFU tornadoes never count as EF0.
                if b >= 0:
                    out[f"{h}_bin"][cy, cx] = np.maximum(out[f"{h}_bin"][cy, cx], b)
                if pb >= 0:
                    out[f"{h}_pib"][cy, cx] = np.maximum(out[f"{h}_pib"][cy, cx], pb)

    # SVRGIS tornado tracks are stamped regardless of whether point reports exist
    # in this exact window (track start times can differ from LSR times).
    _stamp_tornado_tracks(valid_time, out, off)
    # DAT tornado damage tracks: full path + rated maxwind (mph) -> tornado PIB.
    _stamp_dat_tornado_pib(valid_time, out, off)
    return out


def _stamp_tornado_tracks(valid_time, out, off):
    """Stamp the full SVRGIS tornado-track path (not just the touchdown point) into
    the tornado labels, so the probability estimate reflects the whole damage path.
    Samples points along each start→end segment (~5 km spacing) and stamps the
    neighborhood stencil at each."""
    tracks = reports.svrgis_tracks_in_window(valid_time, NEIGHBORHOOD_TIME_MIN)
    if tracks.empty:
        return
    xc, yc = grid.figs_xy()
    dx, dy = xc[1] - xc[0], yc[1] - yc[0]
    for _, tr in tracks.iterrows():
        # approximate path length (km) -> sample every ~5 km
        midlat = np.radians((tr.slat + tr.elat) / 2)
        dlat_km = (tr.elat - tr.slat) * 111.0
        dlon_km = (tr.elon - tr.slon) * 111.0 * np.cos(midlat)
        length_km = float(np.hypot(dlat_km, dlon_km))
        n = max(2, int(length_km / 5.0) + 1)
        lats = np.linspace(tr.slat, tr.elat, n)
        lons = np.linspace(tr.slon, tr.elon, n)
        x, y = grid.lcc_forward(lons, lats)
        ix = np.round((x - xc[0]) / dx).astype(int)
        iy = np.round((y - yc[0]) / dy).astype(int)
        ef = int(tr.ef)
        b = intensity_bin("tor", float(ef)) if ef >= 0 else -1
        is_sig = ef >= int(SIGNIFICANT_THRESHOLDS["tor_ef"])
        for pi in range(n):
            cy = iy[pi] + off[:, 0]
            cx = ix[pi] + off[:, 1]
            m = (cy >= 0) & (cy < FIGS_NY) & (cx >= 0) & (cx < FIGS_NX)
            cy, cx = cy[m], cx[m]
            out["tor"][cy, cx] = 1
            if is_sig:
                out["tor_sig"][cy, cx] = 1
            if b >= 0:
                out["tor_bin"][cy, cx] = np.maximum(out["tor_bin"][cy, cx], b)
    return out


def _stamp_dat_tornado_pib(valid_time, out, off):
    """Stamp tornado PIB along each DAT damage track's FULL path (like the SVRGIS
    tracks), using the track's rated ``maxwind`` (mph) → PIB. DAT tracks are timed by
    association to the nearest same-day SVRGIS/LSR tornado (see reports), so they sit
    in the right valid-hour window. Tracks with no DAT speed are skipped (no PIB)."""
    tracks = reports.dat_tornado_tracks_in_window(valid_time, NEIGHBORHOOD_TIME_MIN)
    if tracks.empty:
        return
    xc, yc = grid.figs_xy()
    dx, dy = xc[1] - xc[0], yc[1] - yc[0]
    for _, tr in tracks.iterrows():
        pb = pib_bin("tor", float(tr["maxwind_mph"]))
        ef = int(tr["ef"])
        b = intensity_bin("tor", float(ef)) if ef >= 0 else -1
        is_sig = ef >= int(SIGNIFICANT_THRESHOLDS["tor_ef"])
        path = np.asarray(tr["path"], dtype=float)         # (M, 2) lat, lon
        if path.ndim != 2 or len(path) == 0:
            continue
        # densify the polyline to ~5 km spacing so the stencil covers the whole path
        lats, lons = _densify_path(path[:, 0], path[:, 1])
        x, y = grid.lcc_forward(lons, lats)
        ix = np.round((x - xc[0]) / dx).astype(int)
        iy = np.round((y - yc[0]) / dy).astype(int)
        for pi in range(len(ix)):
            cy = iy[pi] + off[:, 0]
            cx = ix[pi] + off[:, 1]
            m = (cy >= 0) & (cy < FIGS_NY) & (cx >= 0) & (cx < FIGS_NX)
            cy, cx = cy[m], cx[m]
            # a DAT damage track is a first-class tornado occurrence along its path
            out["tor"][cy, cx] = 1
            if is_sig:
                out["tor_sig"][cy, cx] = 1
            if b >= 0:
                out["tor_bin"][cy, cx] = np.maximum(out["tor_bin"][cy, cx], b)
            if pb >= 0:
                out["tor_pib"][cy, cx] = np.maximum(out["tor_pib"][cy, cx], pb)


def _densify_path(lats: np.ndarray, lons: np.ndarray, step_km: float = 5.0):
    """Resample a lat/lon polyline to ~``step_km`` spacing (segment-wise linear)."""
    lats = np.asarray(lats, float); lons = np.asarray(lons, float)
    if len(lats) == 1:
        return lats, lons
    out_lat, out_lon = [], []
    for i in range(len(lats) - 1):
        midlat = np.radians((lats[i] + lats[i + 1]) / 2)
        dkm = np.hypot((lats[i + 1] - lats[i]) * 111.0,
                       (lons[i + 1] - lons[i]) * 111.0 * np.cos(midlat))
        n = max(2, int(dkm / step_km) + 1)
        out_lat.append(np.linspace(lats[i], lats[i + 1], n))
        out_lon.append(np.linspace(lons[i], lons[i + 1], n))
    return np.concatenate(out_lat), np.concatenate(out_lon)


def label_summary(labels: dict[str, np.ndarray]) -> dict:
    """Counts of positive cells per hazard (handy for sanity checks)."""
    s = {}
    for h in HAZARDS:
        s[h] = int(labels[h].sum())
        s[f"{h}_sig"] = int(labels[f"{h}_sig"].sum())
    return s
