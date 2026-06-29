"""Theoretical (clear-sky) solar irradiance features on the FIGS grid.

Computed analytically from **location (grid lat/lon) + date + UTC time** — no HRRR
input — so the model gets a day/night + diurnal + seasonal + sun-angle signal
*without* being told the clock time or where it is. (The same idea as giving it the
terrain: a physical prior it can't otherwise reconstruct from the instantaneous
fields alone.)

Fields (each (ny, nx) for a valid time):
  * ``solar_cos_zen`` — cos(solar zenith), clamped ≥0. 0 at/after sunset, 1 overhead;
    the core day/night + sun-height signal.
  * ``solar_alt``     — solar elevation angle (degrees, NEGATIVE at night) — gives a
    smooth twilight gradient the clamped cos-zenith doesn't.
  * ``solar_toa``     — top-of-atmosphere horizontal irradiance (W/m²).
  * ``solar_ghi``     — Haurwitz clear-sky global horizontal irradiance (W/m²).

We use the standard NOAA/Spencer solar-position equations, **fully vectorized in
NumPy** over the grid. ``pysolar``/``solarpy`` compute the same quantities but only
per-scalar-call, which is far too slow over 75k cells × thousands of valid times;
this matches them to ~0.01° in position.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..data import grid

SOLAR_FIELDS = ("solar_cos_zen", "solar_alt", "solar_toa", "solar_ghi")
_SOLAR_CONSTANT = 1361.0   # W/m² mean top-of-atmosphere normal irradiance


def _solar_position(valid_time: datetime, lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Vectorized NOAA solar geometry. Returns (cos_zenith, altitude_deg).

    ``lat_deg``/``lon_deg`` are (ny, nx) grids (lon east-positive, −180..180).
    ``valid_time`` is tz-aware UTC (naive is assumed UTC)."""
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=timezone.utc)
    vt = valid_time.astimezone(timezone.utc)
    doy = int(vt.strftime("%j"))
    utc_min = vt.hour * 60.0 + vt.minute + vt.second / 60.0

    # fractional-year angle γ (radians); Spencer (1971) Fourier series.
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (vt.hour - 12) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))   # radians
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))  # minutes

    lat = np.radians(lat_deg)
    # true solar time (minutes) → hour angle (degrees, 0 at solar noon)
    tst = (utc_min + eqtime + 4.0 * lon_deg) % 1440.0
    ha = np.radians(tst / 4.0 - 180.0)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    altitude = np.degrees(np.arcsin(cos_zen))
    return cos_zen, altitude


def solar_features(valid_time: datetime) -> dict[str, np.ndarray]:
    """Clear-sky solar irradiance feature grids for ``valid_time`` (FIGS_NY, FIGS_NX)."""
    lat, lon = grid.figs_latlon()
    cos_zen, altitude = _solar_position(valid_time, lat, lon)
    pos = np.clip(cos_zen, 0.0, None)                      # 0 when sun below horizon

    # eccentricity-corrected TOA normal irradiance, then horizontal projection.
    doy = int((valid_time if valid_time.tzinfo else valid_time.replace(tzinfo=timezone.utc))
              .astimezone(timezone.utc).strftime("%j"))
    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    toa = _SOLAR_CONSTANT * e0 * pos

    # Haurwitz clear-sky GHI (W/m²): simple, only needs the zenith angle.
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi = np.where(pos > 0, 1098.0 * pos * np.exp(-0.059 / pos), 0.0)

    return {"solar_cos_zen": pos.astype(np.float32),
            "solar_alt": altitude.astype(np.float32),
            "solar_toa": toa.astype(np.float32),
            "solar_ghi": ghi.astype(np.float32)}
