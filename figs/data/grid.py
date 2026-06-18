"""FIGS output grid: HRRR 3 km Lambert Conformal -> ~15 km block-averaged grid,
lat/lon of every output cell, and cached neighborhood / gradient stencils used
by the spatial-mean and gradient features.

The output grid is a regular grid in the HRRR Lambert projection, so distances
between nearby cells are well-approximated by planar projected distance. We
therefore build circular neighborhood stencils once (as integer cell offsets)
and reuse them across the whole grid.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from ..config import (
    BLOCK,
    FIGS_DX_KM,
    FIGS_NX,
    FIGS_NY,
    GRID_CACHE,
    HRRR_GRID,
    SPATIAL_MEAN_RADII_MI,
)

MI_TO_KM = 1.609344


# --------------------------------------------------------------------------- #
# Spherical Lambert Conformal Conic transform (HRRR uses a spherical earth, so
# these closed-form formulas are exact). Implemented in NumPy to avoid a pyproj
# dependency — pyproj/PROJ database resolution is fragile in some conda envs.
# Reference: Snyder, "Map Projections: A Working Manual" (1987), eqs. for the
# spherical LCC.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _lcc_constants() -> tuple[float, float, float, float, float]:
    """Return (n, F, rho0, lon0_rad, R) for the HRRR LCC projection."""
    g = HRRR_GRID
    R = g.earth_radius
    lat0 = np.radians(g.lat_0)
    lat1 = np.radians(g.lat_1)
    lat2 = np.radians(g.lat_2)
    lon0 = np.radians(g.lon_0)
    if abs(lat1 - lat2) < 1e-9:
        n = np.sin(lat1)
    else:
        n = np.log(np.cos(lat1) / np.cos(lat2)) / np.log(
            np.tan(np.pi / 4 + lat2 / 2) / np.tan(np.pi / 4 + lat1 / 2)
        )
    F = np.cos(lat1) * np.tan(np.pi / 4 + lat1 / 2) ** n / n
    rho0 = R * F / np.tan(np.pi / 4 + lat0 / 2) ** n
    return float(n), float(F), float(rho0), float(lon0), float(R)


def lcc_forward(lon_deg, lat_deg):
    """lon/lat (deg) -> projected x, y (m). Vectorized over array inputs."""
    n, F, rho0, lon0, R = _lcc_constants()
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    rho = R * F / np.tan(np.pi / 4 + lat / 2) ** n
    theta = n * (lon - lon0)
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y


def lcc_inverse(x, y):
    """projected x, y (m) -> lon, lat (deg). Vectorized over array inputs."""
    n, F, rho0, lon0, R = _lcc_constants()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.sign(n) * np.sqrt(x**2 + (rho0 - y) ** 2)
    theta = np.arctan2(x, rho0 - y)
    lon = lon0 + theta / n
    lat = 2 * np.arctan((R * F / rho) ** (1.0 / n)) - np.pi / 2
    return np.degrees(lon), np.degrees(lat)


@lru_cache(maxsize=1)
def hrrr_xy() -> tuple[np.ndarray, np.ndarray]:
    """1-D projected x and y coordinates (m) of the native HRRR grid points."""
    x0, y0 = lcc_forward(HRRR_GRID.sw_lon, HRRR_GRID.sw_lat)
    x = float(x0) + np.arange(HRRR_GRID.nx) * HRRR_GRID.dx
    y = float(y0) + np.arange(HRRR_GRID.ny) * HRRR_GRID.dy
    return x, y


@lru_cache(maxsize=1)
def figs_xy() -> tuple[np.ndarray, np.ndarray]:
    """1-D projected x and y (m) of FIGS output cell *centers* (block means of
    the native HRRR coordinates)."""
    x, y = hrrr_xy()
    xc = x[: FIGS_NX * BLOCK].reshape(FIGS_NX, BLOCK).mean(axis=1)
    yc = y[: FIGS_NY * BLOCK].reshape(FIGS_NY, BLOCK).mean(axis=1)
    return xc, yc


@lru_cache(maxsize=1)
def figs_latlon() -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) arrays of shape (FIGS_NY, FIGS_NX) for the output grid."""
    cache = GRID_CACHE / "figs_latlon.npz"
    if cache.exists():
        d = np.load(cache)
        return d["lat"], d["lon"]
    xc, yc = figs_xy()
    xx, yy = np.meshgrid(xc, yc)        # (NY, NX)
    lon, lat = lcc_inverse(xx, yy)
    np.savez(cache, lat=lat, lon=lon)
    return lat, lon


# --------------------------------------------------------------------------- #
# Regrid: HRRR 3 km field -> FIGS ~15 km field by block averaging
# --------------------------------------------------------------------------- #
def block_average(field: np.ndarray) -> np.ndarray:
    """Average a native HRRR 2-D field (ny, nx) into BLOCK x BLOCK cells.

    Trailing rows/cols that don't fill a full block are dropped (consistent with
    ``FIGS_NY/FIGS_NX = ny//BLOCK``). NaNs propagate via ``np.nanmean`` so
    partially-missing blocks still yield a value.
    """
    if field.shape != (HRRR_GRID.ny, HRRR_GRID.nx):
        raise ValueError(
            f"expected native HRRR shape {(HRRR_GRID.ny, HRRR_GRID.nx)}, got {field.shape}"
        )
    cropped = field[: FIGS_NY * BLOCK, : FIGS_NX * BLOCK]
    blocks = cropped.reshape(FIGS_NY, BLOCK, FIGS_NX, BLOCK)
    return np.nanmean(blocks, axis=(1, 3))


def block_max(field: np.ndarray) -> np.ndarray:
    """Like ``block_average`` but takes the BLOCK x BLOCK maximum — used for
    local-maxima fields (reflectivity, updraft helicity) so a coarse cell counts
    as exceeding a threshold if any fine subcell does."""
    if field.shape != (HRRR_GRID.ny, HRRR_GRID.nx):
        raise ValueError(
            f"expected native HRRR shape {(HRRR_GRID.ny, HRRR_GRID.nx)}, got {field.shape}"
        )
    cropped = field[: FIGS_NY * BLOCK, : FIGS_NX * BLOCK]
    blocks = cropped.reshape(FIGS_NY, BLOCK, FIGS_NX, BLOCK)
    return np.nanmax(blocks, axis=(1, 3))


# --------------------------------------------------------------------------- #
# Neighborhood stencils (integer cell offsets within a radius)
# --------------------------------------------------------------------------- #
def _stencil_offsets(radius_mi: float) -> np.ndarray:
    """(K, 2) array of (dy, dx) cell offsets whose center distance is within
    ``radius_mi`` of the origin cell, using the ~15 km cell size."""
    radius_cells = radius_mi * MI_TO_KM / FIGS_DX_KM
    r = int(np.ceil(radius_cells))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    mask = (dy**2 + dx**2) <= radius_cells**2
    return np.stack([dy[mask], dx[mask]], axis=1).astype(np.int16)


@lru_cache(maxsize=None)
def stencil(radius_mi: float) -> np.ndarray:
    """Cached circular neighborhood stencil for a spatial-mean radius."""
    return _stencil_offsets(radius_mi)


def neighborhood_mean(field: np.ndarray, radius_mi: float) -> np.ndarray:
    """Mean of ``field`` over the circular neighborhood of each cell.

    Uses a uniform circular kernel via FFT-free correlation through cumulative
    summation is awkward for circles, so we accumulate shifted copies along the
    stencil. Edges use available cells only (count-normalized). ``field`` may
    contain NaN; NaNs are excluded from each local average.
    """
    off = stencil(radius_mi)
    ny, nx = field.shape
    valid = np.isfinite(field)
    filled = np.where(valid, field, 0.0)
    acc = np.zeros_like(filled, dtype=np.float64)
    cnt = np.zeros_like(filled, dtype=np.float64)
    for dy, dx in off:
        ys0, ys1 = max(0, dy), min(ny, ny + dy)
        xs0, xs1 = max(0, dx), min(nx, nx + dx)
        yd0, yd1 = max(0, -dy), min(ny, ny - dy)
        xd0, xd1 = max(0, -dx), min(nx, nx - dx)
        acc[yd0:yd1, xd0:xd1] += filled[ys0:ys1, xs0:xs1]
        cnt[yd0:yd1, xd0:xd1] += valid[ys0:ys1, xs0:xs1]
    out = np.full_like(acc, np.nan)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out


def grid_info() -> dict:
    """Summary dict (handy for logging / sanity checks)."""
    lat, lon = figs_latlon()
    return dict(
        ny=FIGS_NY,
        nx=FIGS_NX,
        dx_km=FIGS_DX_KM,
        lat_range=(float(lat.min()), float(lat.max())),
        lon_range=(float(lon.min()), float(lon.max())),
        spatial_radii_mi=list(SPATIAL_MEAN_RADII_MI),
    )


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    print(json.dumps(grid_info(), indent=2))
