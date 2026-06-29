"""Static terrain fields on the FIGS ~15 km grid (built once, cached).

Terrain + terrain texture from a real DEM via the ``elevation`` library (SRTM),
regridded to the FIGS grid: elevation, slope, signed east/north slope components
(``elev_gradx``/``elev_grady``), aspect sin/cos, plus ruggedness texture
(TPI / TRI / elevation-std / slope-std over a 25 mi neighborhood). Falls back to
the HRRR surface geopotential if ``elevation``/``rasterio`` aren't installed.

Build once (or on first ``load_terrain_fields()`` call if the cache is missing):
    build_terrain()                  # SRTM via `elevation` (or HRRR fallback)
then ``load_terrain_fields()`` everywhere (cheap + memoized).

Optional deps: ``pip install elevation rasterio``.

Ported from ``figs_w/data/static.py`` (the terrain subset); FIGS only needs the
terrain family (no land-use / population-density).
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from ..config import (
    HRRR_GRID,
    STATIC_CACHE,
    STATIC_TERRAIN_FIELDS,
    STATIC_TERRAIN_TEXTURE_FIELDS,
)
from . import grid
from .grid import FIGS_DX_KM

_TERRAIN_NPZ = STATIC_CACHE / "terrain.npz"
_TEXTURE_RADIUS_MI = 25.0
_CONUS_BOUNDS = (-125.0, 24.0, -66.5, 50.0)   # W, S, E, N for the SRTM clip
_SRTM_BOX_DEG = 10.0


def _figs_crs_transform():
    """(proj4 CRS string, affine transform, (ny, nx)) of the FIGS 15 km grid, for
    rasterio reprojection. Spherical-earth Lambert Conformal matching the HRRR grid."""
    from affine import Affine

    p = HRRR_GRID.proj_params()
    crs = (f"+proj=lcc +lat_1={p['lat_1']} +lat_2={p['lat_2']} +lat_0={p['lat_0']} "
           f"+lon_0={p['lon_0']} +R={p['R']} +units=m +no_defs")
    xc, yc = grid.figs_xy()
    dx = float(xc[1] - xc[0]); dy = float(yc[1] - yc[0])
    transform = Affine(dx, 0.0, float(xc[0]) - dx / 2.0,
                       0.0, dy, float(yc[0]) - dy / 2.0)
    return crs, transform, (len(yc), len(xc))


def _reproject_to_figs(src_path: str, *, resampling="average", band: int = 1, src_nodata=None):
    """Reproject one band of a raster onto the FIGS grid. Returns (ny, nx) float32."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs, transform, (ny, nx) = _figs_crs_transform()
    dst = np.full((ny, nx), np.nan, dtype="float32")
    rs = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        reproject(source=rasterio.band(src, band), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src_nodata if src_nodata is not None else src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=rs)
    return dst


def _gradients(elev: np.ndarray):
    """Signed elevation gradient components (dz/dx east, dz/dy north; rise/run),
    plus slope magnitude and aspect sin/cos."""
    dxy = FIGS_DX_KM * 1000.0
    dzdy = np.gradient(elev, dxy, axis=0)
    dzdx = np.gradient(elev, dxy, axis=1)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    aspect = np.arctan2(-dzdy, -dzdx)
    return dzdx, dzdy, slope, np.sin(aspect), np.cos(aspect)


def _texture(elev, slope) -> dict[str, np.ndarray]:
    def nstd(f):
        m = grid.neighborhood_mean(f, _TEXTURE_RADIUS_MI)
        m2 = grid.neighborhood_mean(f * f, _TEXTURE_RADIUS_MI)
        return np.sqrt(np.maximum(m2 - m * m, 0.0))
    return {"tpi": elev - grid.neighborhood_mean(elev, _TEXTURE_RADIUS_MI),
            "tri": nstd(elev), "elev_std": nstd(elev), "slope_std": nstd(slope)}


def _srtm_boxes(step: float = _SRTM_BOX_DEG):
    w, s, e, n = _CONUS_BOUNDS
    boxes = []
    x = w
    while x < e:
        y = s
        while y < n:
            boxes.append((x, y, min(x + step, e), min(y + step, n)))
            y += step
        x += step
    return boxes


def _srtm_box_to_figs(args):
    """Worker: clip ONE box's SRTM via ``elevation`` (own cache dir → no cross-box
    race) and reproject onto the full FIGS grid (NaN outside the box)."""
    import os
    import tempfile

    import elevation

    idx, box = args
    root = os.path.join(tempfile.gettempdir(), "figs_srtm")
    os.makedirs(root, exist_ok=True)
    tif = os.path.join(root, f"box_{idx:02d}.tif")
    cache = os.path.join(root, f"cache_{idx:02d}")
    try:
        if not os.path.exists(tif):
            elevation.clip(bounds=box, output=tif, product="SRTM3",
                           cache_dir=cache, max_download_tiles=64)
        return _reproject_to_figs(tif, resampling="average")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SRTM box {idx} {box} failed ({str(e)[:70]})", file=sys.stderr)
        return None


def _elev_from_srtm() -> np.ndarray | None:
    """SRTM DEM over CONUS via the ``elevation`` library, fetched in parallel ~10°
    batches, each reprojected to the FIGS grid and mean-combined. None if deps absent."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    try:
        import elevation  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    boxes = list(enumerate(_srtm_boxes()))
    workers = int(os.environ.get("FIGS_SRTM_WORKERS", "6"))
    print(f"[srtm] fetching {len(boxes)} ~{_SRTM_BOX_DEG:.0f}° batches ({workers} parallel)…",
          file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        parts = [a for a in ex.map(_srtm_box_to_figs, boxes) if a is not None]
    if not parts:
        print("[warn] all SRTM batches failed; HRRR fallback", file=sys.stderr)
        return None
    with np.errstate(invalid="ignore"):
        elev = np.nanmean(np.stack(parts, axis=0), axis=0)
    return np.nan_to_num(elev, nan=0.0).astype(np.float32)


def _elev_from_hrrr() -> np.ndarray:
    from datetime import datetime, timezone

    from . import hrrr_store
    sfc = hrrr_store.surface_fields_combined(datetime(2023, 6, 1, 12, tzinfo=timezone.utc), 0)
    return grid.block_average(np.asarray(sfc["zsfc"], dtype=float)).astype(np.float32)


def build_terrain() -> dict[str, np.ndarray]:
    """Build + cache terrain & texture from SRTM (preferred) or HRRR orography."""
    elev = _elev_from_srtm()
    if elev is None:
        elev = _elev_from_hrrr()
    gx, gy, slope, asin, acos = _gradients(elev)
    out = {"elev": elev, "slope": slope.astype(np.float32),
           "elev_gradx": gx.astype(np.float32), "elev_grady": gy.astype(np.float32),
           "aspect_sin": asin.astype(np.float32), "aspect_cos": acos.astype(np.float32)}
    out.update({k: v.astype(np.float32) for k, v in _texture(elev, slope).items()})
    STATIC_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(_TERRAIN_NPZ, **{k: out[k] for k in
                              STATIC_TERRAIN_FIELDS + STATIC_TERRAIN_TEXTURE_FIELDS})
    return out


@lru_cache(maxsize=1)
def load_terrain_fields() -> dict[str, np.ndarray]:
    """All static terrain fields on the 15 km grid (built on first use if missing)."""
    fields = STATIC_TERRAIN_FIELDS + STATIC_TERRAIN_TEXTURE_FIELDS
    if _TERRAIN_NPZ.exists():
        z = np.load(_TERRAIN_NPZ)
        if all(k in z.files for k in fields):
            return {k: z[k].astype(np.float32) for k in fields}
    out = build_terrain()
    return {k: out[k].astype(np.float32) for k in fields}
