"""Static geography fields on the FIGS ~15 km grid (built once, cached).

Three families, ALL given the full Tier-1 spatial treatment downstream:

  * **terrain** + **terrain texture** — from a real DEM via the ``elevation``
    library (SRTM), regridded to the FIGS grid; elevation/slope/aspect plus
    ruggedness (TRI/TPI, elevation/slope neighborhood std). Falls back to the HRRR
    surface geopotential if ``elevation``/``rasterio`` aren't installed.
  * **land use** — USGS **NLCD** (https://www.usgs.gov/centers/eros/science/
    national-land-cover-database): per-cell FRACTIONAL cover of forest / shrub /
    grass / crop / urban / water (each NLCD class reprojected as a 0/1 mask with
    average resampling → fraction).
  * **population density** — **WorldPop** USA (https://data.humdata.org/dataset/
    worldpop-population-density-for-united-states-of-america): people·km⁻²,
    area-averaged into each FIGS cell (+ a log1p companion).

Build once:
    build_terrain()                      # SRTM via `elevation` (or HRRR fallback)
    build_landuse("nlcd_conus.tif")      # downloaded NLCD raster
    build_popdensity("worldpop_usa.tif") # downloaded WorldPop raster
then ``load_static_fields()`` everywhere (cheap + memoized).

Optional deps: ``pip install elevation rasterio``. Rasters are large; download the
NLCD + WorldPop GeoTIFFs once and pass their paths to the build_* functions.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np

from figs.config import HRRR_GRID
from figs.data import grid
from figs.data.grid import FIGS_DX_KM

from ..config import (
    STATIC_CACHE,
    STATIC_FIELDS,
    STATIC_LANDUSE_FIELDS,
    STATIC_POPDENSITY_FIELDS,
    STATIC_TERRAIN_FIELDS,
    STATIC_TERRAIN_TEXTURE_FIELDS,
)

_TERRAIN_NPZ = STATIC_CACHE / "terrain.npz"
_LANDUSE_NPZ = STATIC_CACHE / "landuse.npz"
_POPDENS_NPZ = STATIC_CACHE / "popdensity.npz"
_TEXTURE_RADIUS_MI = 25.0
_CONUS_BOUNDS = (-125.0, 24.0, -66.5, 50.0)   # W, S, E, N for the SRTM clip

# NLCD class code → FIGS-W land-use group (fractions of these 6 are the features;
# barren/wetland codes are intentionally left out, so fractions need not sum to 1).
NLCD_GROUPS = {
    "lu_water": (11, 12),
    "lu_urban": (21, 22, 23, 24),
    "lu_forest": (41, 42, 43),
    "lu_shrub": (51, 52),
    "lu_grass": (71, 72, 73, 74),
    "lu_crop": (81, 82),
}


# --------------------------------------------------------------------------- #
# FIGS destination grid as a rasterio CRS + affine transform
# --------------------------------------------------------------------------- #
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


def _reproject_to_figs(src_path: str, *, resampling="average", band: int = 1,
                       src_nodata=None):
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


# --------------------------------------------------------------------------- #
# Terrain (SRTM via the `elevation` library; HRRR fallback)
# --------------------------------------------------------------------------- #
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


_SRTM_BOX_DEG = 10.0   # CONUS tile size per batch (≤ ~9 SRTM3 5° tiles → under the cap)


def _srtm_boxes(step: float = _SRTM_BOX_DEG):
    """Tile the CONUS bounds into ``step``° boxes (left, bottom, right, top)."""
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
    race) and reproject it onto the full FIGS grid (NaN outside the box). Returns
    the (ny, nx) array, or None on failure (e.g. an all-ocean box)."""
    import os
    import tempfile

    import elevation

    idx, box = args
    root = os.path.join(tempfile.gettempdir(), "figs_w_srtm")
    os.makedirs(root, exist_ok=True)
    tif = os.path.join(root, f"box_{idx:02d}.tif")
    cache = os.path.join(root, f"cache_{idx:02d}")          # isolate concurrent clips
    try:
        if not os.path.exists(tif):
            elevation.clip(bounds=box, output=tif, product="SRTM3",
                           cache_dir=cache, max_download_tiles=64)
        return _reproject_to_figs(tif, resampling="average")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SRTM box {idx} {box} failed ({str(e)[:70]})", file=sys.stderr)
        return None


def _elev_from_srtm() -> np.ndarray | None:
    """SRTM DEM over CONUS via the ``elevation`` library, fetched in **parallel
    ~10° batches** (each under the per-clip tile cap), each reprojected to the FIGS
    grid and mean-combined. Returns None if the optional deps aren't available."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    try:
        import elevation  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    boxes = list(enumerate(_srtm_boxes()))
    workers = int(os.environ.get("FIGS_W_SRTM_WORKERS", "6"))
    print(f"[srtm] fetching {len(boxes)} ~{_SRTM_BOX_DEG:.0f}° batches "
          f"({workers} parallel)…", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        parts = [a for a in ex.map(_srtm_box_to_figs, boxes) if a is not None]
    if not parts:
        print("[warn] all SRTM batches failed; HRRR fallback", file=sys.stderr)
        return None
    with np.errstate(invalid="ignore"):
        elev = np.nanmean(np.stack(parts, axis=0), axis=0)   # mean where boxes overlap
    return np.nan_to_num(elev, nan=0.0).astype(np.float32)


def _elev_from_hrrr() -> np.ndarray:
    from datetime import datetime, timezone

    from figs.data import hrrr_store
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
    np.savez(_TERRAIN_NPZ, **{k: out[k] for k in
                              STATIC_TERRAIN_FIELDS + STATIC_TERRAIN_TEXTURE_FIELDS})
    return out


# --------------------------------------------------------------------------- #
# Land use (NLCD) and population density (WorldPop)
# --------------------------------------------------------------------------- #
def _require_raster(path: str, what: str, url: str):
    from pathlib import Path

    if not path or not Path(path).expanduser().exists():
        raise FileNotFoundError(
            f"{what} raster not found: {path!r}\n"
            f"  Download the GeoTIFF first, then pass its real path. Source: {url}")
    return str(Path(path).expanduser())


def build_landuse(nlcd_path: str) -> dict[str, np.ndarray]:
    """Per-cell fractional cover of each NLCD group, cached. Each class is
    reprojected as a 0/1 mask with average resampling → fraction within the cell."""
    nlcd_path = _require_raster(nlcd_path, "NLCD land-cover",
                                "https://www.mrlc.gov/data (NLCD CONUS Land Cover)")
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs, transform, (ny, nx) = _figs_crs_transform()
    out: dict[str, np.ndarray] = {}
    with rasterio.open(nlcd_path) as src:
        codes = src.read(1)
        for field in STATIC_LANDUSE_FIELDS:
            mask = np.isin(codes, NLCD_GROUPS[field]).astype("float32")
            dst = np.zeros((ny, nx), dtype="float32")
            reproject(source=mask, destination=dst, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.average)
            out[field] = dst
    np.savez(_LANDUSE_NPZ, **out)
    return out


def build_popdensity(worldpop_path: str) -> dict[str, np.ndarray]:
    """Area-averaged WorldPop density (people·km⁻²) per FIGS cell (+ log1p), cached."""
    worldpop_path = _require_raster(
        worldpop_path, "WorldPop population-density",
        "https://data.humdata.org/dataset/worldpop-population-density-for-united-states-of-america")
    dens = _reproject_to_figs(worldpop_path, resampling="average")
    dens = np.nan_to_num(dens, nan=0.0).astype(np.float32)
    out = {"popdens": dens, "popdens_log": np.log1p(np.maximum(dens, 0.0)).astype(np.float32)}
    np.savez(_POPDENS_NPZ, **out)
    return out


# --------------------------------------------------------------------------- #
# Load (memoized)
# --------------------------------------------------------------------------- #
def _load_npz(path, fields):
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in fields if k in z.files}


@lru_cache(maxsize=1)
def load_static_fields() -> dict[str, np.ndarray]:
    """All static fields on the 15 km grid (terrain built on first use if missing;
    land-use / population loaded from cache, else zeros + a warning to run build_*)."""
    ny, nx = grid.FIGS_NY, grid.FIGS_NX
    terrain = _load_npz(_TERRAIN_NPZ, STATIC_TERRAIN_FIELDS + STATIC_TERRAIN_TEXTURE_FIELDS) \
        or build_terrain()
    out: dict[str, np.ndarray] = dict(terrain)
    for path, fields, how in ((_LANDUSE_NPZ, STATIC_LANDUSE_FIELDS, "build_landuse(NLCD.tif)"),
                              (_POPDENS_NPZ, STATIC_POPDENSITY_FIELDS, "build_popdensity(WorldPop.tif)")):
        loaded = _load_npz(path, fields)
        if loaded is None:
            print(f"[warn] {path.name} missing — run static.{how}; zeros for now",
                  file=sys.stderr, flush=True)
            loaded = {k: np.zeros((ny, nx), np.float32) for k in fields}
        out.update(loaded)
    return {k: out.get(k, np.zeros((ny, nx), np.float32)) for k in STATIC_FIELDS}
