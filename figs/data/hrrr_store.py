"""HRRR retrieval and caching via Herbie.

Pulls the variable/level subsets FIGS needs from the NOAA HRRR BDP S3 bucket
(``noaa-hrrr-bdp-pds``) using Herbie's ``.idx`` byte-range subsetting, reads them
with cfgrib/xarray, and returns native-grid (ny, nx) NumPy fields. Herbie and
cfgrib are imported lazily so the rest of FIGS imports without them.

Two product files are used:
  * ``prs`` (wrfprsf): 3-D isobaric TMP/DPT/UGRD/VGRD/HGT for profiles & hodographs
  * ``sfc`` (wrfsfcf): 2-m T/Td, 10/80-m wind, surface CAPE/CIN, REFC, REFD, MXUPHL
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

# cfgrib opens GRIB subsets via xr.merge() without an explicit ``compat`` (it is
# combining the level-type "hypercubes"), which emits a FutureWarning under recent
# xarray. The merge is internal to cfgrib, so we silence that one warning here —
# at module import, so it also applies inside the predict/build worker PROCESSES,
# where the notebook's top-level warnings filter does not reach.
warnings.filterwarnings(
    "ignore",
    message=r"In a future version of xarray the default value for compat",
    category=FutureWarning,
)
# Herbie matches the idx search with pandas str.contains(); our searches contain
# regex groups (e.g. ":(TMP|DPT|...):(...) mb:"), which makes pandas emit a
# UserWarning about match groups on every fetch. The grouping is intentional.
warnings.filterwarnings(
    "ignore",
    message=r"This pattern is interpreted as a regular expression, and has match groups",
    category=UserWarning,
)

from ..config import (
    FETCH_ISOBARIC_LEVELS,
    HRRR_CACHE,
    HRRR_GRID,
    HRRR_LONG_CYCLES,
    HRRR_LONG_LEN,
    HRRR_SHORT_LEN,
)

# native HRRR grid shape every freshly-read 2-D field must have; a concurrent
# byte-range fetch that truncates the GRIB can decode a field to a partial/1-D
# array (e.g. (1799,)), which slips past key-presence checks but explodes in the
# downstream block-average regrid — so the self-heal loops verify this shape too.
_NATIVE_SHAPE = (HRRR_GRID.ny, HRRR_GRID.nx)


def _is_native_2d(v) -> bool:
    return getattr(v, "shape", None) == _NATIVE_SHAPE

# Per-variable Herbie search strings (matched against GRIB .idx lines). cfgrib
# returns a separate "hypercube" per level type, and MXUPHL comes through with a
# generic name, so we fetch each field with an exact search and grab the single
# returned variable — robust against cfgrib short-name quirks.
# Isobaric (wrfprsf) 3-D fields. cfgrib short name -> FIGS key. VVEL ('w') is
# pulled at the full isobaric resolution and later interpolated to the fine
# profile levels.
PRS_CF_TO_KEY = {"t": "tmp", "dpt": "dpt", "u": "ugrd", "v": "vgrd", "gh": "hgt", "w": "vvel"}
# Restrict the isobaric download to FETCH_ISOBARIC_LEVELS (storage saving). The
# idx label is e.g. ":TMP:850 mb:"; we alternate the wanted integer levels.
_LVL_RE = "|".join(str(int(_l)) for _l in FETCH_ISOBARIC_LEVELS)
PRS_SEARCH = rf":(TMP|DPT|UGRD|VGRD|HGT|VVEL):({_LVL_RE}) mb:"

# Surface fields: normalized key -> idx search string. uh03/uh25 use the two
# updraft-helicity layers independently. Native HRRR CAPE/CIN (surface + 90/180 mb
# mixed layers) are included alongside FIGS's own computed thermodynamics. Soil,
# moisture, cloud-cover and categorical-precip fields round out the inputs.
SFC_SEARCHES = {
    "t2m": ":TMP:2 m above ground:",
    "td2m": ":DPT:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
    "psfc": ":PRES:surface:",
    "zsfc": ":HGT:surface:",
    "refc": ":REFC:",
    "refd": ":REFD:1000 m above ground:",
    "uh03": ":MXUPHL:3000-0 m above ground:",
    "uh25": ":MXUPHL:5000-2000 m above ground:",
    # native HRRR CAPE/CIN
    "hrrr_sbcape": ":CAPE:surface:",
    "hrrr_sbcin": ":CIN:surface:",
    "hrrr_mlcape90": ":CAPE:90-0 mb above ground:",
    "hrrr_mlcin90": ":CIN:90-0 mb above ground:",
    "hrrr_mlcape180": ":CAPE:180-0 mb above ground:",
    "hrrr_mlcin180": ":CIN:180-0 mb above ground:",
    # precipitable water / cloud cover / categorical precip
    "pwat": ":PWAT:entire atmosphere",
    "lcdc": ":LCDC:low cloud layer:",
    "mcdc": ":MCDC:middle cloud layer:",
    "hcdc": ":HCDC:high cloud layer:",
    "tcdc": ":TCDC:entire atmosphere:",
    "crain": ":CRAIN:surface:",
    "cfrzr": ":CFRZR:surface:",
    "cicep": ":CICEP:surface:",
    "csnow": ":CSNOW:surface:",
    # low-level relative vorticity (HRRR's surface/near-surface vorticity)
    "relv01": ":RELV:1000-0 m above ground:",
    "relv02": ":RELV:2000-0 m above ground:",
}

# Soil fields live in the pressure (prs/nat) product, not sfc. Top (skin) layer.
SOIL_SEARCHES = {
    "soilw": ":SOILW:0-0 m below ground:",
    "tsoil": ":TSOIL:0-0 m below ground:",
}


def cycle_length(cycle_hour: int) -> int:
    """Forecast length (hours) for a HRRR cycle initialized at ``cycle_hour``."""
    return HRRR_LONG_LEN if cycle_hour in HRRR_LONG_CYCLES else HRRR_SHORT_LEN


@dataclass(frozen=True)
class HRRRRun:
    """A single HRRR cycle + forecast hour reaching a target valid time."""

    run: datetime          # cycle init time (UTC, hour-aligned)
    fxx: int               # forecast hour

    @property
    def valid_time(self) -> datetime:
        return self.run + timedelta(hours=self.fxx)


def recent_runs_for_valid(
    valid_time: datetime,
    max_members: int,
    as_of: datetime | None = None,
    max_lag_hours: int = 48,
) -> list[HRRRRun]:
    """The most-recent HRRR cycles whose forecast reaches ``valid_time``.

    Walks back hour-by-hour from the valid time; a cycle qualifies if (a) it was
    initialized at/before ``as_of`` (the forecast issuance time — runs after it
    don't exist yet) and (b) its forecast length covers the required lead. Returns
    up to ``max_members`` runs, newest first (smallest lead).

    * ``as_of=None`` (hindcast): every prior run is assumed available, so the
      result is simply f01..f06 of the nearest cycles.
    * ``as_of`` set (real-time / issuance-faithful): far valid times fall back to
      the 6-hourly 48-h cycles, and moderate lead times return a **mix of 18-h and
      48-h cycles** (e.g. recent 18-h cycles at short leads plus older 48-h cycles
      at longer leads). Far hours naturally **narrow** to fewer members.
    """
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=timezone.utc)
    if as_of is not None and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    out: list[HRRRRun] = []
    for lag in range(1, max_lag_hours + 1):
        run = valid_time - timedelta(hours=lag)
        if as_of is not None and run > as_of:
            continue  # run not yet issued at forecast time
        if cycle_length(run.hour) >= lag:
            out.append(HRRRRun(run=run, fxx=lag))
            if len(out) >= max_members:
                break
    return out


def members_for(run: datetime, fxx: int, max_members: int = None) -> list[HRRRRun]:
    """Time-lagged ensemble for forecast hour ``fxx`` of a primary ``run``,
    issuance-capped at that run (real-time-faithful). Valid time = run + fxx."""
    from ..config import ENSEMBLE_MAX_MEMBERS

    if run.tzinfo is None:
        run = run.replace(tzinfo=timezone.utc)
    m = ENSEMBLE_MAX_MEMBERS if max_members is None else max_members
    return recent_runs_for_valid(run + timedelta(hours=fxx), m, as_of=run)


# transient S3/HTTP failures to retry (the bucket throttles / has brief outages)
_TRANSIENT = ("503", "502", "500", "504", "429", "Service Unavailable", "SlowDown",
              "Timeout", "timed out", "Connection", "ConnectionError", "reset by peer",
              "Temporary", "InternalError", "RequestTimeout")


def _with_retry(fn, *, what="fetch", tries: int = 5, base: float = 1.5):
    """Call ``fn`` with exponential backoff on TRANSIENT errors (e.g. the HRRR S3
    bucket returning 503). Re-raises immediately on non-transient errors and after
    the last try, so a genuinely missing file still fails fast."""
    import time

    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if i == tries - 1 or not any(s in msg for s in _TRANSIENT):
                raise
            wait = base * (2 ** i)
            print(f"[warn] transient {what} error ({msg.splitlines()[0][:80]}); "
                  f"retry {i + 1}/{tries - 1} in {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)


def _herbie(run: datetime, fxx: int, product: str):
    """Construct a Herbie object for a run/fxx/product (lazy import).

    Herbie probes the S3/Google sources over HTTP at construction time, so this is
    a network call — wrap it in the transient-error retry too (a stalled probe
    raising ``Read timed out`` would otherwise escape the read paths' retries)."""
    from herbie import Herbie

    return _with_retry(lambda: Herbie(
        run.strftime("%Y-%m-%d %H:%M"),
        model="hrrr",
        product=product,   # 'prs' or 'sfc'
        fxx=fxx,
        save_dir=str(HRRR_CACHE),
        verbose=False,     # suppress the "✅ Found ┊ …" banner per fetch
    ), what="herbie-init")


def _open(H, search, overwrite: bool = False):
    """Open a Herbie subset, always returning a list of xarray Datasets
    (cfgrib may split a search into several level-type 'hypercubes').
    ``overwrite`` forces a fresh re-download first (used to recover a truncated
    byte-range read — see ``isobaric_cube``)."""
    if overwrite:
        try:
            _with_retry(lambda: H.download(search, overwrite=True, verbose=False),
                        what="download")
        except Exception:  # noqa: BLE001 - fall through to xarray's own fetch
            pass
    res = _with_retry(lambda: H.xarray(search, remove_grib=False), what="xarray")
    return res if isinstance(res, list) else [res]


def _single_field(H, search) -> np.ndarray:
    """Fetch a search expected to resolve to one 2-D field; return it (ny, nx),
    squeezing any singleton level dimension. Raises KeyError if the search
    matches no GRIB messages (checked via the idx inventory to avoid Herbie/cfgrib
    erroring on an empty subset download)."""
    if len(_with_retry(lambda: H.inventory(search, verbose=False), what="inventory")) == 0:
        raise KeyError(f"no idx match for search {search!r}")
    dss = _open(H, search)
    try:
        for ds in dss:
            for v in ds.data_vars:
                return np.squeeze(np.asarray(ds[v].values, dtype=float))
    finally:
        for ds in dss:
            ds.close()
    raise KeyError(f"no variable matched search {search!r}")


def isobaric_cube(run: datetime, fxx: int, regrid=None) -> dict[str, np.ndarray]:
    """Return 3-D isobaric fields as (nlev, ny, nx) arrays, surface-first.

    Keys: 'tmp', 'dpt', 'ugrd', 'vgrd', 'hgt', 'vvel' plus 'levels' (the pressure
    levels present, descending mb). A single combined search downloads all
    variables in one pass.

    ``regrid`` (a callable on a native 2-D field) is applied per level as each
    variable is read, and the native array is freed immediately — so the full
    native cube (~1 GB/member) is never held in memory at once. Output is float32.
    """
    H = _herbie(run, fxx, "prs")
    # Concurrent S3 byte-range fetches can truncate the read, leaving some
    # variables with fewer pressure levels than others (an inconsistent cube that
    # crashes the profile interpolation downstream). Verify every variable has the
    # full level count and re-fetch fresh (overwrite) if not.
    last_bad = None
    for attempt in range(3):
        out: dict[str, np.ndarray] = {}
        levels = None
        malformed = []  # vars whose native levels weren't full (ny, nx) -> refetch
        for ds in _open(H, PRS_SEARCH, overwrite=(attempt > 0)):
            if "isobaricInhPa" not in ds.coords:
                ds.close()
                continue
            levs = np.asarray(ds["isobaricInhPa"].values, dtype=float)
            order = np.argsort(-levs)
            for var in ds.data_vars:
                key = PRS_CF_TO_KEY.get(str(var))
                if key is None:
                    continue
                native = np.asarray(ds[var].values, dtype=np.float32)[order]
                # verify each level is the full native grid BEFORE regridding —
                # else a truncated level would make block_average raise instead of
                # letting this loop self-heal with a fresh fetch.
                if native.ndim != 3 or native.shape[1:] != _NATIVE_SHAPE:
                    malformed.append((key, native.shape))
                    continue
                if regrid is not None:
                    out[key] = np.stack([regrid(native[i]) for i in range(native.shape[0])],
                                        axis=0).astype(np.float32)
                    del native  # free the native cube promptly
                else:
                    out[key] = native
                if levels is None or len(levs) > len(levels):
                    levels = levs[order]
            ds.close()  # release cfgrib/xarray handles + memory promptly
        nlev = 0 if levels is None else len(levels)
        short = [k for k, v in out.items() if v.shape[0] != nlev]
        if levels is not None and not short and not malformed:
            out["levels"] = levels
            return out
        last_bad = f"levels={nlev}, short-vars={short}, malformed={malformed}"
        print(f"[warn] truncated isobaric read {run:%Y-%m-%d %H}Z f{fxx:02d} "
              f"({last_bad}); re-fetching (attempt {attempt + 1}/3)", file=sys.stderr, flush=True)
    raise RuntimeError(f"isobaric_cube {run:%Y-%m-%d %H}Z f{fxx:02d}: incomplete after "
                       f"3 fetches ({last_bad}); reduce predict --workers")


# cfgrib short-name -> FIGS key for scalar surface fields (single level).
_SFC_SCALAR = {
    "tcc": "tcdc", "refc": "refc", "pwat": "pwat", "u10": "u10", "v10": "v10",
    "t2m": "t2m", "d2m": "td2m", "refd": "refd", "hcc": "hcdc", "lcc": "lcdc",
    "mcc": "mcdc", "crain": "crain", "cfrzr": "cfrzr", "cicep": "cicep",
    "csnow": "csnow", "sp": "psfc", "orog": "zsfc",
}


def _iter_levels(da, tol):
    """Yield (level_value, 2-D field) for each level of ``da`` along level-type
    ``tol``. A truncated byte-range read can decode just one level, in which case
    cfgrib collapses ``tol`` from a dimension to a scalar coordinate — then
    ``da.isel({tol: i})`` raises "Dimensions {...} do not exist". Handle both: if
    ``tol`` is still a dimension, iterate it (using the array's OWN coord so the
    index can't run past a short read); if it collapsed to a scalar coord, the
    array already IS the single level."""
    if tol in getattr(da, "dims", ()):
        vals = np.atleast_1d(da[tol].values)
        for i, lv in enumerate(vals):
            yield float(lv), da.isel({tol: i})
    elif tol in getattr(da, "coords", ()):
        yield float(np.atleast_1d(da[tol].values)[0]), da


def _map_surface(dss) -> dict[str, np.ndarray]:
    """Map the combined-search surface datasets to FIGS-normalized keys, handling
    the multi-layer fields (UH, RELV, ML CAPE/CIN) by their level values."""
    out: dict[str, np.ndarray] = {}

    def a(da):
        return np.asarray(da.values, dtype=float)

    for ds in dss:
        for v in ds.data_vars:
            da = ds[v]
            name = str(v)
            tol = da.attrs.get("GRIB_typeOfLevel", "")
            if name in _SFC_SCALAR and tol != "pressureFromGroundLayer":
                # surface cape/cin handled below; everything else is a plain scalar
                if not (name in ("cape", "cin")):
                    out[_SFC_SCALAR[name]] = np.squeeze(a(da))
            if name == "cape" and tol == "surface":
                out["hrrr_sbcape"] = np.squeeze(a(da))
            elif name == "cin" and tol == "surface":
                out["hrrr_sbcin"] = np.squeeze(a(da))
            elif name in ("cape", "cin") and tol == "pressureFromGroundLayer":
                key = "mlcape" if name == "cape" else "mlcin"
                for p, lvl in _iter_levels(da, tol):
                    out[f"hrrr_{key}{int(round(p)) // 100}"] = np.squeeze(a(lvl))
            elif name == "unknown" and tol == "heightAboveGroundLayer":  # MXUPHL
                for top, lvl in _iter_levels(da, tol):
                    if int(round(top)) == 3000:
                        out["uh03"] = np.squeeze(a(lvl))
                    elif int(round(top)) == 5000:
                        out["uh25"] = np.squeeze(a(lvl))
            elif name == "max_vo" and tol == "heightAboveGroundLayer":  # RELV
                for top, lvl in _iter_levels(da, tol):
                    if int(round(top)) == 1000:
                        out["relv01"] = np.squeeze(a(lvl))
                    elif int(round(top)) == 2000:
                        out["relv02"] = np.squeeze(a(lvl))
    return out


# surface fields the deterministic pipeline (profiles) cannot proceed without
_SFC_REQUIRED = ("t2m", "td2m", "u10", "v10", "psfc", "zsfc")


def surface_fields_combined(run: datetime, fxx: int) -> dict[str, np.ndarray]:
    """Like ``surface_fields`` but fetches all surface fields in ONE download +
    one cfgrib open (≈3x faster, far fewer requests), then maps + adds soil.

    Verifies the essential fields are present and re-fetches fresh (overwrite) if a
    concurrent byte-range read truncated the file — the surface analog of the
    ``isobaric_cube`` self-heal (else a missing 't2m' crashes profile building)."""
    H = _herbie(run, fxx, "sfc")
    combined = "|".join(SFC_SEARCHES.values())
    out, missing = {}, None
    for attempt in range(3):
        dss = _open(H, combined, overwrite=(attempt > 0))
        try:
            out = _map_surface(dss)
        except Exception as e:  # noqa: BLE001 - a truncated read can make the level
            # mapping itself raise (e.g. a collapsed level dim); treat as a failed
            # attempt so the loop re-fetches rather than crashing the forecast.
            out = {}
            print(f"[warn] surface read parse failed {run:%Y-%m-%d %H}Z f{fxx:02d} "
                  f"({type(e).__name__}: {str(e).splitlines()[0][:80]}); re-fetching "
                  f"(attempt {attempt + 1}/3)", file=sys.stderr, flush=True)
        finally:
            for ds in dss:
                ds.close()
        # a field present but truncated to a non-native shape (e.g. (1799,)) is as
        # unusable as a missing one — it would explode in block_average downstream.
        missing = [k for k in _SFC_REQUIRED if not _is_native_2d(out.get(k))]
        if not missing:
            break
        print(f"[warn] truncated surface read {run:%Y-%m-%d %H}Z f{fxx:02d} "
              f"(missing/malformed {missing}); re-fetching (attempt {attempt + 1}/3)",
              file=sys.stderr, flush=True)
    if missing:
        raise RuntimeError(f"surface_fields_combined {run:%Y-%m-%d %H}Z f{fxx:02d}: missing "
                           f"{missing} after 3 fetches; reduce predict --workers")
    # soil lives in the pressure product
    Hp = _herbie(run, fxx, "prs")
    for key, search in SOIL_SEARCHES.items():
        try:
            out[key] = _single_field(Hp, search)
        except Exception:  # noqa: BLE001
            continue
    if "psfc" in out and np.nanmedian(out["psfc"]) > 2000.0:
        out["psfc"] = out["psfc"] / 100.0
    return out


def surface_prob_fields(run: datetime, fxx: int) -> dict[str, np.ndarray]:
    """Fetch ONLY the ensemble-probability source fields (REFC/REFD/UH 0-3 & 2-5 km)
    in one small combined download. Used for lagged ensemble members, which
    contribute only the probability fields (not the deterministic state)."""
    H = _herbie(run, fxx, "sfc")
    search = "|".join(SFC_SEARCHES[k] for k in ("refc", "refd", "uh03", "uh25"))
    dss = _open(H, search)
    try:
        mapped = _map_surface(dss)
    finally:
        for ds in dss:
            ds.close()
    return {k: mapped[k] for k in ("refc", "refd", "uh03", "uh25") if k in mapped}


def surface_fields(run: datetime, fxx: int) -> dict[str, np.ndarray]:
    """Return FIGS-normalized 2-D surface fields (ny, nx).

    Keys: psfc (mb), zsfc (m), t2m, td2m (K), u10, v10 (m/s), refc, refd (dBZ),
    uh03, uh25 (m^2/s^2). Each is fetched by an exact idx search so cfgrib's
    multi-hypercube splitting and generic MXUPHL naming don't matter. Missing
    fields are simply omitted (callers handle defensively)."""
    H = _herbie(run, fxx, "sfc")
    out: dict[str, np.ndarray] = {}
    for key, search in SFC_SEARCHES.items():
        try:
            out[key] = _single_field(H, search)
        except Exception:  # noqa: BLE001 - missing field -> omit, handled downstream
            continue
    # soil fields are in the pressure product
    Hp = _herbie(run, fxx, "prs")
    for key, search in SOIL_SEARCHES.items():
        try:
            out[key] = _single_field(Hp, search)
        except Exception:  # noqa: BLE001
            continue
    if "psfc" in out and np.nanmedian(out["psfc"]) > 2000.0:  # Pa -> mb
        out["psfc"] = out["psfc"] / 100.0
    return out


def cache_summary() -> dict:
    """Quick summary of the local GRIB cache (file count, bytes)."""
    files = list(HRRR_CACHE.rglob("*.grib2"))
    return dict(
        cache_dir=str(HRRR_CACHE),
        n_files=len(files),
        bytes=sum(f.stat().st_size for f in files),
    )
