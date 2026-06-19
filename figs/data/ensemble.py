"""Time-lagged HRRR ensemble assembly.

For a target valid time we gather the most-recent HRRR runs that forecast it
(``hrrr_store.recent_runs_for_valid``) and treat them as ensemble members. From
the members we build:

  * **ensemble-mean deterministic state** (isobaric cube + surface fields),
    regridded to ~15 km by block averaging — this feeds the feature engine;
  * **ensemble probability fields** — restricted to reflectivity (REFC/REFD) and
    updraft helicity (0–3 km and 2–5 km, used independently): the fraction of
    members whose block-MAX field exceeds each threshold.

cfgrib short-name resolution depends on the eccodes version, so surface-variable
mapping goes through ``normalize_surface`` with documented best-effort keys; it
is the single place to adjust if cfgrib names differ.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np

# Concurrent member downloads (network-bound). Override with FIGS_MEMBER_WORKERS.
MEMBER_FETCH_WORKERS = int(os.environ.get("FIGS_MEMBER_WORKERS", "6"))

from ..config import (
    ENSEMBLE_MAX_MEMBERS,
    REFC_THRESHOLDS_DBZ,
    REFD_THRESHOLDS_DBZ,
    SURFACE_POINT_FIELDS,
    UH_03KM_THRESHOLDS,
    UH_25KM_THRESHOLDS,
)
from . import grid, hrrr_store

# A member field must be the full native HRRR grid before block-averaging. A
# concurrent byte-range read can truncate a non-essential field to a partial/1-D
# array; essentials self-heal upstream, but here we simply skip a malformed
# optional field so one bad member doesn't crash the whole forecast hour.
_NATIVE_SHAPE = (grid.HRRR_GRID.ny, grid.HRRR_GRID.nx)


def _native(v) -> bool:
    return v is not None and getattr(v, "shape", None) == _NATIVE_SHAPE


def _member_state(run: datetime, fxx: int):
    """Fetch one member: regridded ensemble-input fields. Returns (iso15, sfc15,
    prob_src) where iso/sfc fields are block-mean to 15 km and probability source
    fields (reflectivity/UH) are block-MAX."""
    # regrid isobaric to 15 km during the read so the native cube is never fully
    # held (keeps per-member memory ~hundreds of MB instead of ~1 GB).
    iso15 = hrrr_store.isobaric_cube(run, fxx, regrid=grid.block_average)
    sfc = hrrr_store.surface_fields_combined(run, fxx)  # one download+open, FIGS-normalized

    sfc15 = {}
    det_sfc = ("psfc", "zsfc", "t2m", "td2m", "u10", "v10") + tuple(SURFACE_POINT_FIELDS)
    for k in det_sfc:
        if _native(sfc.get(k)):
            sfc15[k] = grid.block_average(sfc[k]).astype(np.float32)
    # probability source fields use block-MAX (local maxima)
    prob_src = {}
    for k in ("refc", "refd", "uh03", "uh25"):
        if _native(sfc.get(k)):
            prob_src[k] = grid.block_max(sfc[k]).astype(np.float32)
    del sfc  # free native surface fields promptly
    return iso15, sfc15, prob_src


def _probability_fields(prob_src_members: list[dict]) -> dict[str, np.ndarray]:
    """Member-exceedance fractions for reflectivity and UH (the only ensemble
    probability fields FIGS uses)."""
    spec = {
        "refc": REFC_THRESHOLDS_DBZ,
        "refd": REFD_THRESHOLDS_DBZ,
        "uh03": UH_03KM_THRESHOLDS,
        "uh25": UH_25KM_THRESHOLDS,
    }
    out: dict[str, np.ndarray] = {}
    for var, thresholds in spec.items():
        stacks = [m[var] for m in prob_src_members if var in m]
        if not stacks:
            continue
        arr = np.stack(stacks, axis=0)              # (M, ny, nx)
        for thr in thresholds:
            out[f"prob_{var}_ge{int(thr)}"] = np.nanmean(arr >= thr, axis=0)
    return out


def _member_prob_src(run: datetime, fxx: int) -> dict:
    """Lagged-member contribution: ONLY the block-MAX reflectivity/UH fields."""
    sfc = hrrr_store.surface_prob_fields(run, fxx)
    return {k: grid.block_max(v).astype(np.float32) for k, v in sfc.items() if _native(v)}


def assemble_inputs(valid_time: datetime, max_members: int = ENSEMBLE_MAX_MEMBERS,
                    as_of: datetime | None = None) -> dict:
    """Build the input state for ``valid_time``.

    The **deterministic** state (isobaric + surface, → all the engineered
    features) comes from the **main run** (the most-recent member). The remaining
    time-lagged members contribute **only** the reflectivity/UH probability
    fields, so their heavy 3-D isobaric data is never downloaded — a ~4-6x cut in
    download vs an ensemble-mean state.

    ``as_of`` (issuance time / primary run) caps which runs are available
    (real-time-faithful, mixing 18/48-h cycles, narrowing at far leads).

    Returns ``iso`` (main-run isobaric cube), ``sfc`` (main-run surface fields),
    ``prob_fields`` (reflectivity/UH member-exceedance fractions across all
    members), and ``members``."""
    members = hrrr_store.recent_runs_for_valid(valid_time, max_members, as_of=as_of)
    if not members:
        raise RuntimeError(f"no ensemble members for valid {valid_time}")
    main, lagged = members[0], members[1:]

    # main run: full deterministic state + its own reflectivity/UH
    iso15, sfc15, prob_main = _member_state(main.run, main.fxx)
    prob_srcs = {main: prob_main}

    # lagged members: probability fields only, fetched concurrently (small)
    if lagged:
        def _fetch(mem):
            return mem, _member_prob_src(mem.run, mem.fxx)

        with ThreadPoolExecutor(max_workers=min(len(lagged), MEMBER_FETCH_WORKERS)) as ex:
            for fut in as_completed([ex.submit(_fetch, m) for m in lagged]):
                try:
                    mem, ps = fut.result()
                    prob_srcs[mem] = ps
                except Exception as e:  # noqa: BLE001 - skip transient member failure
                    print(f"[warn] prob member skipped: {e}", flush=True)

    used = [m for m in members if m in prob_srcs]
    prob_fields = _probability_fields([prob_srcs[m] for m in used])
    return {"iso": iso15, "sfc": sfc15, "prob_fields": prob_fields, "members": used}
