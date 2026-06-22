"""Single-run HRRR state for FIGS-W (no time-lagged ensemble).

Wildfire prediction uses one deterministic HRRR cycle: the isobaric cube (regridded
to 15 km) plus surface fields. Unlike FIGS there are **no** ensemble member-
exceedance probability fields — but the deterministic REFC / REFD / UH are kept
(block-MAX, like FIGS's probability source) as ordinary fields.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from figs.data import grid, hrrr_store

# block-MAX (local maxima) vs block-MEAN surface fields when regridding to 15 km.
_MAX_FIELDS = ("refc", "refd", "uh03", "uh25")


def assemble_state(run: datetime, fxx: int, *, cached_only: bool = False) -> dict:
    """Return ``{'iso': <15 km isobaric cube>, 'sfc': <15 km surface fields>}`` for
    one HRRR cycle. ``cached_only`` reads from the local GRIB cache with no remote
    probe (see ``figs.data.hrrr_store``)."""
    iso = hrrr_store.isobaric_cube(run, fxx, regrid=grid.block_average, cached_only=cached_only)
    sfc_native = hrrr_store.surface_fields_combined(run, fxx, cached_only=cached_only)
    sfc: dict[str, np.ndarray] = {}
    for k, v in sfc_native.items():
        if v is None or getattr(v, "shape", None) != (hrrr_store.HRRR_GRID.ny, hrrr_store.HRRR_GRID.nx):
            continue
        regrid = grid.block_max if k in _MAX_FIELDS else grid.block_average
        sfc[k] = regrid(np.asarray(v, dtype=float)).astype(np.float32)
    return {"iso": iso, "sfc": sfc}
