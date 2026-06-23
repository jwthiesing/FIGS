"""Wildfire label construction for FIGS-W.

For a valid time, on the FIGS ~15 km grid (25 mi neighborhood), stamps every fire
active that hour (ongoing multi-day fires included) using its footprint as of that
hour. Labels store RAW values (binned later at train/CIG time):
  * ``wildfire``      1 if a fire is active in the neighborhood;
  * ``wildfire_size`` final size (acres) of the LARGEST nearby fire (NaN if none/unknown).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from figs.data import grid

from .. import config as C
from . import fire_reports


def _stamp(points, off, occ, size, *, fsize):
    """Stamp the 25 mi neighborhood around each (lat, lon) footprint point, taking the
    element-wise (NaN-aware) max of size."""
    pts = np.atleast_2d(points)
    xc, yc = grid.figs_xy()
    x, y = grid.lcc_forward(pts[:, 1], pts[:, 0])
    ix = np.round((x - xc[0]) / (xc[1] - xc[0])).astype(int)
    iy = np.round((y - yc[0]) / (yc[1] - yc[0])).astype(int)
    ok = (ix >= 0) & (ix < C.FIGS_NX) & (iy >= 0) & (iy < C.FIGS_NY)
    for j in np.where(ok)[0]:
        cy = iy[j] + off[:, 0]
        cx = ix[j] + off[:, 1]
        m = (cy >= 0) & (cy < C.FIGS_NY) & (cx >= 0) & (cx < C.FIGS_NX)
        cy, cx = cy[m], cx[m]
        occ[cy, cx] = 1
        if np.isfinite(fsize):
            size[cy, cx] = np.fmax(size[cy, cx], fsize)   # NaN-aware: known size wins


def build_labels(valid_time: datetime) -> dict[str, np.ndarray]:
    """Per-cell RAW wildfire label arrays for ``valid_time`` (FIGS_NY, FIGS_NX)."""
    shape = (C.FIGS_NY, C.FIGS_NX)
    occ = np.zeros(shape, dtype=np.int8)
    size = np.full(shape, np.nan, dtype=np.float32)        # acres; NaN where no fire

    off = grid.stencil(C.NEIGHBORHOOD_RADIUS_MI)
    for points, final_size in fire_reports.active_fires(valid_time):
        _stamp(points, off, occ, size, fsize=float(final_size))
    return {"wildfire": occ, "wildfire_size": size}
