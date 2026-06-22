"""Wildfire label construction for FIGS-W.

For a valid time, on the FIGS ~15 km grid (25 mi neighborhood, ±30 min of the
fire's discovery time):
  * ``wildfire``      : 1 if a new fire start falls in the cell's neighborhood;
  * ``wildfire_sig``  : 1 if a **deadly/destructive** fire does (fatalities>0 or
                        structures ≥ threshold) — the "deadliness" target;
  * ``wildfire_bin``  : conditional **size** bin (0–25 .. 1000+ ac) of the LARGEST
                        nearby fire (-1 if none) — the CIG/size target.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from figs.data import grid

from .. import config as C
from . import fire_reports


def size_bin(acres: float) -> int:
    """Final-size acreage → conditional size-bin index (-1 if non-positive)."""
    edges = C.INTENSITY_BINS["wildfire"]["edges"]
    if not np.isfinite(acres) or acres <= 0:
        return -1
    return int(np.searchsorted(edges, acres, side="right") - 1)


def _stamp(points, off, occ, sig, binl, *, deadly: bool, b: int):
    """Stamp the 25 mi neighborhood stencil around each (lat, lon) footprint point."""
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
        if deadly:
            sig[cy, cx] = 1
        if b >= 0:
            binl[cy, cx] = np.maximum(binl[cy, cx], b)


def build_labels(valid_time: datetime) -> dict[str, np.ndarray]:
    """Per-cell wildfire label arrays for ``valid_time`` (FIGS_NY, FIGS_NX).

    Stamps EVERY fire active at ``valid_time`` (ongoing multi-day fires included),
    using its footprint as of that hour (progression perimeter samples, else the
    incident point). Size = bin of the fire's FINAL size (the CIG target);
    deadliness = fatal/destructive flag — both taken as the neighborhood max/OR."""
    occ = np.zeros((C.FIGS_NY, C.FIGS_NX), dtype=np.int8)
    sig = np.zeros((C.FIGS_NY, C.FIGS_NX), dtype=np.int8)
    binl = np.full((C.FIGS_NY, C.FIGS_NX), -1, dtype=np.int8)

    off = grid.stencil(C.NEIGHBORHOOD_RADIUS_MI)
    for points, final_size, deadly in fire_reports.active_fires(valid_time):
        _stamp(points, off, occ, sig, binl, deadly=bool(deadly), b=size_bin(final_size))
    return {"wildfire": occ, "wildfire_sig": sig, "wildfire_bin": binl}
