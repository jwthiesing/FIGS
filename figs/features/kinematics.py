"""Kinematic fields, computed grid-wide.

These are the only nadocast "computed parameters" FIGS retains: divergence,
convergence (divergence with diverging regions zeroed), and absolute vorticity
at selected pressure levels, plus differential divergence (upper minus lower).

Horizontal derivatives use the FIGS projected grid spacing (~15 km). The grid is
regular in the Lambert projection so a constant dx/dy is a good approximation
across CONUS; map-factor variation is small at 15 km and neglected.
"""

from __future__ import annotations

import numpy as np

from ..config import DIFFERENTIAL_DIVERGENCE, KINEMATIC_LEVELS
from ..data import grid

OMEGA = 7.292e-5  # Earth angular velocity, rad/s


def _grid_spacing() -> tuple[float, float]:
    xc, yc = grid.figs_xy()
    return float(xc[1] - xc[0]), float(yc[1] - yc[0])


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """du/dx + dv/dy (1/s) on the FIGS grid; field shape (ny, nx)."""
    dx, dy = _grid_spacing()
    dudx = np.gradient(u, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    return dudx + dvdy


def relative_vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """dv/dx - du/dy (1/s)."""
    dx, dy = _grid_spacing()
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    return dvdx - dudy


def coriolis() -> np.ndarray:
    """Coriolis parameter f = 2Ω sin(lat) on the FIGS grid (ny, nx)."""
    lat, _ = grid.figs_latlon()
    return 2.0 * OMEGA * np.sin(np.radians(lat))


def level_fields(u: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """divergence, convergence (=max(-div,0)), and absolute vorticity for one level."""
    div = divergence(u, v)
    conv = np.maximum(-div, 0.0)
    absvort = relative_vorticity(u, v) + coriolis()
    return {"div": div, "conv": conv, "absvort": absvort}


def all_kinematics(level_winds: dict[float, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Kinematic fields for each level in ``KINEMATIC_LEVELS`` plus differential
    divergence between ``DIFFERENTIAL_DIVERGENCE`` levels.

    ``level_winds`` maps a pressure level (mb) -> (u, v) grids. Output keys are
    ``mb{level}_<field>`` and ``diffdiv_{upper}_{lower}``.
    """
    out: dict[str, np.ndarray] = {}
    div_by_level: dict[float, np.ndarray] = {}
    for lev in KINEMATIC_LEVELS:
        u, v = level_winds[lev]
        f = level_fields(u, v)
        div_by_level[lev] = f["div"]
        for k, val in f.items():
            out[f"mb{int(lev)}_{k}"] = val
    upper, lower = DIFFERENTIAL_DIVERGENCE
    if upper in div_by_level and lower in div_by_level:
        out[f"diffdiv_{int(upper)}_{int(lower)}"] = div_by_level[upper] - div_by_level[lower]
    return out
