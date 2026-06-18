"""Storm-relative parameters, computed grid-wide for each motion vector.

For each storm motion ``C = (cu, cv)`` and each AGL layer in ``config.SR_LAYERS``
(0–500 m, 500–1000 m, 1–3 km, 3–6 km) we compute:
  * ``srh``  : storm-relative helicity (m^2/s^2),
  * ``swv``  : layer-mean streamwise vorticity (1/s),
  * ``srw_*``: storm-relative mean wind components and speed (m/s).

Everything is integrated on a uniform 100 m AGL grid (0–6 km) so the layer means
and helicity integrals are consistent and fully vectorized.
"""

from __future__ import annotations

import numpy as np

from ..config import SR_LAYERS
from .profiles import Profiles, interp_to_heights

DZ = 100.0  # m integration step
FINE_HEIGHTS = np.arange(0.0, 6000.0 + DZ, DZ)


def _fine_winds(prof: Profiles) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) interpolated to the uniform 0–6 km, 100 m grid: each (H, ny, nx)."""
    return interp_to_heights(prof, FINE_HEIGHTS)


def storm_relative(prof: Profiles, motion: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-layer SR parameters for a single motion vector.

    Returns a flat dict keyed ``<layer>_<param>`` where param ∈
    {srh, swv, srw_u, srw_v, srw_spd}.
    """
    cu, cv = motion
    u, v = _fine_winds(prof)                    # (H, ny, nx)
    H = u.shape[0]
    z = FINE_HEIGHTS

    # storm-relative wind on the fine grid
    usr = u - cu[None]
    vsr = v - cv[None]

    # vertical shear (du/dz, dv/dz) via central differences on the uniform grid
    dudz = np.gradient(u, DZ, axis=0)
    dvdz = np.gradient(v, DZ, axis=0)

    # horizontal vorticity omega = (-dv/dz, du/dz); helicity density = vsr . omega
    hel_density = usr * (-dvdz) + vsr * dudz    # (H, ny, nx)  [1/s * m/s]
    # streamwise vorticity = (vsr . omega) / |vsr|
    sr_speed = np.sqrt(usr**2 + vsr**2)
    swv_density = np.divide(hel_density, sr_speed, out=np.zeros_like(hel_density),
                            where=sr_speed > 1e-3)
    # streamwise fraction = streamwise / total vorticity = omega_s / |omega|,
    # where |omega| = sqrt((du/dz)^2 + (dv/dz)^2) is the total horizontal
    # (streamwise + crosswise) vorticity magnitude. Ranges -1..1.
    omega_mag = np.sqrt(dudz**2 + dvdz**2)
    swfrac_density = np.divide(swv_density, omega_mag, out=np.zeros_like(swv_density),
                               where=omega_mag > 1e-6)

    out: dict[str, np.ndarray] = {}
    for name, zb, zt in SR_LAYERS:
        lo = int(round(zb / DZ))
        hi = int(round(zt / DZ))
        sl = slice(lo, hi + 1)
        # SRH: integrate helicity density over the layer (trapezoid)
        out[f"{name}_srh"] = np.trapezoid(hel_density[sl], dx=DZ, axis=0)
        # layer-mean streamwise vorticity
        out[f"{name}_swv"] = swv_density[sl].mean(axis=0)
        # layer-mean streamwise fraction (out of total vorticity)
        out[f"{name}_swfrac"] = swfrac_density[sl].mean(axis=0)
        # storm-relative mean wind in the layer
        mu = usr[sl].mean(axis=0)
        mv = vsr[sl].mean(axis=0)
        out[f"{name}_srw_u"] = mu
        out[f"{name}_srw_v"] = mv
        out[f"{name}_srw_spd"] = np.sqrt(mu**2 + mv**2)
    return out


def all_storm_relative(
    prof: Profiles, motions: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    """SR parameters for every motion, keyed ``<motion>_<layer>_<param>``."""
    out: dict[str, np.ndarray] = {}
    for mname, mvec in motions.items():
        for k, val in storm_relative(prof, mvec).items():
            out[f"{mname}_{k}"] = val
    return out
