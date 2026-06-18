"""Storm-motion / reference vectors, computed grid-wide.

Five vectors (see ``config.MOTION_VECTORS``):
  * ``bunkers_rm`` / ``bunkers_lm`` : Bunkers (2000) right / left movers.
  * ``mean_0_6km``                  : 0–6 km AGL mean wind.
  * ``corfidi_up`` / ``corfidi_down``: Corfidi (2003) upshear / downshear MCS
                                       vectors (SHARPpy convention).

Each is returned as a pair of (ny, nx) u/v component grids (m/s). These feed the
hodograph rotation, spatial gradients, and storm-relative parameters.
"""

from __future__ import annotations

import numpy as np

from .profiles import Profiles, interp_to_heights

BUNKERS_D = 7.5  # m/s deviation magnitude


def _interp_to_pressures(prof: Profiles, pres_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate (u, v) to absolute pressures (mb), linear in log-p.
    ``pres_targets`` is (P,); returns two (P, ny, nx) arrays."""
    # coordinate increasing along axis 0: use -log(pres) (pressure decreases up)
    coord = -np.log(prof.pres)           # (K, ny, nx) increasing
    K, ny, nx = coord.shape
    N = ny * nx
    cf = coord.reshape(K, N)
    uf = prof.u.reshape(K, N)
    vf = prof.v.reshape(K, N)
    cols = np.arange(N)
    P = len(pres_targets)
    ou = np.empty((P, N))
    ov = np.empty((P, N))
    for p, pt in enumerate(pres_targets):
        t = -np.log(pt)
        idx = np.sum(cf < t, axis=0)
        idx_hi = np.clip(idx, 1, K - 1)
        idx_lo = idx_hi - 1
        c_lo = cf[idx_lo, cols]
        c_hi = cf[idx_hi, cols]
        w = np.where(c_hi > c_lo, (t - c_lo) / (c_hi - c_lo), 0.0)
        w = np.clip(w, 0.0, 1.0)
        ou[p] = uf[idx_lo, cols] + w * (uf[idx_hi, cols] - uf[idx_lo, cols])
        ov[p] = vf[idx_lo, cols] + w * (vf[idx_hi, cols] - vf[idx_lo, cols])
    return ou.reshape(P, ny, nx), ov.reshape(P, ny, nx)


def mean_wind_height(prof: Profiles, zbot: float, ztop: float, n: int = 11):
    """Layer-mean (u, v) over an AGL height layer (m), by sampling ``n`` levels."""
    heights = np.linspace(zbot, ztop, n)
    u, v = interp_to_heights(prof, heights)
    return u.mean(axis=0), v.mean(axis=0)


def mean_wind_pressure(prof: Profiles, pbot: float, ptop: float, n: int = 11):
    """Layer-mean (u, v) over a pressure layer (mb), by sampling ``n`` levels."""
    pres = np.linspace(pbot, ptop, n)
    u, v = _interp_to_pressures(prof, pres)
    return u.mean(axis=0), v.mean(axis=0)


def bunkers(prof: Profiles):
    """Return (rm, lm, mean06) where each is (u, v) of (ny, nx) grids.

    Bunkers (2000): mean = 0–6 km mean wind; shear = (5.5–6 km mean) - (0–0.5 km
    mean); RM = mean + D*(shear x k)/|shear|, LM = mean - D*(shear x k)/|shear|.
    """
    mu, mv = mean_wind_height(prof, 0.0, 6000.0)
    lu, lv = mean_wind_height(prof, 0.0, 500.0)
    hu, hv = mean_wind_height(prof, 5500.0, 6000.0)
    su, sv = hu - lu, hv - lv
    smag = np.sqrt(su**2 + sv**2)
    smag = np.where(smag < 1e-6, 1e-6, smag)
    # (shear x k) = (sv, -su): points to the right of the shear vector
    perp_u, perp_v = sv / smag, -su / smag
    rm = (mu + BUNKERS_D * perp_u, mv + BUNKERS_D * perp_v)
    lm = (mu - BUNKERS_D * perp_u, mv - BUNKERS_D * perp_v)
    return rm, lm, (mu, mv)


def corfidi(prof: Profiles):
    """Return (upshear, downshear), each (u, v) of (ny, nx) grids.

    SHARPpy convention: cloud-layer mean = pressure-weighted-ish 850–300 mb mean;
    LLJ = surface–1500 m mean wind; upshear = cloud - LLJ; downshear = 2*cloud - LLJ.
    """
    cu, cv = mean_wind_pressure(prof, 850.0, 300.0)
    ju, jv = mean_wind_height(prof, 0.0, 1500.0)
    up = (cu - ju, cv - jv)
    down = (cu + up[0], cv + up[1])
    return up, down


def all_motions(prof: Profiles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All five motion vectors keyed by ``config.MOTION_VECTORS`` names."""
    rm, lm, mean06 = bunkers(prof)
    up, down = corfidi(prof)
    return {
        "bunkers_rm": rm,
        "bunkers_lm": lm,
        "mean_0_6km": mean06,
        "corfidi_up": up,
        "corfidi_down": down,
    }
