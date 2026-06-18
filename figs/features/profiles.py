"""Fine vertical profiles, computed grid-wide.

HRRR isobaric fields (fixed pressure levels) plus surface/near-surface fields are
interpolated onto a fine, surface-relative pressure set (``config.PROFILE_DEPTHS``:
every 10 mb in the lowest 100 mb, 25 mb to 250 mb, 50 mb above) so the model sees
the boundary layer in detail. The resulting profiles also carry height AGL (from
geopotential height minus surface height), which the storm-relative parameters use
to interpolate to specific AGL layers.

All interpolation is vectorized across the (ny, nx) grid; the only Python loop is
over the ~24 profile levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PROFILE_DEPTHS


def interp_logp(levels_mb: np.ndarray, field: np.ndarray, target_p: np.ndarray) -> np.ndarray:
    """Interpolate ``field`` (linear in log-pressure) to ``target_p``.

    Parameters
    ----------
    levels_mb : (L,) source isobaric levels, strictly monotonic (any order).
    field     : (L, ny, nx) values on those levels.
    target_p  : (K, ny, nx) target pressures (mb), per cell.

    Returns (K, ny, nx). Targets outside the source range are clamped to the
    nearest source level (no extrapolation).
    """
    L = levels_mb.shape[0]
    ny, nx = field.shape[1:]
    N = ny * nx
    # ascending log-pressure
    order = np.argsort(levels_mb)
    logP = np.log(levels_mb[order])                 # (L,) ascending
    F = field[order].reshape(L, N)                  # (L, N)
    cols = np.arange(N)
    out = np.empty((target_p.shape[0], N), dtype=float)
    logt = np.log(target_p.reshape(target_p.shape[0], N))
    for k in range(logt.shape[0]):
        t = logt[k]                                 # (N,)
        idx = np.searchsorted(logP, t)              # 0..L
        idx_hi = np.clip(idx, 1, L - 1)
        idx_lo = idx_hi - 1
        p_lo = logP[idx_lo]
        p_hi = logP[idx_hi]
        f_lo = F[idx_lo, cols]
        f_hi = F[idx_hi, cols]
        w = np.where(p_hi > p_lo, (t - p_lo) / (p_hi - p_lo), 0.0)
        w = np.clip(w, 0.0, 1.0)                     # clamp -> no extrapolation
        out[k] = f_lo + w * (f_hi - f_lo)
    return out.reshape(target_p.shape[0], ny, nx)


@dataclass
class Profiles:
    """Fine surface-relative profiles on the grid. All arrays (K, ny, nx) with
    K = len(PROFILE_DEPTHS); level 0 is the surface."""

    pres: np.ndarray        # mb
    hgt_agl: np.ndarray     # m above ground
    tmp: np.ndarray         # K
    dpt: np.ndarray         # K (dewpoint)
    u: np.ndarray           # m/s
    v: np.ndarray           # m/s
    extra: dict = None      # additional interpolated fields, e.g. {'vvel': (K,ny,nx)}

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    @property
    def nlev(self) -> int:
        return self.pres.shape[0]


def build_profiles(iso: dict, sfc: dict) -> Profiles:
    """Build fine profiles from an isobaric cube and normalized surface fields.

    Parameters
    ----------
    iso : dict with
        'levels' (L,) isobaric levels mb (any order),
        'tmp','dpt','ugrd','vgrd','hgt' each (L, ny, nx).
    sfc : dict with normalized surface fields (each (ny, nx)):
        'psfc' (mb), 'zsfc' (m), 't2m' (K), 'td2m' (K), 'u10' (m/s), 'v10' (m/s).
    """
    levels = np.asarray(iso["levels"], dtype=float)
    psfc = np.asarray(sfc["psfc"], dtype=float)
    zsfc = np.asarray(sfc["zsfc"], dtype=float)
    ny, nx = psfc.shape
    K = len(PROFILE_DEPTHS)

    # target pressures: psfc - depth (mb), clipped so we never go below ~50 mb
    depths = PROFILE_DEPTHS[:, None, None]
    target_p = np.maximum(psfc[None] - depths, 50.0)  # (K, ny, nx)

    tmp = interp_logp(levels, iso["tmp"], target_p)
    dpt = interp_logp(levels, iso["dpt"], target_p)
    u = interp_logp(levels, iso["ugrd"], target_p)
    v = interp_logp(levels, iso["vgrd"], target_p)
    hgt = interp_logp(levels, iso["hgt"], target_p)   # geopotential height (m MSL)

    # overwrite the surface level (depth 0) with the 2 m / 10 m fields
    tmp[0] = sfc["t2m"]
    dpt[0] = sfc["td2m"]
    u[0] = sfc["u10"]
    v[0] = sfc["v10"]
    hgt[0] = zsfc

    hgt_agl = hgt - zsfc[None]
    hgt_agl[0] = 0.0
    # enforce monotonic increasing height (guards tiny interpolation inversions)
    hgt_agl = np.maximum.accumulate(hgt_agl, axis=0)

    # additional isobaric fields (e.g. VVEL) interpolated to the fine levels
    extra: dict = {}
    for key in ("vvel",):
        if key in iso:
            extra[key] = interp_logp(levels, iso[key], target_p)

    return Profiles(pres=target_p, hgt_agl=hgt_agl, tmp=tmp, dpt=dpt, u=u, v=v, extra=extra)


def interp_to_heights(prof: Profiles, heights_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate (u, v) to specified AGL heights (m). ``heights_m`` is (H,);
    returns two (H, ny, nx) arrays. Linear in height; clamped at profile top."""
    z = prof.hgt_agl                       # (K, ny, nx), increasing
    K, ny, nx = z.shape
    N = ny * nx
    zf = z.reshape(K, N)
    uf = prof.u.reshape(K, N)
    vf = prof.v.reshape(K, N)
    cols = np.arange(N)
    H = len(heights_m)
    ou = np.empty((H, N))
    ov = np.empty((H, N))
    for h, zt in enumerate(heights_m):
        # z increases along axis 0, so searchsorted index per column is the
        # count of levels strictly below the target height.
        idx = np.sum(zf < zt, axis=0)
        idx_hi = np.clip(idx, 1, K - 1)
        idx_lo = idx_hi - 1
        z_lo = zf[idx_lo, cols]
        z_hi = zf[idx_hi, cols]
        w = np.where(z_hi > z_lo, (zt - z_lo) / (z_hi - z_lo), 0.0)
        w = np.clip(w, 0.0, 1.0)
        ou[h] = uf[idx_lo, cols] + w * (uf[idx_hi, cols] - uf[idx_lo, cols])
        ov[h] = vf[idx_lo, cols] + w * (vf[idx_hi, cols] - vf[idx_lo, cols])
    return ou.reshape(H, ny, nx), ov.reshape(H, ny, nx)
