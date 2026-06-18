"""Vectorized thermodynamics, computed grid-wide.

Implements surface-based CAPE/CIN, mixed-layer CAPE/CIN over the lowest 90 mb and
180 mb, and DCAPE, all as NumPy operations over the (ny, nx) grid (the only loops
are over the dense vertical pressure grid). sounderpy/MetPy are far too slow
per-column for full HRRR grids, so the parcel ascent is reimplemented here with
standard formulas (Bolton 1980 for saturation vapor pressure and the LCL;
pseudoadiabatic lapse for the moist ascent).
"""

from __future__ import annotations

import numpy as np

from .profiles import Profiles, interp_logp

# constants (SI)
RD = 287.04
RV = 461.5
CP = 1005.7
LV = 2.501e6
EPS = RD / RV
G = 9.81

# dense pressure grid step for parcel integration (mb)
DP = 5.0

# standard mixed-layer depth (mb) used to define the ML parcel
ML_DEPTH = 100.0
# partial-depth bands (mb above ground) over which to additionally report
# CAPE/CIN for the SB and ML parcels
PARTIAL_DEPTHS = (90.0, 180.0)


def sat_vapor_pressure(T: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (hPa) from temperature (K), Bolton (1980)."""
    Tc = T - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def mixing_ratio(e: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mixing ratio (kg/kg) from vapor pressure and total pressure (same units)."""
    return EPS * e / np.maximum(p - e, 1e-6)


def sat_mixing_ratio(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return mixing_ratio(sat_vapor_pressure(T), p)


def virtual_temperature(T: np.ndarray, w: np.ndarray) -> np.ndarray:
    return T * (1.0 + 0.608 * w)


def lcl(p0: np.ndarray, T0: np.ndarray, Td0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LCL pressure (mb) and temperature (K) from parcel p/T/Td (Bolton 1980)."""
    Tlcl = 56.0 + 1.0 / (1.0 / (Td0 - 56.0) + np.log(T0 / Td0) / 800.0)
    plcl = p0 * (Tlcl / T0) ** (CP / RD)
    return plcl, Tlcl


def _dense_grid(psfc: np.ndarray, ptop: float = 100.0) -> np.ndarray:
    """Dense pressure levels (P, ny, nx), descending from max(psfc) to ptop.

    A single shared 1-D pressure axis is broadcast over the grid; cells whose
    surface pressure is below a given level are masked later via the parcel
    being "underground" (handled by callers using psfc)."""
    pmax = float(np.nanmax(psfc))
    levs = np.arange(pmax, ptop - DP, -DP)
    return levs


def _parcel_ascent(p_levels, Tenv, Tdenv, p0, T0, Td0):
    """Lift a parcel (p0,T0,Td0) along descending ``p_levels`` (1-D, mb).

    Returns (Tv_parcel, Tv_env, valid) each (P, ny, nx); ``valid`` marks levels
    at or above the parcel origin (p <= p0).
    """
    P = len(p_levels)
    ny, nx = T0.shape
    w0 = mixing_ratio(sat_vapor_pressure(Td0), p0)        # parcel mixing ratio
    plcl, Tlcl = lcl(p0, T0, Td0)

    Tvp = np.full((P, ny, nx), np.nan)
    Tve = np.full((P, ny, nx), np.nan)

    Tprev = None
    pprev = None
    for i, p in enumerate(p_levels):
        above_origin = p <= p0
        below_lcl = p >= plcl                              # still unsaturated
        # dry adiabatic temperature
        Tdry = T0 * (p / p0) ** (RD / CP)
        if Tprev is None:
            Tcur = Tdry.copy()
        else:
            # moist pseudoadiabatic step from previous level when saturated
            ws = sat_mixing_ratio(Tprev, pprev)
            dTdp = (1.0 / pprev) * (RD * Tprev + LV * ws) / (
                CP + LV**2 * ws * EPS / (RD * Tprev**2)
            )
            Tmoist = Tprev + dTdp * (p - pprev)
            Tcur = np.where(below_lcl, Tdry, Tmoist)
        # parcel water vapor: w0 if unsaturated, else saturation
        wp = np.where(below_lcl, w0, sat_mixing_ratio(Tcur, p))
        Tvp[i] = np.where(above_origin, virtual_temperature(Tcur, wp), np.nan)
        # environment Tv at this level
        Tenv_i = Tenv[i]
        wenv = mixing_ratio(sat_vapor_pressure(Tdenv[i]), p)
        Tve[i] = np.where(above_origin, virtual_temperature(Tenv_i, wenv), np.nan)
        Tprev, pprev = Tcur, p

    valid = (p_levels[:, None, None] <= p0[None])
    return Tvp, Tve, valid


def _cape_cin(p_levels, Tvp, Tve, valid):
    """Integrate CAPE/CIN (J/kg) from parcel/env virtual temps over the column."""
    lnp = np.log(p_levels)
    dlnp = np.zeros_like(lnp)
    dlnp[1:] = lnp[:-1] - lnp[1:]                          # >0 going up
    b = RD * (Tvp - Tve)                                   # buoyancy (J/kg per dlnp)
    contrib = b * dlnp[:, None, None]
    contrib = np.where(valid & np.isfinite(contrib), contrib, 0.0)
    pos = np.clip(contrib, 0.0, None)
    # cumulative buoyancy to locate the LFC (first level where net turns positive)
    cum = np.cumsum(contrib, axis=0)
    reached = cum > 0
    lfc_idx = np.argmax(reached, axis=0)                   # first True (0 if none)
    has_cape = reached.any(axis=0)
    P = len(p_levels)
    idx = np.arange(P)[:, None, None]
    above_lfc = idx >= lfc_idx[None]
    cape = np.where(has_cape, np.sum(np.where(above_lfc, pos, 0.0), axis=0), 0.0)
    neg = np.clip(contrib, None, 0.0)
    below_lfc = idx < lfc_idx[None]
    cin = np.where(has_cape, np.sum(np.where(below_lfc, neg, 0.0), axis=0), 0.0)
    return cape, cin


def _band_cape_cin(p_levels, Tvp, Tve, valid, psfc, depth):
    """CAPE/CIN accumulated only within the lowest ``depth`` mb above ground.

    Unlike full CAPE/CIN (which use the LFC), the partial-band version simply sums
    positive buoyancy (-> CAPE) and negative buoyancy (-> CIN) for levels with
    p in [psfc-depth, psfc]. This gives the low-level CAPE/CIN within that depth.
    """
    lnp = np.log(p_levels)
    dlnp = np.zeros_like(lnp)
    dlnp[1:] = lnp[:-1] - lnp[1:]
    contrib = RD * (Tvp - Tve) * dlnp[:, None, None]
    band = (p_levels[:, None, None] <= psfc[None]) & (
        p_levels[:, None, None] >= (psfc[None] - depth)
    )
    contrib = np.where(valid & band & np.isfinite(contrib), contrib, 0.0)
    cape = np.clip(contrib, 0.0, None).sum(axis=0)
    cin = np.clip(contrib, None, 0.0).sum(axis=0)
    return cape, cin


def _env_on_grid(prof: Profiles, p_levels: np.ndarray):
    """Environment T and Td interpolated to the dense pressure grid (P, ny, nx).

    prof pressures vary by cell, so we interpolate per cell in log-pressure."""
    src_p = prof.pres                                  # (K, ny, nx)
    K, ny, nx = src_p.shape
    target = np.broadcast_to(p_levels[:, None, None], (len(p_levels), ny, nx))
    Tenv = _interp_per_cell_logp(src_p, prof.tmp, target)
    Tdenv = _interp_per_cell_logp(src_p, prof.dpt, target)
    return Tenv, Tdenv


def _interp_per_cell_logp(src_p, field, target_p):
    """Per-cell log-p interpolation where source pressures vary by cell.
    src_p, field: (K, ny, nx) with src_p decreasing along axis 0. target_p:
    (P, ny, nx). Returns (P, ny, nx)."""
    K, ny, nx = src_p.shape
    N = ny * nx
    sp = src_p.reshape(K, N)
    lf = field.reshape(K, N)
    coord = -np.log(sp)                                # increasing along axis 0
    cols = np.arange(N)
    P = target_p.shape[0]
    out = np.empty((P, N))
    tt = -np.log(target_p.reshape(P, N))
    for k in range(P):
        t = tt[k]
        idx = (coord < t).sum(axis=0)
        idx_hi = np.clip(idx, 1, K - 1)
        idx_lo = idx_hi - 1
        c_lo = coord[idx_lo, cols]
        c_hi = coord[idx_hi, cols]
        w = np.where(c_hi > c_lo, (t - c_lo) / (c_hi - c_lo), 0.0)
        w = np.clip(w, 0.0, 1.0)
        out[k] = lf[idx_lo, cols] + w * (lf[idx_hi, cols] - lf[idx_lo, cols])
    return out.reshape(P, ny, nx)


def surface_based(prof: Profiles) -> dict[str, np.ndarray]:
    """Surface-based CAPE/CIN: full (to the EL) plus the partial CAPE/CIN within
    the lowest 90 / 180 mb above ground."""
    p_levels = _dense_grid(prof.pres[0])
    Tenv, Tdenv = _env_on_grid(prof, p_levels)
    psfc = prof.pres[0]
    T0 = prof.tmp[0]
    Td0 = np.minimum(prof.dpt[0], T0)
    Tvp, Tve, valid = _parcel_ascent(p_levels, Tenv, Tdenv, psfc, T0, Td0)
    cape, cin = _cape_cin(p_levels, Tvp, Tve, valid)
    out = {"sbcape": cape, "sbcin": cin}
    for d in PARTIAL_DEPTHS:
        c, ci = _band_cape_cin(p_levels, Tvp, Tve, valid, psfc, d)
        out[f"sbcape_0_{int(d)}"] = c
        out[f"sbcin_0_{int(d)}"] = ci
    return out


def mixed_layer(prof: Profiles, ml_depth: float = ML_DEPTH) -> dict[str, np.ndarray]:
    """Mixed-layer CAPE/CIN using a standard ``ml_depth`` (mb) mixed parcel: full
    (to the EL) plus the partial CAPE/CIN within the lowest 90 / 180 mb."""
    p_levels = _dense_grid(prof.pres[0])
    Tenv, Tdenv = _env_on_grid(prof, p_levels)
    psfc = prof.pres[0]
    # average potential temperature and mixing ratio over the standard mixed layer
    in_layer = (p_levels[:, None, None] <= psfc[None]) & (
        p_levels[:, None, None] >= (psfc[None] - ml_depth)
    )
    theta = Tenv * (1000.0 / p_levels[:, None, None]) ** (RD / CP)
    w = mixing_ratio(sat_vapor_pressure(Tdenv), p_levels[:, None, None])
    cnt = np.where(in_layer.sum(axis=0) < 1, 1, in_layer.sum(axis=0))
    theta_ml = np.sum(np.where(in_layer, theta, 0.0), axis=0) / cnt
    w_ml = np.sum(np.where(in_layer, w, 0.0), axis=0) / cnt
    # lift the ML parcel from the surface
    p0 = psfc
    T0 = theta_ml * (p0 / 1000.0) ** (RD / CP)
    e0 = w_ml * p0 / (EPS + w_ml)
    Td0 = 243.5 / (17.67 / np.log(np.maximum(e0, 1e-6) / 6.112) - 1.0) + 273.15
    Td0 = np.minimum(Td0, T0)
    Tvp, Tve, valid = _parcel_ascent(p_levels, Tenv, Tdenv, p0, T0, Td0)
    cape, cin = _cape_cin(p_levels, Tvp, Tve, valid)
    out = {"mlcape": cape, "mlcin": cin}
    for d in PARTIAL_DEPTHS:
        c, ci = _band_cape_cin(p_levels, Tvp, Tve, valid, psfc, d)
        out[f"mlcape_0_{int(d)}"] = c
        out[f"mlcin_0_{int(d)}"] = ci
    return out


def wetbulb_from_thetae(thetae_target: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Invert saturated θe to a wet-bulb temperature (K) at pressure ``p`` (mb).
    θe_s increases monotonically with T, so a fixed-iteration bisection works
    grid-wide."""
    lo = np.full_like(thetae_target, 200.0)
    hi = np.full_like(thetae_target, 320.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ws = sat_mixing_ratio(mid, p)
        theta = mid * (1000.0 / p) ** (RD / CP)
        te = theta * np.exp(LV * ws / (CP * mid))
        too_high = te > thetae_target
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def dcape(prof: Profiles) -> dict[str, np.ndarray]:
    """Downdraft CAPE (J/kg): saturated descent from the minimum-θe level
    (searched between the surface and ~400 mb above it) to the surface."""
    p_levels = _dense_grid(prof.pres[0])
    Tenv, Tdenv = _env_on_grid(prof, p_levels)
    psfc = prof.pres[0]
    # equivalent potential temperature (approx, Bolton) on the dense grid
    e = sat_vapor_pressure(Tdenv)
    w = mixing_ratio(e, p_levels[:, None, None])
    theta = Tenv * (1000.0 / p_levels[:, None, None]) ** (RD / CP)
    thetae = theta * np.exp(LV * w / (CP * Tenv))
    search = (p_levels[:, None, None] <= psfc[None]) & (
        p_levels[:, None, None] >= (psfc[None] - 400.0)
    )
    thetae_masked = np.where(search, thetae, np.inf)
    src = np.argmin(thetae_masked, axis=0)              # min-θe level index
    P, ny, nx = Tenv.shape
    # descend saturated parcel from src to surface; integrate (Tve - Tvp)
    # parcel starts saturated at the env wet-bulb-ish temp -> use env T,Td at src
    src_thetae = np.take_along_axis(thetae, src[None], axis=0)[0]
    src_p = p_levels[src]
    # parcel starts saturated at the source: seed with its wet-bulb temperature
    src_T = wetbulb_from_thetae(src_thetae, src_p)
    # Descend a saturated parcel from the min-θe level to the surface. Iterate
    # from the top down (pressure increasing); each cell's parcel is "born" at
    # src_T when the level first reaches src_p, then follows the moist adiabat
    # down to the surface.
    Tvp = np.full((P, ny, nx), np.nan)
    Tcur = np.full((ny, nx), np.nan)
    pprev = np.full((ny, nx), np.nan)
    for i in range(P - 1, -1, -1):                      # top -> surface (p increasing)
        p = p_levels[i]
        region = (p >= src_p) & (p <= psfc)             # src down to surface
        newly = region & ~np.isfinite(Tcur)
        ws = sat_mixing_ratio(Tcur, pprev)
        dTdp = (1.0 / pprev) * (RD * Tcur + LV * ws) / (
            CP + LV**2 * ws * EPS / (RD * Tcur**2)
        )
        Tstep = Tcur + dTdp * (p - pprev)
        Tcur = np.where(newly, src_T, np.where(region, Tstep, Tcur))
        pprev = np.where(region, p, pprev)
        wp = sat_mixing_ratio(Tcur, p)
        Tvp[i] = np.where(region, virtual_temperature(Tcur, wp), np.nan)
    wenv = mixing_ratio(sat_vapor_pressure(Tdenv), p_levels[:, None, None])
    Tve = virtual_temperature(Tenv, wenv)
    lnp = np.log(p_levels)
    dlnp = np.zeros_like(lnp)
    dlnp[1:] = lnp[:-1] - lnp[1:]
    below_src_full = (p_levels[:, None, None] >= src_p[None]) & (
        p_levels[:, None, None] <= psfc[None]
    )
    integrand = RD * (Tve - Tvp) * dlnp[:, None, None]
    integrand = np.where(below_src_full & np.isfinite(integrand), integrand, 0.0)
    dcape_val = np.clip(integrand, 0.0, None).sum(axis=0)
    return {"dcape": dcape_val}


def all_thermo(prof: Profiles) -> dict[str, np.ndarray]:
    """All thermodynamic features: surface-based and standard mixed-layer CAPE/CIN
    (full + lowest-90/180 mb partial), and DCAPE."""
    out: dict[str, np.ndarray] = {}
    out.update(surface_based(prof))   # sbcape/sbcin + sb{cape,cin}_0_{90,180}
    out.update(mixed_layer(prof))     # mlcape/mlcin + ml{cape,cin}_0_{90,180}
    out.update(dcape(prof))
    return out
