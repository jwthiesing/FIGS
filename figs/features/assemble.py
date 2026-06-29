"""Assemble the full single-hour feature set for one (ensemble-mean) HRRR field
state on the FIGS grid.

Inputs are already block-averaged to ~15 km and (at M4) ensemble-averaged across
the time-lagged members. This module computes the deterministic features:
fine TMP/DPT profiles, the five motion vectors, thermodynamics, per-motion
storm-relative parameters and rotated hodographs, kinematic fields, and the
spatial means + motion-relative gradients of a curated base-field set.

Ensemble *probability* fields (reflectivity / UH) are added separately by the
ensemble layer; temporal (t-1, t, t+1) stacking is done by the dataset builder.
"""

from __future__ import annotations

import numpy as np

from ..config import (
    KINEMATIC_LEVELS,
    MANDATORY_HGT_LEVELS,
    STATIC_FIELDS,
    SURFACE_POINT_FIELDS,
)
from . import (
    boundaries,
    hodograph,
    kinematics,
    lapse,
    motions,
    profiles,
    spatial,
    sr_params,
    thermo,
)


def _static_terrain_features() -> dict[str, np.ndarray]:
    """Static terrain fields (elevation, slope, x/y gradients, aspect, texture) on the
    FIGS grid — constant across valid times, built/cached once. Empty if unavailable."""
    try:
        from ..data.static import load_terrain_fields

        return {k: np.asarray(v, dtype=np.float32) for k, v in load_terrain_fields().items()}
    except Exception as e:  # noqa: BLE001 - terrain optional (deps/cache); features still build
        import sys

        print(f"[warn] terrain static fields unavailable ({str(e)[:80]}); skipping",
              file=sys.stderr)
        return {}


def _level_winds(prof: profiles.Profiles) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Interpolate (u, v) to the kinematic pressure levels."""
    levels = np.array(KINEMATIC_LEVELS, dtype=float)
    u, v = motions._interp_to_pressures(prof, levels)
    return {lev: (u[i], v[i]) for i, lev in enumerate(KINEMATIC_LEVELS)}


def _profile_point_features(prof: profiles.Profiles) -> dict[str, np.ndarray]:
    """Fine TMP/DPT (and T-Td spread) at each profile depth as point features."""
    from ..config import PROFILE_DEPTHS

    out: dict[str, np.ndarray] = {}
    has_vvel = "vvel" in prof.extra
    for k, depth in enumerate(PROFILE_DEPTHS):
        d = int(depth)
        out[f"tmp_d{d}"] = prof.tmp[k]
        out[f"dpt_d{d}"] = prof.dpt[k]
        out[f"tdspread_d{d}"] = prof.tmp[k] - prof.dpt[k]
        if has_vvel:
            out[f"vvel_d{d}"] = prof.extra["vvel"][k]
    return out


def _mandatory_hgt_features(iso: dict) -> dict[str, np.ndarray]:
    """Geopotential height at mandatory pressure levels (nadocast convention)."""
    levels = np.asarray(iso["levels"], dtype=float)
    targets = np.array(MANDATORY_HGT_LEVELS, dtype=float)
    ny, nx = iso["hgt"].shape[1:]
    tp = np.broadcast_to(targets[:, None, None], (len(targets), ny, nx))
    hgt = profiles.interp_logp(levels, iso["hgt"], tp)
    return {f"hgt_{int(lev)}mb": hgt[i] for i, lev in enumerate(MANDATORY_HGT_LEVELS)}


def _level_tmp_dpt_features(iso: dict) -> dict[str, np.ndarray]:
    """TMP/DPT (and T-Td spread) interpolated to ``TMP_DPT_SPATIAL_LEVELS`` as
    point features. These are added to the tier-1 spatial base set so they get
    means + all-motion gradients (thermal/moisture fields, not motion-tied)."""
    from ..config import TMP_DPT_SPATIAL_LEVELS

    levels = np.asarray(iso["levels"], dtype=float)
    targets = np.array(TMP_DPT_SPATIAL_LEVELS, dtype=float)
    ny, nx = iso["tmp"].shape[1:]
    tp = np.broadcast_to(targets[:, None, None], (len(targets), ny, nx))
    tmp = profiles.interp_logp(levels, iso["tmp"], tp)
    dpt = profiles.interp_logp(levels, iso["dpt"], tp)
    out: dict[str, np.ndarray] = {}
    for i, lev in enumerate(TMP_DPT_SPATIAL_LEVELS):
        L = int(lev)
        out[f"tmp_{L}mb"] = tmp[i]
        out[f"dpt_{L}mb"] = dpt[i]
        out[f"tdspread_{L}mb"] = tmp[i] - dpt[i]
    return out


def _tmp_dpt_spatial_names() -> list[str]:
    """Tier-1 spatial base names contributed by ``_level_tmp_dpt_features``
    (T and Td at each level; the T-Td spread stays a raw point feature)."""
    from ..config import TMP_DPT_SPATIAL_LEVELS

    return [f"{p}_{int(lev)}mb" for lev in TMP_DPT_SPATIAL_LEVELS for p in ("tmp", "dpt")]


def compute_features(iso: dict, sfc: dict,
                     prob_fields: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Full single-hour feature dict (each value (ny, nx)).

    ``prob_fields`` are the ensemble member-exceedance probability fields
    (REFC/REFD/UH); when given they are merged in AND given the full tier-1
    spatial treatment (means + all-motion gradients) — they're non-motion
    scalar probabilities, so smoothed neighborhood/gradient versions help just
    like the environmental scalars."""
    prof = profiles.build_profiles(iso, sfc)
    mvecs = motions.all_motions(prof)

    feats: dict[str, np.ndarray] = {}

    # motion vector components themselves
    for mname, (mu, mv) in mvecs.items():
        feats[f"motion_{mname}_u"] = mu
        feats[f"motion_{mname}_v"] = mv
        feats[f"motion_{mname}_spd"] = np.sqrt(mu**2 + mv**2)

    # thermodynamics
    thermo_f = thermo.all_thermo(prof)
    feats.update(thermo_f)

    # storm-relative params (all motions x layers)
    sr_f = sr_params.all_storm_relative(prof, mvecs)
    feats.update(sr_f)

    # rotated hodographs (all motions x heights)
    feats.update(hodograph.all_rotated_hodographs(prof, mvecs))

    # kinematic fields
    kin_f = kinematics.all_kinematics(_level_winds(prof))
    feats.update(kin_f)

    # fine profile point features (TMP/DPT/spread/VVEL)
    feats.update(_profile_point_features(prof))

    # lapse rates (SR layers + 6-9 km) and surface-boundary gradients — both are
    # non-motion scalars given the Tier-1 spatial treatment below.
    feats.update(lapse.lapse_rate_features(prof))
    feats.update(boundaries.boundary_features(sfc))

    # geopotential height at mandatory levels
    feats.update(_mandatory_hgt_features(iso))

    # TMP/DPT at the levels we already carry (point features; spatially smoothed below)
    feats.update(_level_tmp_dpt_features(iso))

    # surface scalar point features (native CAPE/CIN, soil, PWAT, clouds, categorical precip)
    for key in SURFACE_POINT_FIELDS:
        if sfc.get(key) is not None:
            feats[key] = sfc[key]

    # raw surface state as direct features: 2 m temp/dewpoint, 10 m u/v + speed
    for key in ("t2m", "td2m", "u10", "v10"):
        if sfc.get(key) is not None:
            feats[f"sfc_{key}"] = sfc[key]
    if sfc.get("u10") is not None and sfc.get("v10") is not None:
        u10, v10 = sfc["u10"], sfc["v10"]
        feats["sfc_wspd10"] = np.sqrt(u10**2 + v10**2)
        # surface wind rotated into each storm-motion frame (rotationally invariant)
        for mname, mvec in mvecs.items():
            ur, vr = hodograph.rotate(u10, v10, mvec)
            feats[f"sfc_u_{mname}_rot"] = ur
            feats[f"sfc_v_{mname}_rot"] = vr
    if sfc.get("t2m") is not None and sfc.get("td2m") is not None:
        feats["sfc_tdspread2m"] = sfc["t2m"] - sfc["td2m"]

    # --- spatial smoothing (two tiers; see config.SPATIAL_BASE_FIELDS) -------- #
    from ..config import SPATIAL_BASE_FIELDS, SPATIAL_SR_SCALARS

    # ensemble probability fields (REFC/REFD/UH member-exceedance) as point
    # features; their names join the tier-1 set below for the full treatment.
    prob_names: list[str] = []
    if prob_fields:
        feats.update(prob_fields)
        prob_names = list(prob_fields.keys())

    # static terrain (elevation/slope/x-y gradients/aspect/texture) as point features;
    # join the tier-1 set for means + all-motion gradients (e.g. up/downslope vs motion).
    terrain = _static_terrain_features()
    feats.update(terrain)
    terrain_names = [k for k in STATIC_FIELDS if k in terrain]

    # Tier 1: non-motion environmental scalars -> means + gradients vs ALL motions.
    # (incl. T/Td at the levels we carry, lapse rates, surface-boundary gradients,
    # the ensemble probability fields, and the static terrain fields).
    tier1_names = (list(SPATIAL_BASE_FIELDS) + _tmp_dpt_spatial_names()
                   + lapse.lapse_feature_names() + boundaries.boundary_feature_names()
                   + prob_names + terrain_names)
    base_fields = {name: feats[name] for name in tier1_names if name in feats}
    feats.update(spatial.spatial_features(base_fields, mvecs))

    # Tier 2: motion-relative fields -> means + gradients in OWN frame only
    # (a cross-frame gradient of an already-rotated quantity is meaningless).
    # Two families, both keyed by their motion prefix:
    #   * SR scalars   '{motion}_{layer}_{component}'  (component in SPATIAL_SR_SCALARS)
    #   * rotated hodograph winds '{motion}_{u|v}{height}'  (SPATIAL_INCLUDE_HODOGRAPH)
    # The SR *vector* components (srw_u/srw_v) remain raw, unsmoothed.
    import re as _re
    from ..config import SPATIAL_INCLUDE_HODOGRAPH

    sr_suffixes = tuple(f"_{c}" for c in SPATIAL_SR_SCALARS)
    hodo_re = _re.compile(r"[uv]\d+$")

    def _tier2_match(key: str, mname: str) -> bool:
        if not key.startswith(f"{mname}_"):
            return False
        if key.endswith(sr_suffixes):
            return True
        return SPATIAL_INCLUDE_HODOGRAPH and bool(hodo_re.fullmatch(key[len(mname) + 1:]))

    tier2_keys = [k for k in feats                       # snapshot before mutating
                  if any(_tier2_match(k, m) for m in mvecs)]
    tier2_spatial: dict[str, np.ndarray] = {}
    for mname, mvec in mvecs.items():
        group = {k: feats[k] for k in tier2_keys if _tier2_match(k, mname)}
        if group:
            tier2_spatial.update(spatial.spatial_features(group, {mname: mvec}))
    feats.update(tier2_spatial)

    return feats


def added_features(iso: dict, sfc: dict) -> dict[str, np.ndarray]:
    """Compute ONLY the newer feature families (lapse rates + surface-boundary
    gradients + static terrain) AND their Tier-1 spatial expansion (means +
    all-motion gradients).

    This mirrors exactly what ``compute_features`` adds for these families, but
    skips the ~5,000 existing features — so an existing parquet can be augmented in
    place (recomputed from the cached GRIB) without re-deriving everything. Returns
    a dict of (ny, nx) grids."""
    prof = profiles.build_profiles(iso, sfc)
    mvecs = motions.all_motions(prof)

    pts: dict[str, np.ndarray] = {}
    pts.update(lapse.lapse_rate_features(prof))
    pts.update(boundaries.boundary_features(sfc))
    terrain = _static_terrain_features()
    pts.update(terrain)

    out = dict(pts)
    # Tier-1 spatial treatment for the lapse + boundary + terrain scalars (same as in
    # compute_features: means at every radius + all-motion directional gradients).
    base = {k: pts[k] for k in (lapse.lapse_feature_names()
                                + boundaries.boundary_feature_names()
                                + [t for t in STATIC_FIELDS if t in terrain]) if k in pts}
    out.update(spatial.spatial_features(base, mvecs))
    return out


def to_array(feats: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Stack a feature dict into (names, array of shape (F, ny, nx))."""
    names = sorted(feats.keys())
    arr = np.stack([feats[n] for n in names], axis=0)
    return names, arr


def feature_count(feats: dict[str, np.ndarray]) -> int:
    return len(feats)
