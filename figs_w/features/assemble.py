"""Assemble the FIGS-W per-hour feature set.

Reuses the FIGS feature engine (profiles / thermo / SR params / rotated hodographs /
kinematics / spatial means+gradients) verbatim — those functions already take a
``motions`` dict and a ``Profiles`` object, so passing the 2-motion FIGS-W frame
set is all that's needed. On top of FIGS this:

  * drops the ensemble probability fields (none are passed in);
  * keeps the deterministic REFC/REFD/UH (Tier-1);
  * adds SOILW/TSOIL/cloud cover and surface T/Td/T−Td/wind-speed to Tier-1;
  * adds the static geography fields (terrain, terrain texture, land use,
    population density) to Tier-1.
"""

from __future__ import annotations

import re as _re

import numpy as np

from figs.features import assemble as FA      # reuse the private level/profile helpers
from figs.features import hodograph, kinematics, profiles, spatial, sr_params, thermo

from .. import config as C
from ..data import static as static_mod
from . import motions as motions_w


def compute_features(iso: dict, sfc: dict,
                     static: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Full single-hour FIGS-W feature dict (each value (ny, nx)). ``static`` is the
    cached geography field dict; loaded automatically if omitted."""
    if static is None:
        static = static_mod.load_static_fields()

    prof = profiles.build_profiles(iso, sfc)
    mvecs = motions_w.all_motions(prof)          # {'none', 'mean_wind'}

    feats: dict[str, np.ndarray] = {}

    for mname, (mu, mv) in mvecs.items():
        feats[f"motion_{mname}_u"] = mu
        feats[f"motion_{mname}_v"] = mv
        feats[f"motion_{mname}_spd"] = np.sqrt(mu**2 + mv**2)

    feats.update(thermo.all_thermo(prof))
    feats.update(sr_params.all_storm_relative(prof, mvecs))
    feats.update(hodograph.all_rotated_hodographs(prof, mvecs))
    feats.update(kinematics.all_kinematics(FA._level_winds(prof)))
    feats.update(FA._profile_point_features(prof))
    feats.update(FA._mandatory_hgt_features(iso))
    feats.update(FA._level_tmp_dpt_features(iso))

    # surface scalar point fields (incl. SOILW/TSOIL/cloud + deterministic REFC/REFD/UH)
    for key in C.SURFACE_POINT_FIELDS:
        if sfc.get(key) is not None:
            feats[key] = sfc[key]

    # raw surface state + rotated 10 m wind into each (of the 2) motion frames
    for key in ("t2m", "td2m", "u10", "v10"):
        if sfc.get(key) is not None:
            feats[f"sfc_{key}"] = sfc[key]
    if sfc.get("u10") is not None and sfc.get("v10") is not None:
        u10, v10 = sfc["u10"], sfc["v10"]
        feats["sfc_wspd10"] = np.sqrt(u10**2 + v10**2)
        for mname, mvec in mvecs.items():
            ur, vr = hodograph.rotate(u10, v10, mvec)
            feats[f"sfc_u_{mname}_rot"] = ur
            feats[f"sfc_v_{mname}_rot"] = vr
    if sfc.get("t2m") is not None and sfc.get("td2m") is not None:
        feats["sfc_tdspread2m"] = sfc["t2m"] - sfc["td2m"]

    # static geography (terrain + texture + land use + population density)
    for k in C.STATIC_FIELDS:
        if k in static:
            feats[k] = static[k]

    # ---- Tier 1: non-motion scalars → means + gradients vs BOTH motions --------
    tier1_names = (list(C.SPATIAL_BASE_FIELDS) + FA._tmp_dpt_spatial_names()
                   + list(C.STATIC_FIELDS))
    base_fields = {n: feats[n] for n in tier1_names if n in feats}
    feats.update(spatial.spatial_features(base_fields, mvecs))

    # ---- Tier 2: motion-relative scalars + rotated hodograph winds, own frame --
    sr_suffixes = tuple(f"_{c}" for c in C.SR_SCALARS)
    hodo_re = _re.compile(r"[uv]\d+$")

    def _tier2_match(key: str, mname: str) -> bool:
        if not key.startswith(f"{mname}_"):
            return False
        if key.endswith(sr_suffixes):
            return True
        return C.INCLUDE_HODOGRAPH and bool(hodo_re.fullmatch(key[len(mname) + 1:]))

    tier2_keys = [k for k in feats if any(_tier2_match(k, m) for m in mvecs)]
    tier2_spatial: dict[str, np.ndarray] = {}
    for mname, mvec in mvecs.items():
        group = {k: feats[k] for k in tier2_keys if _tier2_match(k, mname)}
        if group:
            tier2_spatial.update(spatial.spatial_features(group, {mname: mvec}))
    feats.update(tier2_spatial)

    return feats


def to_array(feats: dict[str, np.ndarray]):
    names = sorted(feats.keys())
    return names, np.stack([feats[n] for n in names], axis=0)
