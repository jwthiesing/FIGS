"""Environmental temperature lapse-rate features (Tier-1 non-motion scalars).

The lapse rate −dT/dz over the storm-relative AGL layers (the SAME layers as the
SR winds) plus an upper 6–9 km layer, in K/km (positive = temperature decreasing
with height; steeper = more unstable / more conducive to strong updrafts and large
hail aloft). Computed from the fine profile's temperature on its height-AGL grid —
no new HRRR fields are read, so this is cache-safe.

These are non-motion environmental scalars, so they join the Tier-1 spatial set
(means + all-motion gradients) in ``features.assemble``.
"""

from __future__ import annotations

import numpy as np

from ..config import SR_LAYERS
from .profiles import Profiles

# SR-wind layers + an upper 6–9 km layer (mid/upper-level lapse rate).
LAPSE_LAYERS = tuple(SR_LAYERS) + (("6_9km", 6000.0, 9000.0),)


def lapse_feature_names() -> list[str]:
    """Names of the raw lapse-rate point features (for the Tier-1 spatial set)."""
    return [f"lapse_{name}" for name, _, _ in LAPSE_LAYERS]


def _temp_at_heights(prof: Profiles, heights_m: np.ndarray) -> np.ndarray:
    """Profile temperature (K) interpolated to AGL heights (m). Returns
    (H, ny, nx). Linear in height; clamped at the profile top (no extrapolation) —
    mirrors ``profiles.interp_to_heights`` but for the temperature field."""
    z = prof.hgt_agl                       # (K, ny, nx), increasing with level
    K, ny, nx = z.shape
    N = ny * nx
    zf = z.reshape(K, N)
    tf = prof.tmp.reshape(K, N)
    cols = np.arange(N)
    out = np.empty((len(heights_m), N))
    for h, zt in enumerate(heights_m):
        idx = np.sum(zf < zt, axis=0)      # levels strictly below target
        idx_hi = np.clip(idx, 1, K - 1)
        idx_lo = idx_hi - 1
        z_lo = zf[idx_lo, cols]
        z_hi = zf[idx_hi, cols]
        w = np.where(z_hi > z_lo, (zt - z_lo) / (z_hi - z_lo), 0.0)
        w = np.clip(w, 0.0, 1.0)
        out[h] = tf[idx_lo, cols] + w * (tf[idx_hi, cols] - tf[idx_lo, cols])
    return out.reshape(len(heights_m), ny, nx)


def lapse_rate_features(prof: Profiles) -> dict[str, np.ndarray]:
    """Per-layer lapse rate (K/km) keyed ``lapse_<layer>`` for every LAPSE_LAYERS
    entry. Each is (top−bottom) temperature drop over the layer depth."""
    heights = sorted({zb for _, zb, _ in LAPSE_LAYERS} | {zt for _, _, zt in LAPSE_LAYERS})
    T = _temp_at_heights(prof, np.asarray(heights, dtype=float))
    hidx = {h: i for i, h in enumerate(heights)}
    out: dict[str, np.ndarray] = {}
    for name, zb, zt in LAPSE_LAYERS:
        t_bot = T[hidx[zb]]
        t_top = T[hidx[zt]]
        out[f"lapse_{name}"] = (t_bot - t_top) / (zt - zb) * 1000.0  # K per km
    return out
