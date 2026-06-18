"""Per-motion hodograph rotation, computed grid-wide.

nadocast rotates all winds relative to a single mean-wind shear vector so the
learner sees a rotationally-invariant hodograph. FIGS does this independently
for each storm-motion vector: the wind field is rotated so the motion vector
points along +x, and winds are expressed storm-relative. The rotated
storm-relative winds at a set of AGL heights become rotationally-invariant
hodograph features.
"""

from __future__ import annotations

import numpy as np

from .profiles import Profiles, interp_to_heights

# AGL heights (m) at which to sample the rotated hodograph as features.
HODO_HEIGHTS = np.array([0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000.0])


def rotate(u: np.ndarray, v: np.ndarray, motion: tuple[np.ndarray, np.ndarray]):
    """Rotate (u, v) so the motion vector maps onto +x. Returns (u', v').

    With φ = atan2(cv, cu), the rotation by -φ gives:
        u' =  u cosφ + v sinφ
        v' = -u sinφ + v cosφ
    so the motion vector (cu, cv) -> (|C|, 0).
    """
    cu, cv = motion
    phi = np.arctan2(cv, cu)
    cphi, sphi = np.cos(phi), np.sin(phi)
    up = u * cphi + v * sphi
    vp = -u * sphi + v * cphi
    return up, vp


def rotated_sr_hodograph(prof: Profiles, motion: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Storm-relative winds at HODO_HEIGHTS, rotated into the motion's frame.

    Returns ``{f'u{h}': ..., f'v{h}': ...}`` (m/s) for each height ``h`` (m).
    The +x axis aligns with the storm motion, so these components are invariant
    to the absolute direction of the environment — only the hodograph *shape*
    relative to the storm motion matters.
    """
    cu, cv = motion
    u, v = interp_to_heights(prof, HODO_HEIGHTS)      # (H, ny, nx)
    usr = u - cu[None]
    vsr = v - cv[None]
    up, vp = rotate(usr, vsr, motion)
    out: dict[str, np.ndarray] = {}
    for i, h in enumerate(HODO_HEIGHTS):
        out[f"u{int(h)}"] = up[i]
        out[f"v{int(h)}"] = vp[i]
    return out


def all_rotated_hodographs(
    prof: Profiles, motions: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Rotated SR hodograph features for every motion, keyed ``<motion>_<comp><h>``."""
    out: dict[str, np.ndarray] = {}
    for mname, mvec in motions.items():
        for k, val in rotated_sr_hodograph(prof, mvec).items():
            out[f"{mname}_{k}"] = val
    return out
