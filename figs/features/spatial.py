"""Spatial means and motion-relative gradient fields, computed grid-wide.

For a base field we compute its circular spatial mean at 25/50/100 mi
(``grid.neighborhood_mean``), then, relative to each storm-motion vector, the
forward / leftward / straddling gradients of that mean (nadocast definitions):

  * forward    = mean_ahead  - mean_behind
  * leftward   = mean_left   - mean_right
  * straddling = (mean_ahead + mean_behind) - (mean_left + mean_right)

"ahead/behind/left/right" are the spatial-mean field sampled at points displaced
one radius along / across the (per-cell) motion direction, via bilinear sampling.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..config import FIGS_DX_KM, GRADIENT_TYPES, SPATIAL_MEAN_RADII_MI
from ..data import grid
from ..data.grid import MI_TO_KM


def spatial_mean(field: np.ndarray, radius_mi: float) -> np.ndarray:
    """Circular neighborhood mean (delegates to grid.neighborhood_mean)."""
    return grid.neighborhood_mean(field, radius_mi)


def _unit_motion(motion: tuple[np.ndarray, np.ndarray]):
    cu, cv = motion
    mag = np.sqrt(cu**2 + cv**2)
    mag = np.where(mag < 1e-3, 1e-3, mag)
    return cu / mag, cv / mag  # east, north unit components


def directional_gradients(
    mean_field: np.ndarray, motion: tuple[np.ndarray, np.ndarray], radius_mi: float
) -> dict[str, np.ndarray]:
    """forward/leftward/straddling gradients of ``mean_field`` for one motion."""
    du, dv = _unit_motion(motion)                 # east(col), north(row) units
    r = radius_mi * MI_TO_KM / FIGS_DX_KM         # displacement in cells
    rows, cols = np.indices(mean_field.shape).astype(float)

    def sample(drow, dcol):
        coords = np.array([rows + drow, cols + dcol])
        return map_coordinates(mean_field, coords, order=1, mode="nearest")

    ahead = sample(r * dv, r * du)
    behind = sample(-r * dv, -r * du)
    left = sample(r * du, -r * dv)                # +90° (CCW) of motion
    right = sample(-r * du, r * dv)
    return {
        "forward": ahead - behind,
        "leftward": left - right,
        "straddling": (ahead + behind) - (left + right),
    }


def spatial_features(
    base_fields: dict[str, np.ndarray],
    motions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    do_gradients: bool = True,
) -> dict[str, np.ndarray]:
    """Spatial means (all radii) + per-motion directional gradients of those means.

    Output keys:
      ``{field}_mean{r}``                          (spatial mean)
      ``{field}_mean{r}_{motion}_{gradtype}``      (directional gradient)
    """
    out: dict[str, np.ndarray] = {}
    for fname, field in base_fields.items():
        for r in SPATIAL_MEAN_RADII_MI:
            m = spatial_mean(field, r)
            rkey = int(r)
            out[f"{fname}_mean{rkey}"] = m
            if do_gradients:
                for mname, mvec in motions.items():
                    grads = directional_gradients(m, mvec, r)
                    for gt in GRADIENT_TYPES:
                        out[f"{fname}_mean{rkey}_{mname}_{gt}"] = grads[gt]
    return out
