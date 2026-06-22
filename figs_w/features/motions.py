"""Reference motions for FIGS-W: just two frames.

  * ``none``      : zero motion → ground-relative SR winds / hodographs.
  * ``mean_wind`` : 0–6 km AGL mean wind (the same vector FIGS uses for Bunkers'
                    mean, reused here as the single advective frame).

Each is a pair of (ny, nx) u/v grids, matching the interface of
``figs.features.motions.all_motions`` so the downstream feature functions
(sr_params, hodograph, spatial) work unchanged.
"""

from __future__ import annotations

import numpy as np

from figs.features.motions import mean_wind_height
from figs.features.profiles import Profiles


def all_motions(prof: Profiles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ny, nx = prof.tmp.shape[1:]
    zero = (np.zeros((ny, nx)), np.zeros((ny, nx)))
    mu, mv = mean_wind_height(prof, 0.0, 6000.0)
    return {"none": zero, "mean_wind": (mu, mv)}
