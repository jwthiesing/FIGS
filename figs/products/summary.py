"""Day-total / cumulative summary products.

From per-forecast-hour prediction grids:
  * ``day_max``                  — max of a field over all forecast hours;
  * ``cumulative_categorical``   — apply the probability×CIG conversion to the
                                   day-max probability + day-max CIG -> a single
                                   cumulative daily SPC-style categorical outlook;
  * ``median_intensity_bin``     — median conditional-intensity bin from a
                                   conditional distribution (per cell).
"""

from __future__ import annotations

import numpy as np

from ..config import HAZARDS
from . import cig


def day_max(grids_by_fxx: dict[int, np.ndarray]) -> np.ndarray:
    """Element-wise maximum of a per-forecast-hour field over all hours."""
    stack = np.stack(list(grids_by_fxx.values()), axis=0)
    return np.nanmax(stack, axis=0)


def median_intensity_bin(dist_stack: np.ndarray) -> np.ndarray:
    """Median conditional-intensity bin index from a (nbins, ny, nx) distribution
    (the smallest bin where the cumulative probability reaches 0.5). Cells with no
    mass return -1."""
    s = dist_stack.sum(axis=0)
    cdf = np.cumsum(dist_stack, axis=0)
    total = np.where(s <= 0, 1.0, s)
    cdf = cdf / total
    med = np.argmax(cdf >= 0.5, axis=0).astype(np.int16)
    med[s <= 0] = -1
    return med


def cumulative_categorical(
    hazard: str,
    prob_by_fxx: dict[int, np.ndarray],
    dist_by_fxx: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Cumulative daily categorical risk for a hazard.

    Takes the day-max probability and the day-max CIG category (CIG derived per
    hour from the conditional-intensity distribution, then maxed over hours), and
    converts to the SPC categorical level via ``cig.prob_to_category``.

    Returns {'prob' day-max prob (0..1), 'cig' day-max CIG idx, 'category' 0..5}.
    """
    prob_max = day_max(prob_by_fxx)
    cig_per_fxx = {f: cig.derive_cig_category(hazard, dist_by_fxx[f]) for f in dist_by_fxx}
    cig_max = np.nanmax(np.stack(list(cig_per_fxx.values()), axis=0), axis=0).astype(int)
    category = cig.prob_to_category(hazard, prob_max * 100.0, cig_max)
    return {"prob": prob_max, "cig": cig_max, "category": category}


def combined_categorical(predictions: dict) -> dict:
    """SPC-style cumulative daily CATEGORICAL outlook **across all hazards**.

    The SPC Day-1 categorical outlook is the single highest risk implied by any
    hazard, so we take the element-wise MAX of each hazard's cumulative daily
    category (day-max prob + day-max CIG -> category). ``predictions`` maps
    fxx -> {'p_<h>':grid, 'dist_<h>':(nbins,ny,nx)}.

    Returns {'category' (0..5 combined), 'by_hazard' {h: category}}.
    """
    fxxs = sorted(predictions)
    by_hazard: dict[str, np.ndarray] = {}
    for h in HAZARDS:
        prob_by = {f: predictions[f][f"p_{h}"] for f in fxxs}
        dist_by = {f: np.nan_to_num(predictions[f][f"dist_{h}"]) for f in fxxs}
        by_hazard[h] = cumulative_categorical(h, prob_by, dist_by)["category"]
    category = np.nanmax(np.stack(list(by_hazard.values()), axis=0), axis=0).astype(int)
    return {"category": category, "by_hazard": by_hazard}
