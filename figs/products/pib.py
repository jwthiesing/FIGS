"""Peak Intensity Bin (PIB) products.

The PIB subsystem predicts, per hazard and grid cell, a 7-class distribution over
the unified Peak-Intensity-Bin scale (PIB1..PIB7 = ``config.PIB_LABELS``) — the
"most probable peak intensity within 25 mi". These helpers turn that predicted
distribution into map-ready fields:

  * ``most_probable_pib`` — the modal (argmax) PIB index per cell (0..6), the
    "Most Probable Peak Intensity" shown on the SPC-style chart; -1 where no mass.
  * ``expected_pib`` — the probability-weighted mean PIB index (continuous), useful
    for a smoother fill.
"""

from __future__ import annotations

import numpy as np

from ..config import PIB_LABELS


def most_probable_pib(dist_stack: np.ndarray) -> np.ndarray:
    """Modal PIB index (0..6 == PIB1..PIB7) from a (7, ny, nx) PIB distribution.
    Returns (ny, nx) int8; -1 where the distribution has no mass (NaN / all-zero)."""
    nb = dist_stack.shape[0]
    assert nb == len(PIB_LABELS), f"expected {len(PIB_LABELS)} PIB bins, got {nb}"
    d = np.nan_to_num(dist_stack, nan=0.0)
    s = d.sum(axis=0)
    out = np.argmax(d, axis=0).astype(np.int8)
    out[s <= 0] = -1
    return out


def expected_pib(dist_stack: np.ndarray) -> np.ndarray:
    """Probability-weighted mean PIB index (continuous, 0..6) from a (7, ny, nx) PIB
    distribution. Returns (ny, nx) float32; NaN where the distribution has no mass."""
    nb = dist_stack.shape[0]
    d = np.nan_to_num(dist_stack, nan=0.0)
    s = d.sum(axis=0)
    weights = np.arange(nb)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ev = (d * weights).sum(axis=0) / s
    return np.where(s > 0, ev, np.nan).astype(np.float32)
