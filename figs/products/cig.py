"""CIG-category derivation and probability -> SPC categorical-risk conversion.

Two steps, both grid-vectorized:

1. ``derive_cig_category`` — from a predicted conditional-intensity distribution,
   assign the CIG category (0=<CIG1 .. 3=CIG3) by comparing the distribution's
   "severity" (expected intensity-bin index) against the reference distributions
   in ``config.CIG_REFERENCE``.

2. ``prob_to_category`` — from hazard probability + CIG category, look up the SPC
   categorical level (0 TSTM .. 5 HIGH) via ``config.CIG_CONVERSION``.
"""

from __future__ import annotations

import numpy as np

from ..config import CIG_CATEGORIES, CIG_CONVERSION, CIG_REFERENCE, INTENSITY_BINS


def _ref_severities(hazard: str) -> np.ndarray:
    """Expected bin index (severity) of each CIG reference distribution, ascending
    by category (<CIG1, CIG1, CIG2, CIG3)."""
    sev = []
    for cat in CIG_CATEGORIES:
        ref = np.array(CIG_REFERENCE[hazard][cat], dtype=float)
        ref = ref / ref.sum()
        sev.append(float(np.dot(ref, np.arange(len(ref)))))
    return np.array(sev)


def derive_cig_category(hazard: str, dist_stack: np.ndarray) -> np.ndarray:
    """Map a predicted conditional-intensity distribution to a CIG category index.

    ``dist_stack`` is (nbins, ny, nx) of conditional probabilities (need not sum
    to 1 exactly; it is renormalized). Returns an (ny, nx) int array in 0..3.
    A cell's severity (expected bin index) is bucketed at the midpoints between
    consecutive reference severities.
    """
    nbins = dist_stack.shape[0]
    assert nbins == len(INTENSITY_BINS[hazard]["labels"])
    s = dist_stack.sum(axis=0)
    s = np.where(s <= 0, 1.0, s)
    weights = np.arange(nbins)[:, None, None]
    severity = (dist_stack * weights).sum(axis=0) / s    # (ny, nx)

    ref = _ref_severities(hazard)                        # (4,) ascending
    edges = (ref[:-1] + ref[1:]) / 2.0                   # 3 midpoint thresholds
    return np.digitize(severity, edges).astype(np.int8)  # 0..3


def _filled_table(hazard: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds (R,), category table (R, 4)) with None entries filled by
    the row's max defined category (so very-high-CIG / low-prob cells are capped)."""
    rows = CIG_CONVERSION[hazard]
    thr = np.array([r[0] for r in rows], dtype=float)
    R = len(rows)
    table = np.zeros((R, 4), dtype=np.int8)
    for i, (_, cats) in enumerate(rows):
        defined = [c for c in cats if c is not None]
        rowmax = max(defined) if defined else 0
        for j in range(4):
            c = cats[j] if j < len(cats) else None
            table[i, j] = rowmax if c is None else c
    return thr, table


def prob_to_category(hazard: str, prob_pct: np.ndarray, cig_idx: np.ndarray) -> np.ndarray:
    """SPC categorical level (0..5) from hazard probability (%) and CIG index.

    ``prob_pct`` and ``cig_idx`` are (ny, nx). Returns 0..5 (0 = TSTM) where the
    probability reaches the lowest threshold, and **-1 below it** (no risk) so that
    the genuine TSTM (0) area is distinct from the un-drawn background."""
    thr, table = _filled_table(hazard)
    prob_pct = np.asarray(prob_pct, dtype=float)
    cig_idx = np.clip(np.asarray(cig_idx, dtype=int), 0, 3)
    row = np.searchsorted(thr, prob_pct, side="right") - 1   # -1 below all
    out = np.full(prob_pct.shape, -1, dtype=np.int8)         # -1 = below lowest threshold
    valid = row >= 0
    rr = np.clip(row, 0, len(thr) - 1)
    looked = table[rr, cig_idx]
    out[valid] = looked[valid]
    return out


def categorical_risk(hazard: str, prob_pct: np.ndarray, dist_stack: np.ndarray) -> dict:
    """Convenience: derive CIG category then the SPC categorical level.
    Returns {'cig': (ny,nx) 0..3, 'category': (ny,nx) 0..5}."""
    cig = derive_cig_category(hazard, dist_stack)
    cat = prob_to_category(hazard, prob_pct, cig)
    return {"cig": cig, "category": cat}
