"""Fire CIG-category derivation and probability → SPC fire-weather conversion.

Mirrors ``figs.products.cig`` but for wildfires: the conditional **size**
distribution drives the CIG category, and probability + CIG map to the fire-weather
outlook (0 NONE, 1 ELEVATED, 2 CRITICAL, 3 EXTREME) via ``config.CIG_CONVERSION``.

The four CIG reference size distributions in ``config.CIG_REFERENCE`` are meant to
be **fit from the training size distribution** — ``fit_cig_reference`` does that
(and ``marginal_size_distribution`` reads it off a built parquet).
"""

from __future__ import annotations

import numpy as np

from ..config import CIG_CATEGORIES, CIG_CONVERSION, CIG_REFERENCE, INTENSITY_BINS


def _ref_severities(hazard: str) -> np.ndarray:
    sev = []
    for cat in CIG_CATEGORIES:
        ref = np.array(CIG_REFERENCE[hazard][cat], dtype=float)
        ref = ref / ref.sum()
        sev.append(float(np.dot(ref, np.arange(len(ref)))))
    return np.array(sev)


def derive_cig_category(hazard: str, dist_stack: np.ndarray) -> np.ndarray:
    """Map a predicted conditional-size distribution (nbins, ny, nx) to a CIG index
    0..3, by bucketing the distribution's expected bin index at the midpoints
    between the reference distributions' severities."""
    nbins = dist_stack.shape[0]
    assert nbins == len(INTENSITY_BINS[hazard]["labels"])
    s = dist_stack.sum(axis=0)
    s = np.where(s <= 0, 1.0, s)
    weights = np.arange(nbins)[:, None, None]
    severity = (dist_stack * weights).sum(axis=0) / s
    ref = _ref_severities(hazard)
    edges = (ref[:-1] + ref[1:]) / 2.0
    return np.digitize(severity, edges).astype(np.int8)


def _filled_table(hazard: str):
    rows = CIG_CONVERSION[hazard]
    thr = np.array([r[0] for r in rows], dtype=float)
    table = np.zeros((len(rows), 4), dtype=np.int8)
    for i, (_, cats) in enumerate(rows):
        defined = [c for c in cats if c is not None]
        rowmax = max(defined) if defined else 0
        for j in range(4):
            c = cats[j] if j < len(cats) else None
            table[i, j] = rowmax if c is None else c
    return thr, table


def prob_to_category(hazard: str, prob_pct: np.ndarray, cig_idx: np.ndarray) -> np.ndarray:
    """Fire-weather category (0 NONE .. 3 EXTREME) from probability (%) + CIG index;
    **-1 below the lowest probability level** (no risk drawn)."""
    thr, table = _filled_table(hazard)
    prob_pct = np.asarray(prob_pct, dtype=float)
    cig_idx = np.clip(np.asarray(cig_idx, dtype=int), 0, 3)
    row = np.searchsorted(thr, prob_pct, side="right") - 1
    out = np.full(prob_pct.shape, -1, dtype=np.int8)
    valid = row >= 0
    rr = np.clip(row, 0, len(thr) - 1)
    looked = table[rr, cig_idx]
    out[valid] = looked[valid]
    return out


def categorical_risk(hazard: str, prob_pct: np.ndarray, dist_stack: np.ndarray) -> dict:
    cig = derive_cig_category(hazard, dist_stack)
    return {"cig": cig, "category": prob_to_category(hazard, prob_pct, cig)}


# --------------------------------------------------------------------------- #
# Fitting the CIG reference size distributions from training data
# --------------------------------------------------------------------------- #
def fit_cig_reference(marginal, tilts=(-1.0, 0.0, 1.0, 2.0)) -> dict:
    """Build the four CIG reference size distributions from the **empirical marginal**
    size distribution ``marginal`` (length = n size bins, frequency per bin among
    wildfires). Each CIG category tilts the marginal toward larger sizes by
    ``exp(tilt · bin_index)`` and renormalizes, giving a monotone-increasing-severity
    ladder anchored on the data's actual shape. Returns a dict shaped like
    ``CIG_REFERENCE['wildfire']`` (percentages). Refine ``tilts`` once the
    size data is in hand."""
    p = np.asarray(marginal, dtype=float)
    p = p / p.sum()
    idx = np.arange(len(p))
    refs = {}
    for cat, k in zip(CIG_CATEGORIES, tilts):
        w = p * np.exp(k * idx)
        w = w / w.sum()
        refs[cat] = tuple(round(100.0 * x, 2) for x in w)
    return refs


def marginal_size_distribution(parquet_path: str, hazard: str = "wildfire") -> np.ndarray:
    """Empirical conditional size distribution: frequency of each size bin among
    cells with a wildfire (``{hazard}_bin >= 0``), read off a built parquet."""
    import pandas as pd
    from pathlib import Path

    col = f"{hazard}_bin"
    parts = (sorted(Path(parquet_path).glob("*.parquet"))
             if Path(parquet_path).is_dir() else [Path(parquet_path)])
    nb = len(INTENSITY_BINS[hazard]["labels"])
    counts = np.zeros(nb, dtype=float)
    for p in parts:
        b = pd.read_parquet(p, columns=[col])[col].to_numpy()
        b = b[b >= 0]
        counts += np.bincount(b.astype(int), minlength=nb)[:nb]
    return counts / counts.sum() if counts.sum() else counts
