"""Probability calibration.

Calibrates raw hazard probabilities against observed frequencies on the
validation split. Default is isotonic regression (monotone, non-parametric, the
standard reliability calibration); a binned-logistic variant is also provided to
match the nadocast per-bin logistic approach.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Calibrator:
    """Monotone probability calibrator (isotonic by default)."""

    def __init__(self, method: str = "isotonic"):
        assert method in ("isotonic", "logistic")
        self.method = method
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray, sample_weight=None) -> "Calibrator":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p, y, sample_weight=sample_weight)
        else:
            from sklearn.linear_model import LogisticRegression

            eps = 1e-6
            logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
            self._model = LogisticRegression()
            self._model.fit(logit.reshape(-1, 1), y.astype(int), sample_weight=sample_weight)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return self._model.predict(p.ravel()).reshape(p.shape)
        eps = 1e-6
        logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
        return self._model.predict_proba(logit.reshape(-1, 1))[:, 1].reshape(p.shape)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Calibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def low_dense_edges(n_low: int = 12, n_high: int = 4, split: float = 0.15):
    """Bin edges packed at low probabilities (where severe-weather forecasts live)
    and sparse at high: ``n_low`` bins in [0, split], ``n_high`` in [split, 1]."""
    return np.unique(np.concatenate([
        np.linspace(0.0, split, n_low + 1),
        np.linspace(split, 1.0, n_high + 1),
    ]))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10, sample_weight=None, edges=None):
    """Return (bin_mean_pred, bin_obs_freq, bin_count) for a reliability diagram.

    Pass ``sample_weight`` (the subsample-reweighting weights) so the observed
    frequency reflects the TRUE climatological base rate the model/calibrator
    target — an unweighted diagram measures the inflated subsampled rate and makes
    correctly-calibrated probabilities look badly biased. Pass ``edges`` (e.g.
    ``low_dense_edges()``) for non-uniform bins concentrated at low probability."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1) if edges is None else np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred, obs, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        wsum = w[m].sum()
        if wsum > 0:
            pred.append((p[m] * w[m]).sum() / wsum)
            obs.append((y[m] * w[m]).sum() / wsum)
            cnt.append(float(wsum))
    return np.array(pred), np.array(obs), np.array(cnt)


def reliability_ci(p, y, sample_weight=None, edges=None, n_bins: int = 10,
                   n_boot: int = 300, ci: float = 0.95, seed: int = 0, min_pos: int = 0):
    """Reliability with bootstrapped CIs on the observed frequency per bin.

    Returns (bin_mean_pred, bin_obs_freq, lo, hi, bin_weight, n_pos) where lo/hi
    are the central-``ci`` percentile band from resampling each bin's (y, weight)
    with replacement ``n_boot`` times, and ``n_pos`` is the raw positive count in
    each bin (the driver of CI width for rare events). Sparse bins get wide bands.
    ``min_pos`` drops bins with fewer than that many positive samples (their
    weighted frequency is dominated by a handful of points and is unreliable)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1) if edges is None else np.asarray(edges, dtype=float)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    rng = np.random.default_rng(seed)
    a = (1.0 - ci) / 2.0
    pred, obs, lo, hi, cnt, npos = [], [], [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        np_b = int((y[m] > 0).sum())
        if nb == 0 or np_b < min_pos:
            continue
        pb, yb, wb = p[m], y[m], w[m]
        W = wb.sum()
        pred.append((pb * wb).sum() / W)
        obs.append((yb * wb).sum() / W)
        cnt.append(float(W))
        npos.append(np_b)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            s = rng.integers(0, nb, nb)          # resample this bin's points
            ws = wb[s]
            boots[i] = (yb[s] * ws).sum() / ws.sum()
        lo.append(float(np.quantile(boots, a)))
        hi.append(float(np.quantile(boots, 1.0 - a)))
    return (np.array(pred), np.array(obs), np.array(lo), np.array(hi),
            np.array(cnt), np.array(npos))
