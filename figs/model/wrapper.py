"""GBDTModel — a thin abstraction over the gradient-boosting backend.

The default backend is ``mlx-boosting`` (Apple-Silicon GBDT). Its
``XGBoostClassifier`` supports binary and multiclass via ``objective`` and
auto-detected class count, handles NaN, and takes the usual tree hyperparameters
— but its ``fit(X, y)`` has **no sample-weight, early-stopping, or eval-set**
support. FIGS needs sample weights for the subsample-with-reweighting scheme, so
this wrapper emulates them by weighted resampling before fitting.

Set ``backend="lightgbm"`` to use LightGBM instead (native weights/early
stopping), keeping an identical interface for the rest of FIGS.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import contextlib
import warnings


@contextlib.contextmanager
def _no_feature_name_warning():
    """Silence sklearn's benign 'X does not have valid feature names' warning —
    FIGS trains/predicts on NumPy with a fixed column order, so names are moot."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        yield


DEFAULT_HP = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=None,        # None -> 2**max_depth (the aggressive full-tree ceiling)
    min_child_weight=1.0,    # min sum-hessian per leaf
    min_child_samples=20,    # min ROW count per leaf (weight-independent overfit guard)
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    n_bins=256,
    max_bin=255,             # LightGBM feature bins; keep <=255 (uint8 binned data).
                             # histogram RAM scales with num_features*num_leaves*max_bin
)


class GBDTModel:
    """Binary or multiclass gradient-boosted-tree classifier with a uniform API."""

    def __init__(self, task: str = "binary", backend: str = "mlx", *,
                 resample_seed: int = 0, **hp):
        assert task in ("binary", "multiclass")
        self.task = task
        self.backend = backend
        self.hp = {**DEFAULT_HP, **hp}
        self.resample_seed = resample_seed
        self._model = None
        self.classes_ = None

    # ------------------------------------------------------------------ #
    # sample-weight emulation
    # ------------------------------------------------------------------ #
    def _apply_weights(self, X, y, sample_weight):
        """Weighted bootstrap: resample rows with probability ∝ weight so an
        unweighted learner approximates weighted training. Returns (X', y')."""
        if sample_weight is None:
            return X, y
        w = np.asarray(sample_weight, dtype=np.float64)
        w = np.clip(w, 0, None)
        total = w.sum()
        if total <= 0:
            return X, y
        p = w / total
        rng = np.random.default_rng(self.resample_seed)
        n = len(y)
        idx = rng.choice(n, size=n, replace=True, p=p)
        return X[idx], y[idx]

    # ------------------------------------------------------------------ #
    # fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int32)
        self.classes_ = np.unique(y)
        if self.backend == "mlx":
            # mlx-boosting has no sample_weight. Rather than materialize a full
            # resampled copy of X (memory-prohibitive on large bands), train
            # UNWEIGHTED — the subsample-with-reweighting base rate is recovered
            # by the post-hoc Calibrator (fit with validation weights).
            self._fit_mlx(X, y)
        elif self.backend == "lightgbm":
            self._fit_lgbm(X, y, sample_weight)  # native weights, no resample
        else:
            raise ValueError(f"unknown backend {self.backend}")
        return self

    def _fit_mlx(self, X, y):
        import mlx.core as mx
        from mlx_boosting import XGBoostClassifier

        objective = "binary:logistic" if len(self.classes_) == 2 else "multi:softmax"
        self._model = XGBoostClassifier(objective=objective, **self.hp)
        Xmx, ymx = mx.array(X), mx.array(y)
        # free the NumPy feature matrix (the caller's to_numpy() temporary) now
        # that it lives in MLX unified memory — halves peak during the fit.
        del X
        self._model.fit(Xmx, ymx)

    def _lgb_params(self, n_classes: int) -> dict:
        """LightGBM params shared by the sklearn (numpy) and native (Dataset) paths.
        ``n_estimators``/``num_boost_round`` is applied by the caller."""
        params = dict(
            objective="binary" if n_classes == 2 else "multiclass",
            num_leaves=self.hp.get("num_leaves") or 2 ** self.hp["max_depth"],
            max_depth=self.hp["max_depth"],
            learning_rate=self.hp["learning_rate"],
            subsample=self.hp["subsample"],
            subsample_freq=1,                       # else subsample is ignored
            colsample_bytree=self.hp["colsample_bytree"],
            reg_lambda=self.hp["reg_lambda"],
            min_child_weight=self.hp["min_child_weight"],
            min_child_samples=self.hp.get("min_child_samples", 20),
            max_bin=min(self.hp.get("max_bin", 255), 255),  # >255 -> uint16 binned (2x RAM)
            force_col_wise=True,                    # wide data: skip the row/col probe (saves RAM)
            n_jobs=-1,                              # all cores
            verbose=-1,                             # quiet
        )
        if n_classes > 2:
            params["num_class"] = n_classes
        return params

    def _fit_lgbm(self, X, y, sample_weight):
        import lightgbm as lgb

        n_classes = len(self.classes_)
        params = {**self._lgb_params(n_classes), "n_estimators": self.hp["n_estimators"]}
        self._model = lgb.LGBMClassifier(**params)
        with _no_feature_name_warning():
            self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.backend == "mlx":
            import mlx.core as mx

            proba = np.array(self._model.predict_proba(mx.array(X)))
        else:
            with _no_feature_name_warning():
                proba = self._model.predict_proba(X)
        return proba

    def predict_pos(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class (binary) — the common hazard case."""
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return proba
        return proba[:, 1] if proba.shape[1] == 2 else proba

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "GBDTModel":
        with open(path, "rb") as f:
            return pickle.load(f)
