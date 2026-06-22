"""FIGS-W inference: assemble one HRRR cycle's state, run the wildfire models, and
return probability + deadliness + conditional-size grids per forecast hour.

Every forecast hour ensembles ALL trained lead bands (a lead-diverse ensemble, as
in FIGS): band predictions are calibrated then averaged.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from figs.model.calibrate import Calibrator
from figs.model.wrapper import GBDTModel

from .. import config as C
from ..data import dataset


def _band_tags(models_dir: Path) -> list[str]:
    tags = [b.name for b in C.LEAD_BANDS if (models_dir / f"hazard_wildfire_{b.name}.pkl").exists()]
    return tags or (["pooled"] if (models_dir / "hazard_wildfire_pooled.pkl").exists() else [])


def _feature_matrix(run, fxx, feat_cols):
    feats = dataset._features_for_valid(run, fxx, cached_only=False)
    ny, nx = next(iter(feats.values())).shape
    cols = [feats[c] if c in feats else np.full((ny, nx), np.nan, np.float32) for c in feat_cols]
    X = np.stack(cols, axis=0).reshape(len(feat_cols), ny * nx).T.astype(np.float32)
    return X, (ny, nx)


def _avg_binary(models_dir, tags, kind, X):
    """Mean of per-band calibrated probabilities for a binary model family."""
    ps = []
    for t in tags:
        mp = models_dir / f"hazard_{kind}_{t}.pkl"
        if not mp.exists():
            continue
        p = GBDTModel.load(mp).predict_pos(X)
        cp = models_dir / f"calib_{kind}_{t}.pkl"
        if cp.exists():
            p = Calibrator.load(cp).transform(p)
        ps.append(p)
    return np.mean(ps, axis=0) if ps else np.zeros(len(X), np.float32)


def predict_valid(run, fxx, models_dir=None) -> dict:
    models_dir = Path(models_dir) if models_dir else C.MODELS
    feat_cols = json.loads((models_dir / "feature_cols.json").read_text())
    tags = _band_tags(models_dir)
    X, (ny, nx) = _feature_matrix(run, fxx, feat_cols)

    p_fire = _avg_binary(models_dir, tags, "wildfire", X).reshape(ny, nx)
    p_dead = _avg_binary(models_dir, tags, "deadly", X).reshape(ny, nx)

    nb = len(C.INTENSITY_BINS["wildfire"]["labels"])
    probas = []
    for t in tags:
        sp = models_dir / f"intensity_wildfire_{t}.pkl"
        if sp.exists():
            sm = GBDTModel.load(sp)
            pr = sm.predict_proba(X)
            full = np.zeros((len(X), nb), np.float32)
            for j, c in enumerate(np.asarray(sm.classes_).astype(int)):
                if 0 <= c < nb:
                    full[:, c] = pr[:, j]
            probas.append(full)
    dist = (np.mean(probas, axis=0).T.reshape(nb, ny, nx) if probas
            else np.full((nb, ny, nx), np.nan, np.float32))
    return {"p_wildfire": p_fire.astype(np.float32), "p_deadly": p_dead.astype(np.float32),
            "dist_wildfire": dist}


def predict_forecast(run, fxx_list, models_dir=None) -> dict:
    out = {}
    for f in fxx_list:
        out[int(f)] = predict_valid(run, int(f), models_dir=models_dir)
        print(f"[predict-w] f{int(f):02d} done", flush=True)
    return out
