"""Train the FIGS-W models, lead-banded, reusing the FIGS GBDT wrapper + calibrator.

Per band, two models:
  * ``hazard_wildfire_{band}.pkl``  — p(wildfire) occurrence (binary, weighted);
  * ``intensity_wildfire_{band}.pkl`` — conditional SIZE distribution (multiclass,
    positive cells only) → CIG.
Plus ``calib_*_{band}.pkl`` (validation-fit) and ``feature_cols.json``.
(Deadliness is intentionally not modeled — see config.LABEL_FIELDS.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from figs.model.calibrate import Calibrator
from figs.model.wrapper import GBDTModel

from .. import config as C
from ..data.dataset import LABEL_COLS, META_COLS, _dataset_target, read_split


def _feature_cols(parts) -> list[str]:
    import pyarrow.parquet as pq

    names = pq.read_schema(parts[0]).names
    drop = set(META_COLS) | set(LABEL_COLS)
    return [c for c in names if c not in drop]


def train_all(parquet_path: str, out_dir=None, *, band: bool = True,
              max_rows_per_band: int | None = 800_000, val_rows: int = 250_000,
              backend: str = "lightgbm", calibrator: str = "logistic",
              n_estimators: int = 500, max_depth: int = 6, learning_rate: float = 0.05,
              **hp) -> dict:
    out_dir = Path(out_dir) if out_dir else C.MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _dataset_target(parquet_path)
    parts = sorted(Path(target).glob("*.parquet")) if Path(target).is_dir() else [Path(target)]
    feats = _feature_cols(parts)
    (out_dir / "feature_cols.json").write_text(json.dumps(feats))
    aux = ["weight"] + LABEL_COLS
    bands = list(C.LEAD_BANDS) if band else [None]
    hpc = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, **hp)
    metrics: dict = {}

    for b in bands:
        tag = b.name if b else "pooled"
        filters = [("fxx", ">=", b.fmin), ("fxx", "<=", b.fmax)] if b else None
        t0 = time.time()
        Xtr, atr = read_split(parquet_path, feature_cols=feats, aux_cols=aux, split="train",
                              filters=filters, cap=max_rows_per_band, seed=0)
        Xva, ava = read_split(parquet_path, feature_cols=feats, aux_cols=aux, split="validation",
                              filters=filters, cap=val_rows, seed=1)
        if len(Xtr) == 0:
            continue
        wtr = atr["weight"].to_numpy(np.float32)
        wva = ava["weight"].to_numpy(np.float32) if len(Xva) else None
        m: dict = {"n_train": int(len(Xtr)), "n_val": int(len(Xva))}

        # binary occurrence target (p(wildfire) in the 25 mi neighborhood):
        targets = {"wildfire": (atr["wildfire"].to_numpy(int),
                                ava["wildfire"].to_numpy(int) if len(Xva) else None)}
        for name, (ytr_b, yva_b) in targets.items():
            if ytr_b.max() == ytr_b.min():          # no positives in this band → skip
                continue
            mdl = GBDTModel(task="binary", backend=backend, **hpc)
            mdl.fit(Xtr, ytr_b, sample_weight=wtr)
            mdl.save(out_dir / f"hazard_{name}_{tag}.pkl")
            if yva_b is not None and yva_b.max() > yva_b.min():
                p = mdl.predict_pos(Xva)
                Calibrator(method=calibrator).fit(p, yva_b, sample_weight=wva).save(
                    out_dir / f"calib_{name}_{tag}.pkl")

        # conditional SIZE (multiclass): bin the RAW wildfire_size (acres) at train
        # time via config edges → bins can change with a retrain, no rebuild.
        edges = np.asarray(C.INTENSITY_BINS["wildfire"]["edges"], float)
        sz = atr["wildfire_size"].to_numpy(float)
        idx = np.isfinite(sz) & (sz > 0)
        if idx.sum() >= 100:
            ybin = (np.searchsorted(edges, sz[idx], side="right") - 1).clip(0, len(edges) - 1)
            sm = GBDTModel(task="multiclass", backend=backend, **hpc)
            sm.fit(Xtr[idx], ybin.astype(int), sample_weight=wtr[idx])
            sm.save(out_dir / f"intensity_wildfire_{tag}.pkl")
            m["n_size_pos"] = int(idx.sum())
        m["seconds"] = round(time.time() - t0, 1)
        metrics[tag] = m
        print(f"[{tag}] {m}", flush=True)

    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
