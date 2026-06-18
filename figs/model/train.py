"""Training orchestration with lead-time banding and calibration.

Trains, per lead-time band (``config.LEAD_BANDS``) when an ``fxx`` column is
present (else one pooled model):
  * three binary hazard models — p(tor), p(wind), p(hail);
  * three multiclass conditional-intensity models (positive cells, known bin);
  * a probability calibrator per hazard, fit on the validation split.

Models are saved as ``{group}_{band}.pkl`` (band = band name or 'pooled') plus
``calib_{hazard}_{band}.pkl`` and ``feature_cols.json``. The nadocast weekly
split (``split`` column) selects train vs validation rows; sample weights flow
through ``GBDTModel`` (subsample-with-reweighting).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HAZARDS, LEAD_BANDS, MODELS
from ..data.dataset import dataset_feature_columns, read_split
from .calibrate import Calibrator
from .wrapper import GBDTModel




def _fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _clear_mlx_cache():
    """Return MLX's cached unified-memory buffers to the OS (MLX pools freed
    arrays, so without this RSS climbs across successive fits)."""
    try:
        import mlx.core as mx

        for fn in ("clear_cache", "reset_peak_memory"):
            f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", None), fn, None)
            if callable(f):
                f()
    except Exception:  # noqa: BLE001
        pass


def train_all(parquet_path: str, out_dir: str | None = None, *,
              band: bool = True, calibrate: bool = True, max_rows_per_band: int | None = None,
              val_rows: int | None = 1_000_000, backend: str = "mlx",
              calibrator: str = "logistic", n_bags: int = 1, **hp) -> dict:
    """Train all model groups (per lead band when banding) + calibrators, **streaming
    one lead band at a time** from the parquet so the full dataset never has to fit
    in memory. ``max_rows_per_band`` caps a band's (per-bag) train rows.

    ``n_bags`` > 1 enables **bagging**: per band/hazard, train K hazard models, each
    on ALL positive rows plus a disjoint 1/K fold of the negatives, so the ensemble
    covers all stored negatives despite the per-model RAM ceiling (and the averaged
    prediction is smoother). The conditional-intensity model is trained once (all
    positives live in every bag); the calibrator is fit on the bag-MEAN validation
    prediction. Logs per-(band, hazard, bag) progress + ETA."""
    import time

    feats = dataset_feature_columns(parquet_path)
    out_dir = Path(out_dir) if out_dir else MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_cols.json").write_text(json.dumps(feats))
    print(f"{len(feats)} features; streaming per lead band", flush=True)

    # detect banding from the schema (is fxx present?)
    import pyarrow.parquet as pq
    from ..data.dataset import _dataset_target, read_dataset
    target = _dataset_target(parquet_path)
    src = sorted(Path(target).glob("*.parquet"))[0] if Path(target).is_dir() else target
    has_fxx = "fxx" in pq.read_schema(src).names
    use_bands = band and has_fxx
    if use_bands:
        # keep only bands that actually have TRAIN rows (cheap fxx/split scan), so
        # the model count + ETA reflect reality instead of all 8 nominal bands.
        meta = read_dataset(parquet_path, columns=["fxx", "split"])
        tr_mask = meta["split"] == "train"
        bands = [b for b in LEAD_BANDS
                 if (tr_mask & (meta["fxx"] >= b.fmin) & (meta["fxx"] <= b.fmax)).any()]
        del meta
        print(f"populated bands: {[b.name for b in bands]}", flush=True)
    else:
        bands = [None]
    metrics: dict = {}
    n_bags = max(1, int(n_bags))
    total = len(bands) * len(HAZARDS) * n_bags   # hazard fits (the dominant work)
    step, t0 = 0, time.time()

    from sklearn.metrics import average_precision_score, roc_auc_score

    label_cols = [h for h in HAZARDS] + [f"{h}_bin" for h in HAZARDS]
    aux_cols = label_cols + ["weight"]
    for b in bands:
        tag = "pooled" if b is None else b.name
        filters = None if b is None else [("fxx", ">=", b.fmin), ("fxx", "<=", b.fmax)]
        tl = time.time()
        # validation matrix: read ONCE, shared across bags (calibrates the bag-mean).
        Xva, aux_va = read_split(parquet_path, feature_cols=feats, aux_cols=aux_cols,
                                 split="validation", filters=filters, cap=val_rows)
        yva = {h: aux_va[h].to_numpy(np.int8) for h in HAZARDS}
        wva = aux_va["weight"].to_numpy(np.float32)
        del aux_va
        print(f" band {tag}: {len(Xva):,} val rows; training {n_bags} bag(s) "
              f"(loaded val {_fmt(time.time()-tl)})", flush=True)

        # bag loop: each bag = all positives + a disjoint negative fold. Accumulate
        # the bag predictions on the shared val set to calibrate/score the MEAN.
        pva_sum = {h: np.zeros(len(Xva), np.float64) for h in HAZARDS}
        used_bags, rows_per_bag = 0, 0
        for k in range(n_bags):
            Xtr, aux_tr = read_split(parquet_path, feature_cols=feats, aux_cols=aux_cols,
                                     split="train", filters=filters, cap=max_rows_per_band,
                                     bag=(k if n_bags > 1 else None), n_bags=n_bags,
                                     positive_cols=list(HAZARDS))
            if len(Xtr) == 0:
                continue
            ytr = {h: aux_tr[h].to_numpy(np.int8) for h in HAZARDS}
            ybin = {h: aux_tr[f"{h}_bin"].to_numpy(np.int8) for h in HAZARDS}
            wtr = aux_tr["weight"].to_numpy(np.float32)
            rows_per_bag = len(Xtr)
            del aux_tr
            _clear_mlx_cache()
            suffix = f"_bag{k}" if n_bags > 1 else ""
            for h in HAZARDS:
                ts = time.time()
                hm = GBDTModel(task="binary", backend=backend, **hp)
                hm.fit(Xtr, ytr[h], sample_weight=wtr)  # mlx unweighted; lightgbm native weights
                hm.save(out_dir / f"hazard_{h}_{tag}{suffix}.pkl")
                if len(np.unique(yva[h])) > 1:
                    pva_sum[h] += hm.predict_pos(Xva)
                del hm
                # conditional-intensity model: trained ONCE (all positives are in
                # every bag, so bag 0's positive subset is the full positive set).
                if k == 0:
                    idx = np.where((ytr[h] == 1) & (ybin[h] >= 0))[0]
                    if idx.size >= 50 and np.unique(ybin[h][idx]).size >= 2:
                        im = GBDTModel(task="multiclass", backend=backend, **hp)
                        im.fit(Xtr[idx], ybin[h][idx], sample_weight=wtr[idx])
                        im.save(out_dir / f"intensity_{h}_{tag}.pkl")
                        del im
                _clear_mlx_cache()
                step += 1
                elapsed = time.time() - t0
                eta = elapsed / step * (total - step)
                print(f"  [{step}/{total}] {tag} {h:4s} bag {k} ({rows_per_bag:,} rows) "
                      f"in {_fmt(time.time()-ts)} | elapsed {_fmt(elapsed)}, ETA {_fmt(eta)}",
                      flush=True)
            used_bags += 1
            del Xtr, ytr, ybin, wtr
            _clear_mlx_cache()

        # score + calibrate the BAG-MEAN validation prediction
        for h in HAZARDS:
            if used_bags and len(np.unique(yva[h])) > 1:
                pva = pva_sum[h] / used_bags
                auc = float(roc_auc_score(yva[h], pva))
                metrics[f"p_{h}_{tag}"] = {"auc": auc,
                                           "auprc": float(average_precision_score(yva[h], pva)),
                                           "n": len(yva[h]), "pos": int(yva[h].sum()),
                                           "bags": used_bags}
                if calibrate:  # base-rate recovered here via validation weights
                    Calibrator(method=calibrator).fit(pva, yva[h], sample_weight=wva).save(
                        out_dir / f"calib_{h}_{tag}.pkl")
                print(f"   {tag} {h:4s}: mean-of-{used_bags} AUC={round(auc,3)}", flush=True)
        del Xva, yva, wva
        _clear_mlx_cache()
    return metrics
