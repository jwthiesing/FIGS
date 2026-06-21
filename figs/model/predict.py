"""Inference: run the trained models over a forecast valid time to produce
probability grids and conditional-intensity distribution grids.

Builds the feature grid exactly as in training (same column order, incl.
previous/following-hour fields when ``temporal=True``), then evaluates each
hazard model and, where available, the conditional-intensity model. Intensity
classes that were absent in training are filled with zero probability so every
hazard's distribution spans its full bin set.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def _quiet_stdout():
    """Silence stdout for the wrapped block (Herbie's '✅ Found …' banners and
    cfgrib's 'Note: Returning a list …' prints). Used around the download/assemble
    work so the only thing reaching the user is our own progress line. Warnings go
    to STDERR, so they (and our stderr diagnostics) survive this redirect. Set once
    around the whole assemble call — never toggled mid-call — so it's safe even with
    the concurrent member-fetch threads inside."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield

from ..config import (
    FIGS_DX_KM,
    HAZARDS,
    INTENSITY_BINS,
    LEAD_BANDS,
    MODELS,
    PREDICT_SMOOTH_RADII_MI,
)
from ..data import dataset
from ..data.grid import MI_TO_KM
from .calibrate import Calibrator
from .wrapper import GBDTModel


def _smooth_median(grid: np.ndarray, radii_mi=PREDICT_SMOOTH_RADII_MI) -> np.ndarray:
    """Per-cell median over Gaussian-smoothed copies of a 2-D field at the given
    radii (miles; 0 = the raw grid). Robust de-speckle that keeps real maxima."""
    from scipy.ndimage import gaussian_filter

    copies = []
    for r in radii_mi:
        if r <= 0:
            copies.append(grid)
        else:
            sigma = r * MI_TO_KM / FIGS_DX_KM       # miles -> grid cells
            copies.append(gaussian_filter(grid, sigma=sigma, mode="nearest"))
    return np.median(np.stack(copies, axis=0), axis=0).astype(np.float32)


def _postprocess(out: dict, radii_mi=PREDICT_SMOOTH_RADII_MI) -> dict:
    """Apply the smooth-median to every prediction grid IN PLACE. Probability grids
    are smoothed directly; each conditional-intensity bin is smoothed then the
    distribution is renormalized so it still sums to 1 per cell."""
    if radii_mi is None or list(radii_mi) == [0.0]:
        return out
    for h in HAZARDS:
        p = out.get(f"p_{h}")
        if p is not None and np.isfinite(p).any():
            out[f"p_{h}"] = _smooth_median(p, radii_mi)
        d = out.get(f"dist_{h}")
        if d is not None and np.isfinite(d).any():
            sm = np.stack([_smooth_median(d[b], radii_mi) for b in range(d.shape[0])], axis=0)
            s = sm.sum(axis=0, keepdims=True)
            out[f"dist_{h}"] = np.where(s > 0, sm / s, sm).astype(np.float32)  # renormalize
    return out


def _feature_matrix(valid_time, feat_cols, max_members, temporal, as_of=None):
    """(ncells, nfeat) feature matrix + grid shape, ordered to match training."""
    feats = dataset._features_for_valid(valid_time, max_members, temporal, as_of=as_of)
    ny, nx = next(iter(feats.values())).shape
    cols = [feats[c] if c in feats else np.full((ny, nx), np.nan, np.float32) for c in feat_cols]
    X = np.stack(cols, axis=0).reshape(len(feat_cols), ny * nx).T.astype(np.float32)
    return X, (ny, nx)


def _dist_to_bins(proba: np.ndarray, classes: np.ndarray, nbins: int, shape) -> np.ndarray:
    """Map predicted class probabilities to the full (nbins, ny, nx) bin stack."""
    ny, nx = shape
    dist = np.zeros((nbins, ny, nx), dtype=np.float32)
    for j, c in enumerate(np.asarray(classes).astype(int)):
        if 0 <= c < nbins:
            dist[c] = proba[:, j].reshape(ny, nx)
    return dist


def _band_tags(models_dir: Path, fxx=None) -> list[str]:
    """Every forecast hour ensembles ALL trained lead-band models (a lead-diverse
    ensemble), so this returns every band that has a model on disk (``fxx`` is
    ignored; matches both single and ``_bag*`` bagged files). Falls back to
    'pooled' if no banded models exist."""
    tags = [b.name for b in LEAD_BANDS if any(models_dir.glob(f"hazard_*_{b.name}*.pkl"))]
    return tags or ["pooled"]


def _hazard_bag_paths(models_dir: Path, h: str, tag: str):
    """All hazard model files for (h, tag): the single ``hazard_h_tag.pkl`` OR the
    bagged ``hazard_h_tag_bag*.pkl`` set (exactly one form exists)."""
    single = models_dir / f"hazard_{h}_{tag}.pkl"
    if single.exists():
        return [single]
    return sorted(models_dir.glob(f"hazard_{h}_{tag}_bag*.pkl"))


def _load_band_models(models_dir: Path, tags) -> dict:
    """Load every (hazard, band) model(s) + calibrator + intensity model ONCE. The
    hazard entry is a LIST of bag models (length 1 when not bagged)."""
    models: dict = {}
    for h in HAZARDS:
        for tag in tags:
            bag_paths = _hazard_bag_paths(models_dir, h, tag)
            if not bag_paths:
                continue
            entry = {"hazard_bags": [GBDTModel.load(p) for p in bag_paths]}
            cpath = models_dir / f"calib_{h}_{tag}.pkl"
            if cpath.exists():
                entry["calib"] = Calibrator.load(cpath)
            ipath = models_dir / f"intensity_{h}_{tag}.pkl"
            if ipath.exists():
                entry["intensity"] = GBDTModel.load(ipath)
            models[(h, tag)] = entry
    return models


def _predict_from_matrix(X: np.ndarray, shape, models: dict, tags) -> dict:
    """Evaluate the (preloaded) all-band ensemble on an assembled feature matrix.
    Per (hazard, band): average the bag models' raw p, then calibrate (the
    calibrator was fit on the bag-mean); then average those across bands. The
    conditional-intensity distribution is averaged across bands.
    Returns {'p_<h>':(ny,nx),'dist_<h>':(nbins,ny,nx)}."""
    ny, nx = shape
    out: dict[str, np.ndarray] = {}
    for h in HAZARDS:
        nbins = len(INTENSITY_BINS[h]["labels"])
        p_stack, dist_stack = [], []
        for tag in tags:
            e = models.get((h, tag))
            if e is None:
                continue
            bags = e["hazard_bags"]
            p = np.mean([m.predict_pos(X) for m in bags], axis=0)   # average bags first
            if "calib" in e:
                p = e["calib"].transform(p)                          # calibrate the bag-mean
            p_stack.append(p)
            if "intensity" in e:
                im = e["intensity"]
                dist_stack.append(_dist_to_bins(im.predict_proba(X), im.classes_, nbins, shape))
        out[f"p_{h}"] = (np.mean(p_stack, axis=0).reshape(ny, nx) if p_stack
                         else np.full((ny, nx), np.nan, np.float32))
        out[f"dist_{h}"] = (np.mean(dist_stack, axis=0) if dist_stack
                            else np.full((nbins, ny, nx), np.nan, np.float32))
    return _postprocess(out)   # smooth-median -> final predictions (all downstream uses these)


def predict_valid(valid_time, models_dir: str | Path | None = None, *,
                  fxx=None, max_members: int = 6, temporal: bool = False, as_of=None) -> dict:
    """Predict hazard probabilities and conditional-intensity distributions for a
    single valid time (all-band ensemble, calibrated). ``as_of`` makes the
    time-lagged ensemble real-time-faithful. Returns
    {'p_<h>':(ny,nx),'dist_<h>':(nbins,ny,nx)}."""
    models_dir = Path(models_dir) if models_dir else MODELS
    feat_cols = json.loads((models_dir / "feature_cols.json").read_text())
    X, shape = _feature_matrix(valid_time, feat_cols, max_members, temporal, as_of=as_of)
    tags = _band_tags(models_dir, fxx)
    return _predict_from_matrix(X, shape, _load_band_models(models_dir, tags), tags)


# Per-worker state for the multiprocess forecast (set once per process by the
# pool initializer, so models/feat_cols aren't re-loaded or re-shipped per hour).
_WORKER: dict = {}


def _forecast_worker_init(models_dir_str, feat_cols, tags, max_members, temporal, run):
    _WORKER.update(models_dir=Path(models_dir_str), feat_cols=feat_cols, tags=tags,
                   max_members=max_members, temporal=temporal, run=run,
                   models=_load_band_models(Path(models_dir_str), tags))


def _forecast_worker(fxx):
    """Assemble (download + regrid + features) AND predict one forecast hour, in
    this worker process — so BOTH the I/O and the CPU-bound feature build use a
    full core. Returns only the small prediction grids (cheap to ship back)."""
    from datetime import timedelta

    f = int(fxx)
    vt = _WORKER["run"] + timedelta(hours=f)
    print(f"[predict] f{f:02d}: downloading + preprocessing…", flush=True)
    with _quiet_stdout():                          # mute Herbie/cfgrib chatter, keep our lines
        X, shape = _feature_matrix(vt, _WORKER["feat_cols"], _WORKER["max_members"],
                                   _WORKER["temporal"], as_of=_WORKER["run"])
    print(f"[predict] f{f:02d}: preprocessed ✓ — predicting", flush=True)
    out = _predict_from_matrix(X, shape, _WORKER["models"], _WORKER["tags"])
    del X
    return f, out


def predict_forecast(run, fxx_list, models_dir=None, *, max_members=6, temporal=False,
                     workers: int = 4) -> dict:
    """Predict for each forecast hour of a run. Returns {fxx: predict_valid(...)}.

    Each forecast hour's full pipeline — the time-lagged HRRR **download** AND the
    CPU-bound **feature build** AND model eval — runs in its own worker PROCESS, so
    both the I/O and the NumPy processing scale across cores (threads would serialize
    the feature build on the GIL). Only the small prediction grids cross the process
    boundary. ``workers`` processes run at once (peak RAM ≈ workers × one hour's
    assembly); ``workers=1`` runs serially in-process. Recycle cadence is set by
    ``FIGS_MAX_TASKS_PER_CHILD`` (eccodes/cfgrib C-side memory, as in the builder)."""
    import os
    from datetime import timedelta

    models_dir = Path(models_dir) if models_dir else MODELS
    feat_cols = json.loads((models_dir / "feature_cols.json").read_text())
    tags = _band_tags(models_dir, None)            # every hour ensembles all bands
    fxxs = [int(f) for f in fxx_list]
    workers = max(1, int(workers))

    n = len(fxxs)
    if workers == 1:                               # serial, in-process (no pool overhead)
        models = _load_band_models(models_dir, tags)
        out: dict = {}
        for done, fxx in enumerate(fxxs, 1):
            vt = run + timedelta(hours=fxx)
            print(f"[predict] f{fxx:02d}: downloading + preprocessing…", flush=True)
            with _quiet_stdout():                  # mute Herbie/cfgrib chatter, keep our lines
                X, shape = _feature_matrix(vt, feat_cols, max_members, temporal, as_of=run)
            print(f"[predict] f{fxx:02d}: preprocessed ✓ — predicting", flush=True)
            out[fxx] = _predict_from_matrix(X, shape, models, tags)
            del X
            print(f"[predict] {done}/{n} ({100 * done // n}%) complete — f{fxx:02d} done", flush=True)
        return out

    from concurrent.futures import ProcessPoolExecutor, as_completed

    max_tasks = int(os.environ.get("FIGS_MAX_TASKS_PER_CHILD", "16"))
    pool_kwargs = {"max_workers": workers,
                   "initializer": _forecast_worker_init,
                   "initargs": (str(models_dir), feat_cols, tags, max_members, temporal, run)}
    try:  # max_tasks_per_child added in Python 3.11 (recycle to free eccodes memory)
        ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1).shutdown()
        pool_kwargs["max_tasks_per_child"] = max_tasks
    except TypeError:
        pass
    out = {}
    with ProcessPoolExecutor(**pool_kwargs) as ex:
        # as_completed (not ex.map) so progress prints the instant EACH hour finishes,
        # rather than batching in submission order when an early hour lags behind.
        futs = [ex.submit(_forecast_worker, f) for f in fxxs]
        for done, fut in enumerate(as_completed(futs), 1):
            fxx, pred = fut.result()
            out[fxx] = pred
            print(f"[predict] {done}/{n} ({100 * done // n}%) complete — f{fxx:02d} done", flush=True)
    return out


def predict_or_load(run, fxx_list, models_dir=None, *, max_members=6, temporal=False,
                    workers: int = 4, cache: bool = True, write: bool = True,
                    out_path=None) -> dict:
    """Predictions for ``run`` with on-disk caching. If a netCDF for this run exists
    and covers every requested fxx, load it (no download); otherwise run
    ``predict_forecast`` and (``write``) save the netCDF — so the same file is both
    the cache and the persistent output. Returns ``{fxx: {...}}``."""
    from ..products import netcdf

    fxxs = [int(f) for f in fxx_list]
    path = netcdf.predictions_path(run, out_path, fxx=fxxs)
    if cache and Path(path).exists():
        try:
            cached = netcdf.read_predictions(path)
            if all(f in cached for f in fxxs):
                print(f"loaded cached predictions: {path}", flush=True)
                return {f: cached[f] for f in fxxs}
        except Exception as e:  # noqa: BLE001 - corrupt/old cache -> recompute
            print(f"[warn] cache read failed ({e}); recomputing", flush=True)
    preds = predict_forecast(run, fxxs, models_dir=models_dir, max_members=max_members,
                             temporal=temporal, workers=workers)
    if write:
        netcdf.write_predictions(preds, run, path)
        print(f"wrote predictions: {path}", flush=True)
    return preds
