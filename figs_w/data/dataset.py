"""FIGS-W training-matrix builder. Mirrors ``figs.data.dataset`` (subsample-with-
reweighting, parquet part-files) but assembles the single-run wildfire state +
static geography and joins the wildfire labels. Reuses the FIGS part-file helpers
and the generic streaming readers for training."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# reuse FIGS's generic part-file + streaming-read machinery
from figs.data.dataset import (  # noqa: F401
    _dataset_target, _disk_free_gb, _fmt_eta, _parts_dir, _progress, _write_part,
    feature_columns, read_band, read_dataset, read_split,
)
from figs.data import grid

from .. import config as C
from ..features import assemble
from . import labels as labels_w
from . import state as state_mod
from . import static as static_mod

META_COLS = ["valid_time", "fxx", "iy", "ix", "lat", "lon", "split", "weight"]
LABEL_COLS = list(C.LABEL_FIELDS)   # wildfire (occurrence), wildfire_size (raw acres)


def _features_for_valid(run: datetime, fxx: int, *, cached_only: bool = False) -> dict:
    st = state_mod.assemble_state(run, fxx, cached_only=cached_only)
    return assemble.compute_features(st["iso"], st["sfc"], static_mod.load_static_fields())


def build_row_table(valid_time: datetime, *, run: datetime, fxx: int,
                    neg_keep: float = 0.02, rng_seed: int = 0) -> pd.DataFrame:
    """One valid hour → subsampled, reweighted feature/label rows for FIGS-W."""
    feats = _features_for_valid(run, fxx)
    labels = labels_w.build_labels(valid_time)
    names = sorted(feats)
    F = np.stack([feats[n].astype(np.float32) for n in names], axis=0)
    ny, nx = F.shape[1:]

    positive = labels["wildfire"].astype(bool)
    rng = np.random.default_rng(rng_seed + int(valid_time.timestamp()) % 100000)
    keep_prob = np.where(positive, 1.0, neg_keep)
    keep = positive | (rng.random((ny, nx)) < neg_keep)
    iy, ix = np.where(keep)

    lat, lon = grid.figs_latlon()
    data = {n: F[i][iy, ix] for i, n in enumerate(names)}
    for k in LABEL_COLS:
        data[k] = labels[k][iy, ix]
    data["valid_time"] = np.full(iy.shape, valid_time.replace(tzinfo=timezone.utc))
    data["fxx"] = np.full(iy.shape, int(fxx), dtype=np.int16)
    data["iy"] = iy.astype(np.int16); data["ix"] = ix.astype(np.int16)
    data["lat"] = lat[iy, ix].astype(np.float32); data["lon"] = lon[iy, ix].astype(np.float32)
    data["split"] = np.full(iy.shape, C.split_for_date(valid_time))
    data["weight"] = (1.0 / keep_prob[iy, ix]).astype(np.float32)
    return pd.DataFrame(data)


def _build_worker(task):
    """Pool entry point: build one (run, fxx) sample's row table."""
    run, fxx, neg_keep = task
    return build_row_table(run + timedelta(hours=int(fxx)), run=run, fxx=fxx, neg_keep=neg_keep)


def build_dataset_for_runs(run_fxx_pairs, out_path=None, *, neg_keep: float = 0.02,
                           flush_every: int = 10, min_free_gb: float = 50.0,
                           workers: int = 1) -> str:
    """Build the FIGS-W matrix from (run, fxx) samples → parquet part-files.

    ``workers`` > 1 builds that many samples concurrently in separate processes
    (each does its own HRRR fetch + feature build); the fire catalog is disk-cached
    so workers don't re-query NIFC. ``workers`` = 1 runs serially in-process."""
    import os
    import time

    out_path = out_path or str(C.PROCESSED / "figs_w.parquet")
    parts_dir = _parts_dir(out_path)
    total, t0 = len(run_fxx_pairs), time.time()
    frames, part_idx, rows = [], 0, 0
    workers = max(1, int(workers))

    def flush():
        nonlocal frames, part_idx, rows
        if frames:
            rows += sum(len(f) for f in frames if f is not None)
            part_idx = _write_part(frames, parts_dir, part_idx)
            frames = []

    if workers == 1:
        for i, (run, fxx) in enumerate(run_fxx_pairs):
            if _disk_free_gb() < min_free_gb:
                print(f"[stop] free disk < {min_free_gb} GB after {i}/{total}", flush=True)
                break
            vt = run + timedelta(hours=int(fxx))
            try:
                frames.append(build_row_table(vt, run=run, fxx=fxx, neg_keep=neg_keep))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {run:%Y-%m-%d %H}Z f{fxx:02d} failed: {e}", flush=True)
            _progress(i + 1, total, t0, f"{run:%Y-%m-%d %H}Z f{fxx:02d}")
            if (i + 1) % flush_every == 0:
                flush()
        flush()
        print(f"wrote {rows} rows across {part_idx} part-files -> {parts_dir}", flush=True)
        return out_path

    # parallel: keep ~2×workers tasks in flight (bounded memory); recycle children
    # so eccodes/cfgrib C-side memory is released (as in FIGS's build).
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    pairs = iter(run_fxx_pairs)
    inflight, done, stop = {}, 0, False
    max_tasks = int(os.environ.get("FIGS_MAX_TASKS_PER_CHILD", "16"))
    pool_kwargs = {"max_workers": workers}
    try:
        ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1).shutdown()
        pool_kwargs["max_tasks_per_child"] = max_tasks
    except TypeError:
        pass
    with ProcessPoolExecutor(**pool_kwargs) as ex:
        def submit_one() -> bool:
            try:
                run, fxx = next(pairs)
            except StopIteration:
                return False
            inflight[ex.submit(_build_worker, (run, int(fxx), neg_keep))] = (run, fxx)
            return True

        for _ in range(workers * 2):
            if not submit_one():
                break
        while inflight and not stop:
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                run, fxx = inflight.pop(fut)
                done += 1
                try:
                    frames.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] {run:%Y-%m-%d %H}Z f{fxx:02d} failed: {e}", flush=True)
                _progress(done, total, t0, f"{run:%Y-%m-%d %H}Z f{fxx:02d}")
                if done % flush_every == 0:
                    flush()
                if _disk_free_gb() < min_free_gb:
                    print(f"[stop] free disk < {min_free_gb} GB — cancelling remaining", flush=True)
                    for f in inflight:
                        f.cancel()
                    inflight.clear(); stop = True
                    break
                submit_one()
    flush()
    print(f"wrote {rows} rows across {part_idx} part-files -> {parts_dir}", flush=True)
    return out_path


def _relabel_part(path: str) -> tuple[int, dict]:
    """Recompute ONLY the label columns for one part-file from the (corrected) fire
    catalog and rewrite it in place. No HRRR / feature recompute — labels depend only
    on valid_time + the stored (iy, ix). Returns (rows, positive counts)."""
    df = pd.read_parquet(path)
    cols = {c: np.zeros(len(df), dtype=(np.int8 if c == "wildfire" else np.float32))
            for c in LABEL_COLS}
    if "wildfire_size" in cols:
        cols["wildfire_size"][:] = np.nan
    for vt, grp in df.groupby("valid_time", sort=False):
        vt_dt = pd.Timestamp(vt).to_pydatetime()
        if vt_dt.tzinfo is None:
            vt_dt = vt_dt.replace(tzinfo=timezone.utc)
        labs = labels_w.build_labels(vt_dt)
        iy = grp["iy"].to_numpy(); ix = grp["ix"].to_numpy(); idx = grp.index.to_numpy()
        for c in LABEL_COLS:
            cols[c][idx] = labs[c][iy, ix]
    for c in LABEL_COLS:
        df[c] = cols[c]
    df.to_parquet(path, index=False)
    pos = {"wildfire": int((df["wildfire"] == 1).sum()),
           "size>0": int((df["wildfire_size"] > 0).sum())}
    return len(df), pos


def augment_labels(path: str, *, workers: int = 1) -> str:
    """Recompute the label columns of an EXISTING dataset IN PLACE from the corrected
    fire catalog — NO preprocessing/HRRR rebuild (features untouched). Use this after
    a catalog/label-logic fix (e.g. the NIFC field + RX corrections).
    Primes the catalog first so parallel workers reuse the on-disk cache."""
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    target = _dataset_target(path)
    parts = sorted(Path(target).glob("*.parquet")) if Path(target).is_dir() else [Path(target)]
    if not parts:
        raise FileNotFoundError(f"no parquet part-files under {target}")
    # prime the catalog over the dataset's valid-time span (main process; workers reuse disk cache)
    vt = read_dataset(path, columns=["valid_time"])["valid_time"]
    from . import fire_reports
    fire_reports.prime_catalog(pd.Timestamp(vt.min()).to_pydatetime(),
                               pd.Timestamp(vt.max()).to_pydatetime())
    workers = max(1, int(workers)); t0 = time.time()
    print(f"relabeling {len(parts)} part-file(s) ({workers} worker(s))", flush=True)
    tot = {"wildfire": 0, "size>0": 0}; nrows = 0

    def _accum(res):
        nonlocal nrows
        n, pos = res; nrows += n
        for k in tot: tot[k] += pos[k]

    paths = [str(p) for p in parts]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_relabel_part, paths), 1):
                _accum(res); print(f"  [{i}/{len(parts)}] {_fmt_eta(time.time()-t0)}", flush=True)
    else:
        for i, p in enumerate(paths, 1):
            _accum(_relabel_part(p)); print(f"  [{i}/{len(parts)}] {_fmt_eta(time.time()-t0)}", flush=True)
    print(f"relabeled {nrows} rows | positives: {tot} -> {target}", flush=True)
    return str(target)


def fire_valid_hours(start: datetime, end: datetime, min_fires: int = 1) -> list[datetime]:
    """UTC hours in [start, end] with >= ``min_fires`` ACTIVE fires (ongoing multi-
    day fires included, not just ignitions). Primes the fire catalog first."""
    from . import fire_reports

    fire_reports.prime_catalog(start, end)
    return fire_reports.active_fire_hours(start, end, min_fires=min_fires)
