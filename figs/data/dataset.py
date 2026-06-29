"""Training-dataset builder: ensemble state -> features -> labels -> subsampled
rows -> parquet.

For each selected valid hour we assemble the time-lagged ensemble, compute the
feature grid (optionally with previous/following-hour fields), join the
neighborhood labels, flatten to per-cell rows, and apply nadocast-style
subsample-with-reweighting: keep all near-report cells, randomly downsample
far-from-report cells, and weight each kept row by 1/retention so the totals
stay unbiased.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import HAZARDS, HRRR_CACHE, PROCESSED, split_for_date
from ..features import assemble
from . import ensemble, grid
from . import labels as labels_mod

META_COLS = ["valid_time", "fxx", "iy", "ix", "lat", "lon", "split", "weight"]
LABEL_COLS = ([h for h in HAZARDS] + [f"{h}_sig" for h in HAZARDS]
              + [f"{h}_bin" for h in HAZARDS] + [f"{h}_pib" for h in HAZARDS])


def _disk_free_gb(path=HRRR_CACHE) -> float:
    return shutil.disk_usage(str(path)).free / 1e9


def _fmt_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _progress(done: int, total: int, t0: float, label: str):
    """Log progress with elapsed/ETA and free disk."""
    elapsed = time.time() - t0
    rate = elapsed / max(done, 1)
    eta = rate * (total - done)
    pct = 100 * done // max(total, 1)
    print(f"  [{done}/{total}] ({pct}%) {label} | {_fmt_eta(elapsed)} elapsed, "
          f"ETA {_fmt_eta(eta)} (~{rate:.0f}s/sample) | free disk {_disk_free_gb():.0f} GB",
          flush=True)


def _parts_path(out_path: str):
    """Path of the parquet part-file directory for ``out_path`` (no mkdir)."""
    from pathlib import Path

    base = str(Path(out_path))
    base = base[:-8] if base.endswith(".parquet") else base
    return Path(base + "_parts")


def _parts_dir(out_path: str):
    """Like ``_parts_path`` but creates the directory (for writing)."""
    d = _parts_path(out_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dataset_target(path: str):
    """Resolve a dataset path to the thing pandas should read: a part-file
    directory if one exists, else the single file/dir as given."""
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        return p
    pd_dir = _parts_path(path)
    if pd_dir.is_dir() and any(pd_dir.glob("*.parquet")):
        return pd_dir
    return p


def _write_part(frames: list, parts_dir, idx: int) -> int:
    """Concatenate a small batch of sample frames into one part file, return the
    next part index. Frees nothing here; caller clears ``frames``."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return idx
    pd.concat(frames, ignore_index=True).to_parquet(parts_dir / f"part-{idx:05d}.parquet", index=False)
    return idx + 1


def read_dataset(path: str, *, filters=None, columns=None) -> pd.DataFrame:
    """Read a dataset (single parquet file or directory of part files). ``filters``
    (pyarrow predicate-pushdown, e.g. ``[('fxx','>=',13),('fxx','<=',24)]``) and
    ``columns`` let callers load only the rows/columns they need — used to stream
    one lead band at a time so the full set never has to fit in RAM."""
    return pd.read_parquet(_dataset_target(path), filters=filters, columns=columns)


def read_band(path: str, *, feature_cols, aux_cols, filters=None,
              train_cap=None, val_cap=None, seed: int = 0):
    """Read a band's train/validation rows **streaming part-file by part-file into
    pre-allocated NumPy matrices** — the feature data is never held as a pandas
    DataFrame and there is no concat transient, so peak RAM ≈ the output matrices
    plus a single part-file (vs ~1.5–2× that for the old DataFrame path).

    Two passes over the (shuffled) parts: pass 1 reads only the tiny ``split``
    column to count rows (so the output size is known and capped); pass 2 reads
    the needed columns and copies each part straight into the pre-allocated array,
    freeing the part immediately. Caps are honored exactly (the part that would
    overflow a cap is randomly subsampled).

    Returns ``(Xtr, aux_tr, Xva, aux_va)``: float32 feature matrices in
    ``feature_cols`` order, plus small DataFrames of ``aux_cols`` (labels +
    'weight') — those are few columns, so keeping them as frames is cheap."""
    import random
    from pathlib import Path

    target = _dataset_target(path)
    parts = sorted(Path(target).glob("*.parquet")) if Path(target).is_dir() else [Path(target)]
    random.Random(seed).shuffle(parts)
    F = len(feature_cols)

    # --- pass 1: count train/val rows (read only 'split'); stop once caps met ---
    sel, ntr, nva = [], 0, 0
    for p in parts:
        s = pd.read_parquet(p, columns=["split"], filters=filters)["split"]
        sel.append(p)
        ntr += int((s == "train").sum())
        nva += int((s == "validation").sum())
        if train_cap and val_cap and ntr >= train_cap and nva >= val_cap:
            break
    n_tr = min(ntr, train_cap) if train_cap else ntr
    n_va = min(nva, val_cap) if val_cap else nva

    # --- pre-allocate outputs; pass 2 fills them part-by-part ---
    Xtr = np.empty((n_tr, F), np.float32)
    Xva = np.empty((n_va, F), np.float32)
    aux_tr_frames, aux_va_frames = [], []
    rng = np.random.default_rng(seed)
    read_cols = list(dict.fromkeys(list(feature_cols) + list(aux_cols) + ["split"]))
    otr = ova = 0

    def _fill(sub, X, off, cap, aux_frames):
        take = min(len(sub), cap - off)
        if take <= 0:
            return 0
        if take < len(sub):                      # this part would overflow the cap
            idx = rng.choice(len(sub), size=take, replace=False)
            sub = sub.iloc[idx]
        for j, c in enumerate(feature_cols):
            X[off:off + take, j] = sub[c].to_numpy(dtype=np.float32, copy=False)
        aux_frames.append(sub[list(aux_cols)].reset_index(drop=True))
        return take

    for p in sel:
        if otr >= n_tr and ova >= n_va:
            break
        d = pd.read_parquet(p, columns=read_cols, filters=filters)
        if otr < n_tr:
            otr += _fill(d[d["split"] == "train"], Xtr, otr, n_tr, aux_tr_frames)
        if ova < n_va:
            ova += _fill(d[d["split"] == "validation"], Xva, ova, n_va, aux_va_frames)
        del d
    import pandas as _pd
    aux_tr = (_pd.concat(aux_tr_frames, ignore_index=True) if aux_tr_frames
              else _pd.DataFrame(columns=list(aux_cols)))
    aux_va = (_pd.concat(aux_va_frames, ignore_index=True) if aux_va_frames
              else _pd.DataFrame(columns=list(aux_cols)))
    return Xtr[:otr], aux_tr, Xva[:ova], aux_va


def read_split(path: str, *, feature_cols, aux_cols, split: str = "train", filters=None,
               cap=None, seed: int = 0, bag=None, n_bags: int = 1, positive_cols=None):
    """Read ONE split's rows into a pre-allocated float32 matrix + small aux frame,
    part-by-part (peak ≈ output + one part), in ``feature_cols`` order.

    BAGGING (train split only): when ``n_bags > 1`` and ``bag`` is given, every bag
    keeps **all positive rows** (any column in ``positive_cols`` > 0) plus a
    **disjoint 1/n_bags fold of the negatives** (fold assigned deterministically
    per part, independent of ``bag``, so the K bags partition the negatives). This
    lets K small models collectively cover all stored negatives — coverage beyond a
    single model's RAM ceiling — while every bag sees the full positive signal."""
    import random
    from pathlib import Path

    feature_cols, aux_cols = list(feature_cols), list(aux_cols)
    bagging = bool(n_bags and n_bags > 1 and bag is not None
                   and split == "train" and positive_cols)
    pcols = list(positive_cols) if bagging else []
    target = _dataset_target(path)
    parts = sorted(Path(target).glob("*.parquet")) if Path(target).is_dir() else [Path(target)]
    random.Random(seed).shuffle(parts)
    F = len(feature_cols)

    # --- plan: positions (into each part's split-subset, file order) to keep ---
    plan, total = [], 0
    for i, p in enumerate(parts):
        d = pd.read_parquet(p, columns=list(dict.fromkeys(["split"] + pcols)), filters=filters)
        m = (d["split"] == split).to_numpy()
        n_split = int(m.sum())
        if n_split == 0:
            continue
        if bagging:
            sub = d.loc[m, pcols]
            ispos = np.zeros(n_split, bool)
            for c in pcols:
                ispos |= sub[c].to_numpy() > 0
            fold = np.random.default_rng([seed, i]).integers(0, n_bags, n_split)  # bag-independent
            pos = np.where(ispos | (~ispos & (fold == bag)))[0]
        else:
            pos = np.arange(n_split)
        if cap is not None and total + len(pos) > cap:                # clip the overflowing part
            take = cap - total
            pos = np.sort(np.random.default_rng([seed, i, 1]).choice(len(pos), size=take, replace=False))
            plan.append((p, pos)); total += take
            break
        plan.append((p, pos)); total += len(pos)
        if cap is not None and total >= cap:
            break

    # --- pre-allocate + fill ---
    X = np.empty((total, F), np.float32)
    aux_frames, off = [], 0
    read_cols = list(dict.fromkeys(feature_cols + aux_cols + ["split"]))
    for p, pos in plan:
        d = pd.read_parquet(p, columns=read_cols, filters=filters)
        sub = d[d["split"] == split].iloc[pos]
        n = len(sub)
        for j, c in enumerate(feature_cols):
            X[off:off + n, j] = sub[c].to_numpy(dtype=np.float32, copy=False)
        aux_frames.append(sub[aux_cols].reset_index(drop=True))
        off += n
        del d
    import pandas as _pd
    aux = (_pd.concat(aux_frames, ignore_index=True) if aux_frames
           else _pd.DataFrame(columns=aux_cols))
    return X[:off], aux


def dataset_feature_columns(path: str) -> list[str]:
    """Feature-column names from the dataset schema (cheap — reads no row data)."""
    import pyarrow.parquet as pq
    from pathlib import Path

    target = _dataset_target(path)
    src = sorted(Path(target).glob("*.parquet"))[0] if Path(target).is_dir() else target
    drop = set(META_COLS) | set(LABEL_COLS)
    return [c for c in pq.read_schema(src).names if c not in drop]


def severe_valid_hours(start: datetime, end: datetime, *, min_reports: int = 1) -> list[datetime]:
    """Top-of-hour UTC valid times in [start, end] with >= min_reports severe
    reports (across any hazard), based on the combined report DB."""
    from . import reports as reports_mod

    hours: dict[datetime, int] = {}
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        df = reports_mod.reports_for_day(day)
        if not df.empty:
            for t in df["time"]:
                h = t.replace(minute=0, second=0, microsecond=0)
                if start <= h <= end:
                    hours[h] = hours.get(h, 0) + 1
        day += timedelta(days=1)
    return sorted(h for h, c in hours.items() if c >= min_reports)


def _features_for_valid(valid_time: datetime, max_members: int, temporal: bool,
                        as_of: datetime | None = None) -> dict:
    """Feature dict for a valid time; with ``temporal`` also merges t-1 / t+1
    feature fields (suffixed ``_prev`` / ``_next``) and the ensemble prob fields.
    ``as_of`` (issuance/primary-run time) caps the time-lagged ensemble for
    real-time-faithful samples; the t-1/t+1 fields are capped at the same time."""
    def one(vt):
        inp = ensemble.assemble_inputs(vt, max_members, as_of=as_of)
        # prob_fields are merged AND spatially smoothed inside compute_features
        return assemble.compute_features(inp["iso"], inp["sfc"], prob_fields=inp["prob_fields"])

    feats = one(valid_time)
    if temporal:
        prev = one(valid_time - timedelta(hours=1))
        nxt = one(valid_time + timedelta(hours=1))
        for k, v in prev.items():
            feats[f"{k}_prev"] = v
        for k, v in nxt.items():
            feats[f"{k}_next"] = v
    return feats


def build_row_table(
    valid_time: datetime,
    *,
    max_members: int = 6,
    temporal: bool = False,
    neg_keep: float = 0.025,
    rng_seed: int = 0,
    as_of: datetime | None = None,
    fxx: int | None = None,
) -> pd.DataFrame:
    """One valid hour -> a DataFrame of subsampled, reweighted feature/label rows.

    ``as_of`` (primary-run/issuance time) makes the time-lagged ensemble
    real-time-faithful; ``fxx`` (forecast hour) is recorded for lead-time banding.
    """
    feats = _features_for_valid(valid_time, max_members, temporal, as_of=as_of)
    labels = labels_mod.build_labels(valid_time)
    names = sorted(feats)
    F = np.stack([feats[n].astype(np.float32) for n in names], axis=0)  # (Nf, ny, nx)
    ny, nx = F.shape[1:]

    positive = np.zeros((ny, nx), dtype=bool)
    for h in HAZARDS:
        positive |= labels[h].astype(bool)

    rng = np.random.default_rng(rng_seed + int(valid_time.timestamp()) % 100000)
    keep_prob = np.where(positive, 1.0, neg_keep)
    draw = rng.random((ny, nx))
    keep = positive | (draw < neg_keep)
    iy, ix = np.where(keep)

    lat, lon = grid.figs_latlon()
    data = {n: F[i][iy, ix] for i, n in enumerate(names)}
    data.update({h: labels[h][iy, ix] for h in HAZARDS})
    data.update({f"{h}_sig": labels[f"{h}_sig"][iy, ix] for h in HAZARDS})
    data.update({f"{h}_bin": labels[f"{h}_bin"][iy, ix] for h in HAZARDS})
    data.update({f"{h}_pib": labels[f"{h}_pib"][iy, ix] for h in HAZARDS})
    data["valid_time"] = np.full(iy.shape, valid_time.replace(tzinfo=timezone.utc))
    if fxx is not None:
        data["fxx"] = np.full(iy.shape, int(fxx), dtype=np.int16)
    data["iy"] = iy.astype(np.int16)
    data["ix"] = ix.astype(np.int16)
    data["lat"] = lat[iy, ix].astype(np.float32)
    data["lon"] = lon[iy, ix].astype(np.float32)
    data["split"] = np.full(iy.shape, split_for_date(valid_time))
    data["weight"] = (1.0 / keep_prob[iy, ix]).astype(np.float32)
    return pd.DataFrame(data)


def build_dataset(
    valid_hours: list[datetime],
    out_path: str | None = None,
    *,
    max_members: int = 6,
    temporal: bool = False,
    neg_keep: float = 0.025,
    min_free_gb: float = 50.0,
    flush_every: int = 25,
) -> str:
    """Build a parquet matrix over valid hours using the HINDCAST ensemble
    (nearest f01..f06, no issuance cap). Convenient for quick experiments, but for
    train/serve-consistent training use ``build_dataset_for_runs`` (issuance-capped,
    matches how predictions are issued). Includes the same disk-guard / progress /
    checkpoint behavior."""
    if out_path is None:
        out_path = str(PROCESSED / "dataset.parquet")
    parts_dir = _parts_dir(out_path)
    frames, t0, part_idx, rows = [], time.time(), 0, 0
    total = len(valid_hours)
    for i, vt in enumerate(valid_hours):
        free = _disk_free_gb()
        if free < min_free_gb:
            print(f"[stop] free disk {free:.0f} GB < min {min_free_gb} GB — "
                  f"stopping after {i}/{total} hours", flush=True)
            break
        try:
            frames.append(
                build_row_table(vt, max_members=max_members, temporal=temporal, neg_keep=neg_keep)
            )
        except Exception as e:  # noqa: BLE001 - skip hours with fetch failures, keep going
            print(f"[warn] {vt:%Y-%m-%d %H}Z failed: {e}", flush=True)
        _progress(i + 1, total, t0, f"{vt:%Y-%m-%d %H}Z")
        if (i + 1) % flush_every == 0 and frames:
            rows += sum(len(f) for f in frames if f is not None)
            part_idx = _write_part(frames, parts_dir, part_idx)
            frames = []
    if frames:
        rows += sum(len(f) for f in frames if f is not None)
        part_idx = _write_part(frames, parts_dir, part_idx)
    print(f"wrote {rows} rows across {part_idx} part-files -> {parts_dir}", flush=True)
    return out_path


def build_dataset_for_runs(
    run_fxx_pairs: list[tuple[datetime, int]],
    out_path: str | None = None,
    *,
    max_members: int = 6,
    temporal: bool = False,
    neg_keep: float = 0.025,
    min_free_gb: float = 50.0,
    flush_every: int = 25,
    workers: int = 1,
) -> str:
    """Build a parquet matrix over (primary_run, fxx) samples using the
    issuance-capped (real-time-faithful) time-lagged ensemble. This is the
    training-matrix builder that matches how predictions are issued.

    ``workers`` > 1 processes that many samples concurrently in separate
    processes (each still downloads its members in parallel threads), to push
    more simultaneous S3 streams when bandwidth has headroom. Stops cleanly
    (writing what's accumulated) if free disk drops below ``min_free_gb``;
    checkpoints the parquet every ``flush_every`` completed samples.
    """
    if out_path is None:
        out_path = str(PROCESSED / "dataset.parquet")
    parts_dir = _parts_dir(out_path)
    total = len(run_fxx_pairs)
    t0 = time.time()
    frames, part_idx, rows = [], 0, 0

    def flush():
        nonlocal frames, part_idx, rows
        if frames:
            rows += sum(len(f) for f in frames if f is not None)
            part_idx = _write_part(frames, parts_dir, part_idx)
            frames = []  # free memory

    if workers <= 1:
        for i, (run, fxx) in enumerate(run_fxx_pairs):
            if _disk_free_gb() < min_free_gb:
                print(f"[stop] free disk < {min_free_gb} GB — stopping after {i}/{total}", flush=True)
                break
            vt = run + timedelta(hours=int(fxx))
            try:
                frames.append(build_row_table(
                    vt, max_members=max_members, temporal=temporal, neg_keep=neg_keep,
                    as_of=run, fxx=fxx))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {run:%Y-%m-%d %H}Z f{fxx:02d} failed: {e}", flush=True)
            _progress(i + 1, total, t0, f"{run:%Y-%m-%d %H}Z f{fxx:02d}")
            if (i + 1) % flush_every == 0:
                flush()
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        pairs_iter = iter(run_fxx_pairs)
        inflight: dict = {}
        done = 0

        # Recycle each worker after N tasks so eccodes/cfgrib C-side memory (which
        # ds.close() does NOT reclaim) is handed back to the OS. Tune with
        # FIGS_MAX_TASKS_PER_CHILD; lower = tighter memory, slightly more respawns.
        max_tasks = int(os.environ.get("FIGS_MAX_TASKS_PER_CHILD", "16"))
        pool_kwargs = {"max_workers": workers}
        try:  # max_tasks_per_child added in Python 3.11
            ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1).shutdown()
            pool_kwargs["max_tasks_per_child"] = max_tasks
        except TypeError:
            print("[warn] max_tasks_per_child unavailable; workers won't recycle", flush=True)

        with ProcessPoolExecutor(**pool_kwargs) as ex:
            def submit_one() -> bool:
                try:
                    run, fxx = next(pairs_iter)
                except StopIteration:
                    return False
                vt = run + timedelta(hours=int(fxx))
                fut = ex.submit(build_row_table, vt, max_members=max_members,
                                temporal=temporal, neg_keep=neg_keep, as_of=run, fxx=fxx)
                inflight[fut] = (run, fxx)
                return True

            # keep ~2x workers tasks in flight (bounded memory; results freed after flush)
            for _ in range(workers * 2):
                if not submit_one():
                    break
            stop = False
            while inflight and not stop:
                finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    run, fxx = inflight.pop(fut)  # drop our ref to the future + its result
                    done += 1
                    try:
                        frames.append(fut.result())
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] {run:%Y-%m-%d %H}Z f{fxx:02d} failed: {e}", flush=True)
                    del fut  # ensure the cached result can be GC'd after the next flush
                    _progress(done, total, t0, f"{run:%Y-%m-%d %H}Z f{fxx:02d}")
                    if done % flush_every == 0:
                        flush()
                    if _disk_free_gb() < min_free_gb:
                        print(f"[stop] free disk < {min_free_gb} GB — cancelling remaining",
                              flush=True)
                        for f in inflight:
                            f.cancel()
                        inflight.clear()
                        stop = True
                        break
                    submit_one()  # refill the pipeline
    flush()
    print(f"wrote {rows} rows across {part_idx} part-files -> {parts_dir}", flush=True)
    return out_path


def _added_columns_for_group(valid_time: datetime, fxx: int, iy: np.ndarray, ix: np.ndarray,
                             max_members: int, temporal: bool):
    """Compute the newly-added feature families for ONE (valid_time, fxx) sample and
    sample them at the stored cells. Re-assembles the ensemble from the cached GRIB
    (``as_of = valid_time − fxx``) — no re-download — and runs only
    ``assemble.added_features`` (not the full ~5k feature set).

    Returns ``(names, block)``: the sorted column names and a single contiguous
    ``(len(iy), n_features)`` float32 array. One array (not ~980 separate arrays)
    keeps the cross-process pickle/IPC and the main-process merge cheap — otherwise
    shipping a dict-of-arrays per group bottlenecks the parent and starves workers."""
    as_of = valid_time - timedelta(hours=int(fxx))

    def one(vt):
        # only the MAIN member's iso/sfc are needed for lapse/boundary features, and
        # they're read cache-only (no remote Herbie probe) since the dataset's GRIB
        # subsets are already downloaded — far fewer file checks than a full assemble.
        iso15, sfc15 = ensemble.main_member_iso_sfc(vt, max_members, as_of=as_of,
                                                    cached_only=True)
        return assemble.added_features(iso15, sfc15)

    grids = one(valid_time)
    if temporal:                                   # match a temporally-built parquet
        for suf, dt in (("_prev", -1), ("_next", 1)):
            for k, g in one(valid_time + timedelta(hours=dt)).items():
                grids[f"{k}{suf}"] = g
    names = sorted(grids)
    block = np.empty((len(iy), len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        block[:, j] = grids[n][iy, ix]
    return names, block


def _augment_worker(task):
    """Pool entry point: unpack a group task and return ``(names, block)`` (one
    contiguous array). A failed group (e.g. a GRIB subset missing from the cache)
    returns ``(None, None)`` so those rows stay NaN rather than aborting the augment."""
    vt, fxx, iy, ix, max_members, temporal = task
    try:
        return _added_columns_for_group(vt, fxx, iy, ix, max_members, temporal)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] augment group {vt:%Y-%m-%d %H}Z f{int(fxx):02d} failed "
              f"({type(e).__name__}: {str(e).splitlines()[0][:80]}); rows left NaN",
              flush=True)
        return None, None


def augment_features(path: str, *, max_members: int = 6, temporal: bool = False,
                     workers: int = 1) -> str:
    """Add the new feature families (lapse rates + surface-boundary gradients) to an
    EXISTING dataset IN PLACE, recomputing ONLY those columns from the cached GRIB.

    The existing ~5k feature columns are left untouched (no re-preprocessing) and
    nothing is re-downloaded — each (valid_time, fxx) sample's ensemble is
    re-assembled from the local GRIB cache and the new features sampled at the
    stored (iy, ix) cells. Each part-file is rewritten with the extra columns.
    Re-run training afterward to pick them up (``feature_columns`` will include them).

    ``temporal`` must match how the parquet was built (adds ``_prev``/``_next``)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    target = _dataset_target(path)
    parts = sorted(Path(target).glob("*.parquet")) if Path(target).is_dir() else [Path(target)]
    if not parts:
        raise FileNotFoundError(f"no parquet part-files under {target}")
    workers = max(1, int(workers))
    t0 = time.time()
    print(f"augmenting {len(parts)} part-file(s) with new features "
          f"({'temporal, ' if temporal else ''}{workers} worker(s))", flush=True)

    pool = None
    if workers > 1:
        max_tasks = int(os.environ.get("FIGS_MAX_TASKS_PER_CHILD", "16"))
        try:
            ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1).shutdown()
            pool = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=max_tasks)
        except TypeError:
            pool = ProcessPoolExecutor(max_workers=workers)
    try:
        total_groups = done_groups = 0
        for pi, p in enumerate(parts):
            df = pd.read_parquet(p)
            groups = list(df.groupby(["valid_time", "fxx"], sort=False))
            total_groups += len(groups)
            tasks, idxs = [], []
            for (vt, fxx), grp in groups:
                vt_dt = pd.Timestamp(vt).to_pydatetime()
                if vt_dt.tzinfo is None:
                    vt_dt = vt_dt.replace(tzinfo=timezone.utc)
                idxs.append(grp.index.to_numpy())
                tasks.append((vt_dt, int(fxx), grp["iy"].to_numpy(), grp["ix"].to_numpy(),
                              max_members, temporal))
            results = (list(pool.map(_augment_worker, tasks)) if pool
                       else [_augment_worker(t) for t in tasks])
            done_groups += len(results)

            col_names = next((n for n, _ in results if n is not None), None)
            if col_names is not None:
                # one (rows × features) block for the whole part, filled per group
                block = np.full((len(df), len(col_names)), np.nan, dtype=np.float32)
                for idx, (names, arr) in zip(idxs, results):
                    if names is not None:
                        block[idx, :] = arr
                new_df = pd.DataFrame(block, columns=col_names, index=df.index)
                # drop any pre-existing same names so re-running augment overwrites
                # cleanly, then attach all new columns in ONE concat (no fragmentation).
                df = df.drop(columns=[c for c in col_names if c in df.columns])
                df = pd.concat([df, new_df], axis=1)
            df.to_parquet(p, index=False)
            n_added = len(col_names) if col_names is not None else 0
            print(f"  [{pi + 1}/{len(parts)}] {p.name}: +{n_added} cols, "
                  f"{done_groups}/{total_groups} samples | {_fmt_eta(time.time() - t0)} elapsed",
                  flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"augmented {len(parts)} part-file(s) in {_fmt_eta(time.time() - t0)} -> {target}",
          flush=True)
    return str(target)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature column names = everything that isn't meta or a label."""
    drop = set(META_COLS) | set(LABEL_COLS)
    return [c for c in df.columns if c not in drop]
