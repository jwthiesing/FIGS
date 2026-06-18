"""FIGS command-line entry points.

Subcommands:
  build-data  Build a training matrix from severe-report hours over a date range,
              sampling each severe valid hour from one primary run per lead band
              (issuance-capped ensemble).
  train       Train all model groups (lead-banded) + calibrators from a parquet.
  predict     Run a HRRR cycle through the models and render SPC-style products.

Examples:
  python -m figs.cli build-data --start 2024-05-01 --end 2024-05-31 --out Data/processed/may24.parquet
  python -m figs.cli train --parquet Data/processed/may24.parquet
  python -m figs.cli predict --run 2024-05-21T12 --fmax 36
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

# one representative lead time per lead band (so each event is seen across bands)
DEFAULT_FXX_PER_BAND = (6, 12, 18, 24, 30, 36, 42, 48)  # 6-hourly leads, one per band


def _parse_dt(s: str) -> datetime:
    s = s.replace("T", " ")
    for fmt in ("%Y-%m-%d %H", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognized datetime: {s}")


def training_pairs(start: datetime, end: datetime, fxx_per_band=DEFAULT_FXX_PER_BAND,
                   min_reports: int = 1):  # noqa: D401
    """(primary_run, fxx) samples: each severe valid hour is paired with a primary
    run at one lead per band, so events are seen across short/medium/long ranges."""
    from .data.dataset import severe_valid_hours

    valid_hours = severe_valid_hours(start, end, min_reports=min_reports)
    pairs = []
    for v in valid_hours:
        for fxx in fxx_per_band:
            pairs.append((v - timedelta(hours=int(fxx)), int(fxx)))
    return pairs, valid_hours


def cmd_build_data(args):
    from .data.dataset import build_dataset_for_runs

    start, end = _parse_dt(args.start), _parse_dt(args.end)
    fxx_bands = tuple(int(x) for x in args.bands.split(",")) if args.bands else DEFAULT_FXX_PER_BAND
    pairs, valid_hours = training_pairs(start, end, fxx_bands, min_reports=args.min_reports)
    print(f"{len(valid_hours)} severe valid hours -> {len(pairs)} (run,fxx) samples")
    build_dataset_for_runs(pairs, args.out, max_members=args.members,
                           temporal=args.temporal, neg_keep=args.neg_keep,
                           min_free_gb=args.min_free_gb, workers=args.workers,
                           flush_every=args.flush_every)


def cmd_train(args):
    import json

    from .model.train import train_all

    # only forward regularization knobs the user actually set (else wrapper defaults)
    extra = {k: v for k, v in (
        ("num_leaves", args.num_leaves),
        ("min_child_weight", args.min_child_weight),
        ("min_child_samples", args.min_child_samples),
        ("colsample_bytree", args.colsample),
        ("subsample", args.subsample),
        ("reg_lambda", args.reg_lambda),
        ("max_bin", args.max_bin),
    ) if v is not None}
    metrics = train_all(args.parquet, out_dir=args.models, band=not args.no_band,
                        max_rows_per_band=args.max_rows_per_band, val_rows=args.val_rows,
                        backend=args.backend, calibrator=args.calibrator, n_bags=args.bags,
                        n_estimators=args.trees, max_depth=args.depth,
                        learning_rate=args.lr, verbose=args.tree_verbose, **extra)
    print(json.dumps(metrics, indent=2, default=str))


def cmd_predict(args):
    from .model.predict import predict_or_load
    from .products.forecast import render_forecast

    run = _parse_dt(args.run)
    fxx_list = list(range(1, args.fmax + 1))
    # predict_or_load caches/writes the netCDF (same file is cache + output)
    preds = predict_or_load(run, fxx_list, models_dir=args.models,
                            max_members=args.members, temporal=args.temporal,
                            workers=args.workers, cache=not args.no_cache, out_path=args.nc)
    if not args.no_plots:
        out = render_forecast(preds, run, out_dir=args.out)   # datetime -> full valid times in titles
        print("rendered products:")
        for h, d in out.items():
            print(f"  {h}: {d}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="figs", description="FIGS severe-weather guidance")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="build training matrix from severe-report hours")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--out", default=None)
    b.add_argument("--members", type=int, default=6)
    b.add_argument("--bands", default=None,
                   help="comma-separated lead hours, one per band (default 6,18,30,42)")
    b.add_argument("--min-reports", type=int, default=1,
                   help="only build hours with >= this many severe reports (cuts marginal hours)")
    b.add_argument("--workers", type=int, default=1,
                   help="samples to build concurrently (separate processes); pushes more "
                        "simultaneous S3 streams when bandwidth has headroom")
    b.add_argument("--flush-every", type=int, default=25,
                   help="checkpoint the parquet every N completed samples")
    b.add_argument("--neg-keep", type=float, default=0.025)
    b.add_argument("--temporal", action="store_true",
                   help="include previous/following-hour fields (~3x storage; off by default)")
    b.add_argument("--min-free-gb", type=float, default=50.0,
                   help="stop the build if free disk drops below this (GB)")
    b.set_defaults(func=cmd_build_data)

    t = sub.add_parser("train", help="train models from a parquet matrix")
    t.add_argument("--parquet", required=True)
    t.add_argument("--models", default=None)
    t.add_argument("--no-band", action="store_true")
    t.add_argument("--max-rows-per-band", type=int, default=None,
                   help="cap a band's TRAIN rows at read time (bounds RAM; not just a post-load subsample)")
    t.add_argument("--bags", type=int, default=1,
                   help="bagging: train N hazard models per band, each on ALL positives + a disjoint "
                        "1/N fold of negatives, averaged at predict (coverage past the RAM cap + smoothing). "
                        "1 = off. Each bag still obeys --max-rows-per-band")
    t.add_argument("--val-rows", type=int, default=1_000_000,
                   help="cap a band's validation rows read for metrics/calibration")
    t.add_argument("--trees", type=int, default=300)
    t.add_argument("--depth", type=int, default=6)
    t.add_argument("--lr", type=float, default=0.05)
    # regularization knobs (None -> wrapper default); the real overfit levers
    t.add_argument("--num-leaves", type=int, default=None,
                   help="leaves per tree (default 2**depth, the aggressive ceiling); set lower to regularize")
    t.add_argument("--min-child-weight", type=float, default=None,
                   help="min sum-hessian per leaf (default 1.0 is too low for weighted rare classes; try 10-50)")
    t.add_argument("--min-child-samples", type=int, default=None,
                   help="min ROW count per leaf (weight-independent overfit guard; try 200-500)")
    t.add_argument("--colsample", type=float, default=None,
                   help="colsample_bytree (default 0.7; with ~5k features 0.4-0.5 decorrelates + speeds)")
    t.add_argument("--subsample", type=float, default=None, help="row bagging fraction (default 0.7)")
    t.add_argument("--reg-lambda", type=float, default=None, help="L2 regularization (default 1.0)")
    t.add_argument("--max-bin", type=int, default=None,
                   help="LightGBM feature bins (default 255). Lower (e.g. 63/127) cuts histogram "
                        "RAM ~linearly at wide feature counts; keep <=255. Minor accuracy cost")
    t.add_argument("--backend", default="mlx", choices=["mlx", "lightgbm"],
                   help="GBDT backend (lightgbm = fast multithreaded CPU; needs `pip install lightgbm`)")
    t.add_argument("--calibrator", default="logistic", choices=["logistic", "isotonic"],
                   help="probability calibrator (logistic=smooth/robust to sparse bins; isotonic=flexible)")
    t.add_argument("--tree-verbose", type=int, default=0,
                   help="mlx-boosting per-iteration verbosity (1 = show tree progress within each fit)")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="forecast a HRRR cycle -> SPC-style products")
    pr.add_argument("--run", required=True)
    pr.add_argument("--fmax", type=int, default=36)
    pr.add_argument("--models", default=None)
    pr.add_argument("--out", default=None)
    pr.add_argument("--members", type=int, default=6)
    pr.add_argument("--workers", type=int, default=4,
                   help="forecast hours whose HRRR data is downloaded/assembled concurrently "
                        "(I/O-bound; ~workers feature matrices resident at once). 1 = serial")
    pr.add_argument("--temporal", action="store_true",
                   help="include previous/following-hour fields (must match training)")
    pr.add_argument("--nc", default=None,
                   help="output netCDF path (default Data/products/figs_<run>.nc); also the cache")
    pr.add_argument("--no-cache", action="store_true",
                   help="ignore any cached netCDF for this run and recompute from scratch")
    pr.add_argument("--no-plots", action="store_true", help="write only the netCDF, skip plots")
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
