"""FIGS-W command-line entry points.

  build-data  Build the wildfire training matrix from fire-report hours.
  train       Train the wildfire occurrence / deadliness / size models.
  predict     Run a HRRR cycle → fire-weather probability + CIG categorical + size.

Examples:
  python -m figs_w.cli build-static --nlcd nlcd_conus.tif --worldpop worldpop_usa.tif
  python -m figs_w.cli build-data --start 2021-05-01 --end 2021-10-31 --out Data/figs_w/processed/fw.parquet
  python -m figs_w.cli train --parquet Data/figs_w/processed/fw.parquet --backend lightgbm
  python -m figs_w.cli predict --run 2024-08-15T18 --fmax 24
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from . import config as C

DEFAULT_FXX_PER_BAND = (6, 12, 18, 24)   # short-range focus for wildfire ignition/spread


def _parse_dt(s: str) -> datetime:
    s = s.replace("T", " ")
    for fmt in ("%Y-%m-%d %H", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognized datetime: {s}")


def cmd_build_static(args):
    from .data import static as S

    print("terrain:", sorted(S.build_terrain()))
    if args.nlcd:
        print("land use:", sorted(S.build_landuse(args.nlcd)))
    if args.worldpop:
        print("population density:", sorted(S.build_popdensity(args.worldpop)))


def cmd_build_data(args):
    from .data.dataset import build_dataset_for_runs, fire_valid_hours

    start, end = _parse_dt(args.start), _parse_dt(args.end)
    bands = tuple(int(x) for x in args.bands.split(",")) if args.bands else DEFAULT_FXX_PER_BAND
    hours = fire_valid_hours(start, end, min_fires=args.min_fires)
    pairs = [(v - timedelta(hours=f), f) for v in hours for f in bands]
    print(f"{len(hours)} fire valid hours -> {len(pairs)} (run,fxx) samples")
    build_dataset_for_runs(pairs, args.out, neg_keep=args.neg_keep,
                           flush_every=args.flush_every, min_free_gb=args.min_free_gb,
                           workers=args.workers)


def cmd_train(args):
    import json

    from .model.train import train_all

    m = train_all(args.parquet, out_dir=args.models, backend=args.backend,
                  calibrator=args.calibrator, n_estimators=args.trees, max_depth=args.depth,
                  learning_rate=args.lr, max_rows_per_band=args.max_rows_per_band,
                  val_rows=args.val_rows)
    print(json.dumps(m, indent=2, default=str))


def cmd_predict(args):
    import numpy as np

    from .model.predict import predict_forecast
    from .products import cig, plots

    run = _parse_dt(args.run)
    fxxs = list(range(1, args.fmax + 1))
    preds = predict_forecast(run, fxxs, models_dir=args.models)
    if args.extent:
        plots.set_extent([float(x) for x in args.extent.split(",")])
    # day-max probability + categorical (prob×CIG) + day-max median size
    pmax = np.nanmax(np.stack([preds[f]["p_wildfire"] for f in fxxs]), axis=0)
    cigmax = np.nanmax(np.stack([cig.derive_cig_category("wildfire",
                       np.nan_to_num(preds[f]["dist_wildfire"])) for f in fxxs]), axis=0).astype(int)
    cat = cig.prob_to_category("wildfire", pmax * 100.0,
                               np.where(pmax >= C.SPC_PROB_LEVELS["wildfire"][0], cigmax, 0))
    v0, v1 = run + timedelta(hours=fxxs[0]), run + timedelta(hours=fxxs[-1])
    period = f"valid {v0:%Y-%m-%d %HZ}–{v1:%Y-%m-%d %HZ}"
    plots.plot_probability(pmax, f"FIGS-W day-max p(wildfire) — {period}", str(C.PRODUCTS / "fw_prob.png"), cig=cigmax)
    plots.plot_categorical(cat, f"FIGS-W fire-weather categorical — {period}", str(C.PRODUCTS / "fw_cat.png"))
    print(f"wrote products to {C.PRODUCTS}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="figs_w", description="FIGS-W wildfire guidance")
    sub = p.add_subparsers(dest="cmd", required=True)

    bs = sub.add_parser("build-static", help="build static fields: terrain (SRTM via "
                        "`elevation`), + land use (NLCD) and population density (WorldPop) "
                        "if their rasters are given (one-time)")
    bs.add_argument("--nlcd", default=None, help="path to a downloaded NLCD land-cover GeoTIFF")
    bs.add_argument("--worldpop", default=None, help="path to a downloaded WorldPop USA GeoTIFF")
    bs.set_defaults(func=cmd_build_static)

    b = sub.add_parser("build-data", help="build the wildfire training matrix")
    b.add_argument("--start", required=True); b.add_argument("--end", required=True)
    b.add_argument("--out", default=None); b.add_argument("--bands", default=None)
    b.add_argument("--min-fires", type=int, default=1)
    b.add_argument("--neg-keep", type=float, default=0.02)
    b.add_argument("--flush-every", type=int, default=10)
    b.add_argument("--min-free-gb", type=float, default=50.0)
    b.add_argument("--workers", type=int, default=1,
                   help="samples built concurrently in separate processes (HRRR fetch + "
                        "feature build); 1 = serial. Catalog is disk-cached so workers "
                        "don't re-query NIFC")
    b.set_defaults(func=cmd_build_data)

    t = sub.add_parser("train", help="train wildfire models")
    t.add_argument("--parquet", required=True); t.add_argument("--models", default=None)
    t.add_argument("--backend", default="lightgbm", choices=["mlx", "lightgbm"])
    t.add_argument("--calibrator", default="logistic", choices=["logistic", "isotonic"])
    t.add_argument("--trees", type=int, default=500); t.add_argument("--depth", type=int, default=6)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--max-rows-per-band", type=int, default=800_000)
    t.add_argument("--val-rows", type=int, default=250_000)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="forecast a HRRR cycle -> fire-weather products")
    pr.add_argument("--run", required=True); pr.add_argument("--fmax", type=int, default=24)
    pr.add_argument("--models", default=None)
    pr.add_argument("--extent", default=None, help="lon_min,lon_max,lat_min,lat_max")
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
