"""Write model predictions to a CF-style netCDF file.

Stores, per forecast hour, the hazard probabilities and the conditional-intensity
distributions on the FIGS grid (with 2-D lat/lon coordinates), plus the derived
CIG category and SPC categorical risk so the file is self-contained.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from ..config import HAZARDS, INTENSITY_BINS, PIB_LABELS, PRODUCTS
from ..data import grid
from . import cig, pib as pibmod


def predictions_path(run, out_path: str | Path | None = None, fxx=None) -> Path:
    """Canonical netCDF path for a run's predictions (the cache/output location).

    The forecast-hour span is encoded in the name (e.g. ``figs_20260620_12Z_f07-18.nc``)
    so two different inference periods for the SAME run don't overwrite each other."""
    if out_path:
        return Path(out_path)
    if fxx:
        fxxs = [int(f) for f in fxx]
        return PRODUCTS / f"figs_{run:%Y%m%d_%HZ}_f{min(fxxs):02d}-{max(fxxs):02d}.nc"
    return PRODUCTS / f"figs_{run:%Y%m%d_%HZ}.nc"


def read_predictions(path: str | Path) -> dict:
    """Reconstruct the ``{fxx: {'p_<h>':(ny,nx), 'dist_<h>':(nbins,ny,nx)}}`` dict
    from a netCDF written by ``write_predictions`` — the inverse, used as a cache."""
    import xarray as xr

    out: dict = {}
    with xr.open_dataset(path) as ds:
        fxxs = [int(f) for f in ds["fxx"].values]
        for i, f in enumerate(fxxs):
            d = {}
            for h in HAZARDS:
                d[f"p_{h}"] = ds[f"p_{h}"].isel(fxx=i).values.astype("float32")
                d[f"dist_{h}"] = ds[f"dist_{h}"].isel(fxx=i).values.astype("float32")
                if f"pib_{h}" in ds:
                    d[f"pib_{h}"] = ds[f"pib_{h}"].isel(fxx=i).values.astype("float32")
            out[f] = d
    return out


def write_predictions(predictions: dict, run, out_path: str | Path | None = None) -> str:
    """Write ``predictions`` (``{fxx: {'p_<h>':(ny,nx), 'dist_<h>':(nbins,ny,nx)}}``)
    to netCDF. Returns the path."""
    import xarray as xr

    fxxs = sorted(predictions)
    lat, lon = grid.figs_latlon()
    valid_times = np.array([np.datetime64(int((run + timedelta(hours=int(f))).timestamp()), "s")
                            for f in fxxs])

    ds = xr.Dataset(
        coords={
            "fxx": ("fxx", np.array(fxxs, dtype="int16")),
            "valid_time": ("fxx", valid_times),
            "lat": (("y", "x"), lat.astype("float32")),
            "lon": (("y", "x"), lon.astype("float32")),
        }
    )
    for h in HAZARDS:
        labels = list(INTENSITY_BINS[h]["labels"])
        ds.coords[f"{h}_bin"] = (f"{h}_bin", labels)
        prob = np.stack([np.asarray(predictions[f][f"p_{h}"]) for f in fxxs], axis=0)
        dist = np.stack([np.asarray(predictions[f][f"dist_{h}"]) for f in fxxs], axis=0)
        ds[f"p_{h}"] = (("fxx", "y", "x"), prob.astype("float32"))
        ds[f"p_{h}"].attrs["long_name"] = f"probability of {h}"
        ds[f"dist_{h}"] = (("fxx", f"{h}_bin", "y", "x"), dist.astype("float32"))
        ds[f"dist_{h}"].attrs["long_name"] = f"conditional intensity distribution | {h}"
        # derived CIG category (0-3) and SPC categorical risk (0-5) per fxx
        cig_idx = np.stack([cig.derive_cig_category(h, np.nan_to_num(predictions[f][f"dist_{h}"]))
                            for f in fxxs], axis=0)
        cat = np.stack([cig.prob_to_category(h, predictions[f][f"p_{h}"] * 100.0, cig_idx[i])
                        for i, f in enumerate(fxxs)], axis=0)
        ds[f"cig_{h}"] = (("fxx", "y", "x"), cig_idx.astype("int8"))
        ds[f"category_{h}"] = (("fxx", "y", "x"), cat.astype("int8"))

        # PIB (peak-intensity-bin) 7-class distribution + derived most-probable bin
        if all(f"pib_{h}" in predictions[f] for f in fxxs):
            if f"{h}_pib_bin" not in ds.coords:
                ds.coords[f"{h}_pib_bin"] = (f"{h}_pib_bin", list(PIB_LABELS))
            pdist = np.stack([np.asarray(predictions[f][f"pib_{h}"]) for f in fxxs], axis=0)
            mpb = np.stack([pibmod.most_probable_pib(predictions[f][f"pib_{h}"]) for f in fxxs], axis=0)
            ds[f"pib_{h}"] = (("fxx", f"{h}_pib_bin", "y", "x"), pdist.astype("float32"))
            ds[f"pib_{h}"].attrs["long_name"] = f"peak-intensity-bin distribution | {h}"
            ds[f"pib_mode_{h}"] = (("fxx", "y", "x"), mpb.astype("int8"))
            ds[f"pib_mode_{h}"].attrs["long_name"] = f"most probable peak intensity bin (0..6) | {h}"

    ds.attrs["title"] = "FIGS — Forecasting Intensity Guidance for Severe weather"
    ds.attrs["run"] = run.isoformat()
    ds.attrs["grid_dx_km"] = float(grid.FIGS_DX_KM) if hasattr(grid, "FIGS_DX_KM") else 15.0

    out_path = str(out_path) if out_path else str(predictions_path(run, fxx=fxxs))
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path
