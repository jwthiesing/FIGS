"""Write FIGS-W model predictions to a CF-style netCDF file.

Format is intentionally near-identical to ``figs.products.netcdf`` so the same
downstream tools (notebooks, verification scripts, archival) work on both outputs.
Differences from the FIGS format:
  * ``title`` attribute is "FIGS-W — Forecasting Intensity Guidance System: Wildfires"
  * hazards = ("wildfire",); intensity bins are acreage-based (not EF/kt/inch)
  * filename prefix is ``figs_w_`` instead of ``figs_``
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from .. import config as C
from figs.data import grid as F_GRID
from . import cig


def predictions_path(run, out_path: str | Path | None = None, fxx=None) -> Path:
    """Canonical netCDF path for a FIGS-W run (``figs_w_YYYYMMDD_HHZ_fMM-NN.nc``)."""
    if out_path:
        return Path(out_path)
    if fxx:
        fxxs = [int(f) for f in fxx]
        return C.PRODUCTS / f"figs_w_{run:%Y%m%d_%HZ}_f{min(fxxs):02d}-{max(fxxs):02d}.nc"
    return C.PRODUCTS / f"figs_w_{run:%Y%m%d_%HZ}.nc"


def read_predictions(path: str | Path) -> dict:
    """Reconstruct ``{fxx: {'p_wildfire':(ny,nx), 'dist_wildfire':(nbins,ny,nx)}}``
    from a netCDF written by ``write_predictions``."""
    import xarray as xr

    out: dict = {}
    with xr.open_dataset(path) as ds:
        fxxs = [int(f) for f in ds["fxx"].values]
        for i, f in enumerate(fxxs):
            d = {}
            for h in C.HAZARDS:
                d[f"p_{h}"] = ds[f"p_{h}"].isel(fxx=i).values.astype("float32")
                d[f"dist_{h}"] = ds[f"dist_{h}"].isel(fxx=i).values.astype("float32")
            out[f] = d
    return out


def write_predictions(predictions: dict, run, out_path: str | Path | None = None) -> str:
    """Write ``predictions`` to netCDF. Returns the path written."""
    import xarray as xr

    fxxs = sorted(predictions)
    lat, lon = F_GRID.figs_latlon()
    valid_times = np.array(
        [np.datetime64(int((run + timedelta(hours=int(f))).timestamp()), "s") for f in fxxs]
    )

    ds = xr.Dataset(
        coords={
            "fxx": ("fxx", np.array(fxxs, dtype="int16")),
            "valid_time": ("fxx", valid_times),
            "lat": (("y", "x"), lat.astype("float32")),
            "lon": (("y", "x"), lon.astype("float32")),
        }
    )
    for h in C.HAZARDS:
        labels = list(C.INTENSITY_BINS[h]["labels"])
        ds.coords[f"{h}_bin"] = (f"{h}_bin", labels)
        prob = np.stack([np.asarray(predictions[f][f"p_{h}"]) for f in fxxs], axis=0)
        dist = np.stack([np.asarray(predictions[f][f"dist_{h}"]) for f in fxxs], axis=0)
        ds[f"p_{h}"] = (("fxx", "y", "x"), prob.astype("float32"))
        ds[f"p_{h}"].attrs["long_name"] = f"probability of {h}"
        ds[f"dist_{h}"] = (("fxx", f"{h}_bin", "y", "x"), dist.astype("float32"))
        ds[f"dist_{h}"].attrs["long_name"] = f"conditional intensity distribution | {h}"
        cig_idx = np.stack(
            [cig.derive_cig_category(h, np.nan_to_num(predictions[f][f"dist_{h}"]))
             for f in fxxs], axis=0
        )
        cat = np.stack(
            [cig.prob_to_category(h, predictions[f][f"p_{h}"] * 100.0, cig_idx[i])
             for i, f in enumerate(fxxs)], axis=0
        )
        ds[f"cig_{h}"] = (("fxx", "y", "x"), cig_idx.astype("int8"))
        ds[f"category_{h}"] = (("fxx", "y", "x"), cat.astype("int8"))

    ds.attrs["title"] = "FIGS-W — Forecasting Intensity Guidance System: Wildfires"
    ds.attrs["run"] = run.isoformat()
    ds.attrs["grid_dx_km"] = float(F_GRID.FIGS_DX_KM) if hasattr(F_GRID, "FIGS_DX_KM") else 15.0

    out_path = str(out_path) if out_path else str(predictions_path(run, fxx=fxxs))
    C.PRODUCTS.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path
