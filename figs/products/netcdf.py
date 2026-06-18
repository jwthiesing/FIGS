"""Write model predictions to a CF-style netCDF file.

Stores, per forecast hour, the hazard probabilities and the conditional-intensity
distributions on the FIGS grid (with 2-D lat/lon coordinates), plus the derived
CIG category and SPC categorical risk so the file is self-contained.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from ..config import HAZARDS, INTENSITY_BINS, PRODUCTS
from ..data import grid
from . import cig


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

    ds.attrs["title"] = "FIGS — Forecasting Intensity Guidance for Severe weather"
    ds.attrs["run"] = run.isoformat()
    ds.attrs["grid_dx_km"] = float(grid.FIGS_DX_KM) if hasattr(grid, "FIGS_DX_KM") else 15.0

    out_path = str(out_path) if out_path else str(PRODUCTS / f"figs_{run:%Y%m%d_%HZ}.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path
