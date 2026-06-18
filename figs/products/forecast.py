"""Turn a model forecast into SPC-style products.

Given per-forecast-hour predictions (from ``model.predict.predict_forecast``),
render for each hazard:
  * per-fxx probability + categorical + median-intensity plots and their GIFs;
  * the cumulative daily probability/CIG/categorical day-total products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import HAZARDS, PRODUCTS
from . import animate, cig, plots, summary


def render_forecast(predictions: dict, run_label: str, out_dir: str | Path | None = None) -> dict:
    """``predictions`` maps fxx -> {'p_<h>':grid, 'dist_<h>':stack}. Returns a dict
    of output file paths per hazard."""
    out_dir = Path(out_dir) if out_dir else (PRODUCTS / run_label)
    out_dir.mkdir(parents=True, exist_ok=True)
    fxxs = sorted(predictions)
    results: dict[str, dict] = {}

    for h in HAZARDS:
        prob_by_fxx = {f: predictions[f][f"p_{h}"] for f in fxxs}
        dist_by_fxx = {f: predictions[f][f"dist_{h}"] for f in fxxs}

        # single combined probability+CIG map per forecast hour (was separate
        # probability and CIG/categorical plots), plus the median-intensity map
        probcig_frames, int_frames = [], []
        for f in fxxs:
            prob = prob_by_fxx[f]
            dist = np.nan_to_num(dist_by_fxx[f])
            cig_idx = cig.derive_cig_category(h, dist)
            med = summary.median_intensity_bin(dist)
            probcig_frames.append(plots.plot_probability(
                prob, h, f"{h} p+CIG f{f:02d} ({run_label})",
                out_dir / f"{h}_probcig_f{f:02d}.png", cig=cig_idx))
            int_frames.append(plots.plot_intensity(
                med, h, f"{h} median intensity f{f:02d}", out_dir / f"{h}_int_f{f:02d}.png"))

        # day-total: per-hazard combined prob+CIG day-max map (probabilistic).
        # The categorical outlook is NOT per-hazard — see the combined one below.
        cum = summary.cumulative_categorical(h, prob_by_fxx, {f: np.nan_to_num(dist_by_fxx[f]) for f in fxxs})
        day_probcig = plots.plot_probability(cum["prob"], h, f"{h} day-max prob+CIG ({run_label})",
                                             out_dir / f"{h}_DAYMAX_probcig.png", cig=cum["cig"])

        results[h] = {
            "probcig_gif": animate.make_gif(probcig_frames, out_dir / f"{h}_probcig.gif"),
            "intensity_gif": animate.make_gif(int_frames, out_dir / f"{h}_intensity.gif"),
            "day_probcig": day_probcig,
        }

    # Single cumulative daily SPC CATEGORICAL outlook across ALL hazards
    # (element-wise max category over tor/wind/hail) — the SPC Day-1 categorical.
    combined = summary.combined_categorical(predictions)
    results["categorical"] = {
        "day_cat": plots.plot_categorical(
            combined["category"], "all", f"cumulative daily categorical risk ({run_label})",
            out_dir / "DAYMAX_categorical.png"),
    }
    return results
