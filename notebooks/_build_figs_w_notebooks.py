"""Generate the FIGS-W (wildfire) analysis notebooks — SEPARATE from the FIGS
builder, writes ONLY its own files (never touches the existing notebooks):

  W01_fire_case_analysis.ipynb    — run a forecast, fire-weather maps, overlay fires
  W02_fire_training_progress.ipynb — dataset balance, per-band skill, calibration, size

Run once: ``python notebooks/_build_figs_w_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CID = [0]


def _id():
    _CID[0] += 1
    return f"w{_CID[0]:03d}"


def _src(lines):
    return [ln + ("\n" if i < len(lines) - 1 else "") for i, ln in enumerate(lines)]


def md(*lines):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def nb(*cells):
    return {"cells": list(cells),
            "metadata": {"kernelspec": {"display_name": "Python 3 (met)", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = code(
    "# Run with the `met` conda env kernel.",
    "import warnings; warnings.filterwarnings('ignore')",
    "import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt",
    "sys.path.insert(0, '..')   # so `import figs_w` works from notebooks/",
    "from figs_w import config as C",
)

# --------------------------------------------------------------------------- #
# W01 — fire-weather case analysis
# --------------------------------------------------------------------------- #
case = nb(
    md("# FIGS-W — Wildfire Case Analysis & Display",
       "",
       "Run a forecast for a HRRR cycle, render SPC fire-weather-style probability /",
       "categorical / size maps, and overlay the observed",
       "(NIFC + IEM) active fires. Mirrors FIGS notebook 1."),
    SETUP,
    code("from datetime import datetime, timezone, timedelta",
         "from figs_w.model.predict import predict_or_load",
         "from figs_w.products import plots, cig",
         "from figs_w.data import fire_reports",
         "from figs.products import summary           # generic day-max / median-bin helpers",
         "from figs.data import grid",
         "",
         "RUN        = datetime(2023, 8, 15, 18, tzinfo=timezone.utc)  # HRRR cycle to forecast",
         "FXX        = list(range(1, 25))                              # forecast hours",
         "MODELS     = None                                            # None -> Data/figs_w/models",
         "DL_WORKERS = 8                                               # forecast hours downloaded concurrently"),
    md("## Predict",
       "",
       "`predict_or_load` assembles the single-run HRRR state + static geography per",
       "forecast hour, returning `p_wildfire` and the conditional size distribution",
       "`dist_wildfire` (5 acreage bins). Results are cached as netCDF so repeat runs",
       "skip the download."),
    code("try:",
         "    preds = predict_or_load(RUN, FXX, models_dir=MODELS, workers=DL_WORKERS)",
         "    print(f'{len(preds)} forecast hours')",
         "except Exception as e:",
         "    print('predict failed (need trained models?):', e); preds = None"),
    md("## Map extent",
       "",
       "Plan-view extent `[lon_min, lon_max, lat_min, lat_max]` for ALL plots below."),
    code("MAP_EXTENT = [-125, -100, 32, 49]   # western CONUS (fire country)",
         "# MAP_EXTENT = [-125, -66, 24, 50]  # full CONUS",
         "plots.set_extent(MAP_EXTENT)"),
    md("## Per-forecast-hour fire-weather maps — animated over the period",
       "",
       "Looping GIFs over every forecast hour: probability+CIG and the fire-weather",
       "categorical (NONE / ELEVATED / CRITICAL / EXTREME)."),
    code("if preds:",
         "    from IPython.display import Image, display",
         "    from figs.products import animate",
         "    prob_frames, cat_frames = [], []",
         "    for f in sorted(FXX):",
         "        p = preds[f]['p_wildfire']; dist = np.nan_to_num(preds[f]['dist_wildfire'])",
         "        cig_idx = cig.derive_cig_category('wildfire', dist)",
         "        cat = cig.prob_to_category('wildfire', p*100,",
         "                                   np.where(p >= C.SPC_PROB_LEVELS['wildfire'][0], cig_idx, 0))",
         "        vt = RUN + timedelta(hours=f)",
         "        ttl = f'valid {vt:%Y-%m-%d %HZ} (f{f:02d}, init {RUN:%Y-%m-%d %HZ})'",
         "        prob_frames.append(plots.plot_probability(p, f'wildfire p+CIG — {ttl}', f'/tmp/_w_p_{f:02d}.png', cig=cig_idx))",
         "        cat_frames.append(plots.plot_categorical(cat, f'fire-weather categorical — {ttl}', f'/tmp/_w_c_{f:02d}.png'))",
         "    display(Image(animate.make_gif(prob_frames, '/tmp/_w_p.gif', duration_ms=600)))",
         "    display(Image(animate.make_gif(cat_frames, '/tmp/_w_c.gif', duration_ms=600)))"),
    md("## Day-total: probability, categorical, day-max median size"),
    code("if preds:",
         "    from IPython.display import Image, display",
         "    v0 = RUN + timedelta(hours=min(FXX)); v1 = RUN + timedelta(hours=max(FXX))",
         "    period = f'valid {v0:%Y-%m-%d %HZ}–{v1:%Y-%m-%d %HZ}'",
         "    pmax  = summary.day_max({f: preds[f]['p_wildfire'] for f in FXX})",
         "    cigmax = np.stack([cig.derive_cig_category('wildfire', np.nan_to_num(preds[f]['dist_wildfire'])) for f in FXX]).max(axis=0)",
         "    cat = cig.prob_to_category('wildfire', pmax*100, np.where(pmax >= C.SPC_PROB_LEVELS['wildfire'][0], cigmax, 0))",
         "    plots.plot_probability(pmax, f'day-max p(wildfire) + CIG — {period}', '/tmp/_w_pmax.png', cig=cigmax); display(Image('/tmp/_w_pmax.png'))",
         "    plots.plot_categorical(cat, f'fire-weather categorical — {period}', '/tmp/_w_catmax.png'); display(Image('/tmp/_w_catmax.png'))",
         "    # day-max median conditional size bin, masked to the threat area",
         "    med = np.stack([summary.median_intensity_bin(np.nan_to_num(preds[f]['dist_wildfire'])) for f in FXX]).max(axis=0)",
         "    med = np.where(pmax >= C.SPC_PROB_LEVELS['wildfire'][0], med, -1)",
         "    plots.plot_intensity(med, f'median fire size (threat area, day-max) — {period}', '/tmp/_w_size.png'); display(Image('/tmp/_w_size.png'))"),
    md("## Forecast vs observed fires",
       "",
       "Overlay the active NIFC + IEM fire footprints over the valid window on the",
       "day-max probability."),
    code("import cartopy.crs as ccrs, cartopy.feature as cfeature",
         "if preds:",
         "    lat, lon = grid.figs_latlon()",
         "    v0 = RUN + timedelta(hours=min(FXX)); v1 = RUN + timedelta(hours=max(FXX))",
         "    # collect active-fire footprint points across the valid window (hourly)",
         "    pts = []",
         "    h = v0",
         "    while h <= v1:",
         "        for p_xy, *_ in fire_reports.active_fires(h):",
         "            pts.append(np.atleast_2d(p_xy))",
         "        h += timedelta(hours=3)",
         "    pts = np.vstack(pts) if pts else np.empty((0,2))",
         "    print(len(pts), 'active-fire footprint points in window')",
         "    pmax = summary.day_max({f: preds[f]['p_wildfire'] for f in FXX})",
         "    lv = list(C.SPC_PROB_LEVELS['wildfire']) + [1.0]",
         "    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5)",
         "    fig = plt.figure(figsize=(11,7)); ax = plt.axes(projection=proj)",
         "    ax.add_feature(cfeature.STATES, lw=0.3)",
         "    if MAP_EXTENT: ax.set_extent(MAP_EXTENT)",
         "    cf = ax.contourf(lon, lat, pmax, levels=lv, colors=plots.PROB_COLORS['wildfire'], transform=ccrs.PlateCarree(), alpha=0.85, extend='neither')",
         "    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, shrink=0.8, ticks=C.SPC_PROB_LEVELS['wildfire'])",
         "    if len(pts): ax.scatter(pts[:,1], pts[:,0], s=10, c='black', marker='x', lw=0.6, transform=ccrs.PlateCarree(), label='observed fire')",
         "    ax.set_title(f'FIGS-W day-max p(wildfire) vs observed fires — valid {v0:%Y-%m-%d %HZ}–{v1:%Y-%m-%d %HZ}'); ax.legend(loc='lower right'); plt.show()"),
)

# --------------------------------------------------------------------------- #
# W02 — training diagnostics
# --------------------------------------------------------------------------- #
train = nb(
    md("# FIGS-W — Training Diagnostics",
       "",
       "Dataset balance, per-lead-band held-out skill for wildfire occurrence,",
       "calibration curves, and the conditional fire-size distribution.",
       "Mirrors FIGS notebook 2."),
    SETUP,
    code("from figs.data.dataset import read_dataset",
         "from figs_w.data.dataset import META_COLS, LABEL_COLS",
         "from figs_w.model.predict import _band_tags",
         "from figs.model.wrapper import GBDTModel",
         "from figs.model.calibrate import Calibrator, reliability_ci, low_dense_edges",
         "import json, glob",
         "",
         "DATA   = 'Data/figs_w/processed/fw.parquet'   # the built FIGS-W matrix",
         "MODELS = C.MODELS",
         "VAL_SAMPLE = 400_000",
         "SIZE_LABELS = C.INTENSITY_BINS['wildfire']['labels']",
         "EDGES = np.asarray(C.INTENSITY_BINS['wildfire']['edges'], float)",
         "# size bins are DERIVED from the raw wildfire_size column at analysis time",
         "def size_bin(sz): return (np.searchsorted(EDGES, np.asarray(sz, float), side='right') - 1).clip(0, len(SIZE_LABELS)-1)"),
    md("## Split sizes and class balance (metadata only)"),
    code("meta = read_dataset(DATA, columns=META_COLS + LABEL_COLS)",
         "print(meta.split.value_counts())",
         "print(f\"wildfire positives : {100*meta.wildfire.mean():.3f}% ({int(meta.wildfire.sum()):,})\")",
         "sz = meta['wildfire_size'].to_numpy(float); sz = sz[np.isfinite(sz) & (sz > 0)]",
         "hist = np.bincount(size_bin(sz), minlength=len(SIZE_LABELS)) / max(len(sz), 1)",
         "print(f'conditional size distribution (among {len(sz):,} wildfire cells with known size):')",
         "for i, lab in enumerate(SIZE_LABELS): print(f'  {lab:>10s}: {100*hist[i]:5.1f}%')"),
    md("## Positive rate by lead band"),
    code("if 'fxx' in meta:",
         "    rows = []",
         "    for b in C.LEAD_BANDS:",
         "        sub = meta[(meta.fxx>=b.fmin)&(meta.fxx<=b.fmax)]",
         "        if len(sub): rows.append(dict(band=b.name, n=len(sub), wildfire=sub.wildfire.mean()))",
         "    bp = pd.DataFrame(rows).set_index('band'); display(bp)"),
    md("## Held-out validation skill per band (occurrence)",
       "",
       "Loads each band's calibrated model, scores the validation split, reports AUC / AUPRC."),
    code("from sklearn.metrics import roc_auc_score, average_precision_score",
         "feat_cols = json.loads((MODELS/'feature_cols.json').read_text())",
         "RAW = ['wildfire']",
         "def load_val(b):",
         "    f = [('fxx','>=',b.fmin),('fxx','<=',b.fmax),('split','==','validation')]",
         "    d = read_dataset(DATA, filters=f, columns=feat_cols + RAW)",
         "    return d.sample(VAL_SAMPLE, random_state=0) if len(d) > VAL_SAMPLE else d",
         "rows = []",
         "for b in C.LEAD_BANDS:",
         "    if not (MODELS/f'hazard_wildfire_{b.name}.pkl').exists(): continue",
         "    d = load_val(b)",
         "    if d.empty or d.wildfire.nunique() < 2: continue",
         "    X = d[feat_cols].to_numpy(np.float32)",
         "    y = {'wildfire': d.wildfire.to_numpy(int)}",
         "    r = dict(band=b.name, n=len(d), pos=int(d.wildfire.sum()))",
         "    for name in ('wildfire',):",
         "        m = MODELS/f'hazard_{name}_{b.name}.pkl'",
         "        if not m.exists() or len(np.unique(y[name])) < 2: continue",
         "        p = GBDTModel.load(m).predict_pos(X)",
         "        cp = MODELS/f'calib_{name}_{b.name}.pkl'",
         "        if cp.exists(): p = Calibrator.load(cp).transform(p)",
         "        r[f'{name}_auc']   = roc_auc_score(y[name], p)",
         "        r[f'{name}_auprc'] = average_precision_score(y[name], p)",
         "    rows.append(r)",
         "skill = pd.DataFrame(rows).set_index('band'); display(skill)"),
    md("## AUC / AUPRC vs lead band"),
    code("if len(skill):",
         "    fig, ax = plt.subplots(1, 2, figsize=(13,4))",
         "    for c in ('wildfire_auc',):",
         "        if c in skill: ax[0].plot(skill.index, skill[c], 'o-', label=c)",
         "    ax[0].set_title('AUC by band'); ax[0].set_ylim(0.5,1.0); ax[0].legend(); ax[0].grid(alpha=.3)",
         "    for c in ('wildfire_auprc',):",
         "        if c in skill: ax[1].plot(skill.index, skill[c], 'o-', label=c)",
         "    ax[1].set_title('AUPRC by band'); ax[1].legend(); ax[1].grid(alpha=.3)",
         "    for a in ax: a.tick_params(axis='x', rotation=45)",
         "    plt.tight_layout(); plt.show()"),
    md("## Reliability (calibration) — all-band ensemble, held-out TEST split, weighted"),
    code("tags = _band_tags(MODELS)",
         "f = [('split','==','test')]",
         "dt = read_dataset(DATA, filters=f, columns=feat_cols + ['wildfire','weight'])",
         "dt = dt.sample(VAL_SAMPLE, random_state=0) if len(dt) > VAL_SAMPLE else dt",
         "Xt = dt[feat_cols].to_numpy(np.float32)",
         "ytest = {'wildfire': dt.wildfire.to_numpy()}",
         "def ensemble_p(kind):",
         "    ps = []",
         "    for t in tags:",
         "        m = MODELS/f'hazard_{kind}_{t}.pkl'",
         "        if not m.exists(): continue",
         "        p = GBDTModel.load(m).predict_pos(Xt)",
         "        cp = MODELS/f'calib_{kind}_{t}.pkl'",
         "        if cp.exists(): p = Calibrator.load(cp).transform(p)",
         "        ps.append(p)",
         "    return np.mean(ps, axis=0) if ps else None",
         "edges = np.array([0,0.02,0.05,0.10,0.20,0.35,0.6,1.0])",
         "fig, ax = plt.subplots(figsize=(6,6)); ax.plot([0,1],[0,1],'k--',lw=.8)",
         "for kind in ('wildfire',):",
         "    p = ensemble_p(kind)",
         "    if p is None: continue",
         "    mid, obs, lo, hi, _w, _n = reliability_ci(p, ytest[kind], sample_weight=dt.weight.to_numpy(), edges=edges)",
         "    ax.errorbar(mid, obs, yerr=[obs-lo, hi-obs], marker='o', capsize=3, label=kind)",
         "ax.set_xlabel('forecast probability'); ax.set_ylabel('observed frequency')",
         "ax.set_title('FIGS-W reliability (test, weighted)'); ax.legend(); ax.grid(alpha=.3); plt.show()"),
    md("## Conditional fire-size distribution: observed vs predicted (the CIG target)"),
    code("smp = MODELS/f'intensity_wildfire_{tags[0]}.pkl' if tags else None",
         "if smp and smp.exists():",
         "    d = read_dataset(DATA, filters=[('split','==','validation')], columns=feat_cols+['wildfire_size'])",
         "    d = d[np.isfinite(d.wildfire_size) & (d.wildfire_size > 0)]",
         "    d = d.sample(VAL_SAMPLE, random_state=0) if len(d) > VAL_SAMPLE else d",
         "    nb_ = len(SIZE_LABELS)",
         "    obs = np.bincount(size_bin(d.wildfire_size.to_numpy()), minlength=nb_)[:nb_]; obs = obs/obs.sum()",
         "    # predicted conditional distribution averaged over the same cells (all bands)",
         "    X = d[feat_cols].to_numpy(np.float32); acc = np.zeros(nb_); nseen = 0",
         "    for t in tags:",
         "        sp = MODELS/f'intensity_wildfire_{t}.pkl'",
         "        if not sp.exists(): continue",
         "        m = GBDTModel.load(sp); pr = m.predict_proba(X); full = np.zeros((len(X), nb_))",
         "        for j, c in enumerate(np.asarray(m.classes_).astype(int)):",
         "            if 0 <= c < nb_: full[:, c] = pr[:, j]",
         "        acc += full.mean(axis=0); nseen += 1",
         "    pred = acc/nseen if nseen else np.full(nb_, np.nan)",
         "    x = np.arange(nb_); w=0.4",
         "    fig, ax = plt.subplots(figsize=(8,4))",
         "    ax.bar(x-w/2, obs, w, label='observed'); ax.bar(x+w/2, pred, w, label='predicted')",
         "    ax.set_xticks(x); ax.set_xticklabels([f'{l} ac' for l in SIZE_LABELS], rotation=20)",
         "    ax.set_ylabel('frequency'); ax.legend(); ax.set_title('conditional fire size: observed vs predicted'); plt.show()"),
)


def main():
    (HERE / "W01_fire_case_analysis.ipynb").write_text(json.dumps(case, indent=1))
    (HERE / "W02_fire_training_progress.ipynb").write_text(json.dumps(train, indent=1))
    print("wrote W01_fire_case_analysis.ipynb, W02_fire_training_progress.ipynb")


if __name__ == "__main__":
    main()
