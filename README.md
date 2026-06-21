# FIGS — Forecasting Intensity Guidance for Severe weather

A gradient-boosted-tree severe-weather model in the style of
[nadocast](https://github.com/brianhempel/nadocast), in Python, trained on
**time-lagged HRRR** data. Predicts, per forecast hour and per ~15 km grid cell,
the probability of each hazard (tornado / wind / hail) **and** the conditional
intensity distribution within each hazard, then renders SPC-style probability +
Conditional Intensity Guidance (CIG) products.

## Architecture (download-optimized)

- **Deterministic features come from the main run only** (the most-recent HRRR
  cycle reaching the valid time). The other time-lagged members contribute
  **only** the reflectivity / updraft-helicity probability fields (small surface
  fields), so the heavy 3-D isobaric data is downloaded **once** per valid time,
  not per member (~4–6× less download than an ensemble-mean state).
- **Issuance-capped** member selection: at moderate leads the ensemble mixes
  18-h and 48-h HRRR cycles; at long leads it narrows to the 6-hourly 48-h cycles.
- **Memory-bounded build**: isobaric is regridded to 15 km *as it is read*
  (native cube never fully held), stored float32; the builder streams to parquet
  **part files**; worker processes recycle to release eccodes C-side memory.
- **Output post-processing**: every prediction grid is the per-cell **median of
  Gaussian smooths at 0 / 25 / 50 mi** (a robust de-speckle that keeps real maxima
  but tempers single-cell noise) before any prob/CIG/categorical product.
- **Bagging (optional, `--bags K`)**: per band/hazard, train K hazard models each
  on **all positives + a disjoint 1/K fold of the negatives**, averaged at predict
  — covers all stored negatives past a single model's RAM ceiling, and smooths.
- **Parallel inference**: `predict` assembles the forecast hours across worker
  **processes** (`--workers`), so both the HRRR download and the feature build
  scale across cores; isobaric reads self-heal truncated concurrent fetches.

## Model outputs (per forecast hour)

`p(tor) p(wind) p(hail)` · `p(EF0..EF4+|tor)` · `p(50-55..83+kt|wind)` ·
`p(1-1.49..3.5+in|hail)` → mapped to SPC probability + CIG categorical outlooks.

### Probability + CIG → categorical (custom)

The probability fills use the SPC outlook colors, with a **custom low level added**
below the SPC scale (tor **1%** pastel green; wind/hail **2%** pastel brown):

| hazard | probability levels (%) |
|---|---|
| tornado | **1**, 2, 5, 10, 15, 30, 45, 60 |
| wind | **2**, 5, 15, 30, 45, 60, 75, 90 |
| hail | **2**, 5, 15, 30, 45, 60 |

The SPC-style **categorical** outlook is then looked up from the probability **and**
the cell's CIG (conditional-intensity) category — a custom extension of the official
SPC probability-to-category tables that adds a low **TSTM** row and a **CIG3**
("extreme conditional intensity") column. Cells: `TSTM`(light green) `MRGL`(green)
`SLGT`(yellow) `ENH`(orange) `MDT`(red) `HIGH`(magenta); below the lowest probability = no risk.

**Tornado**

| p(tor) | <CIG1 | CIG1 | CIG2 | CIG3 |
|---|---|---|---|---|
| 1% | TSTM | MRGL | MRGL | SLGT |
| 2% | MRGL | MRGL | SLGT | SLGT |
| 5% | SLGT | SLGT | ENH | ENH |
| 10% | SLGT | ENH | ENH | ENH |
| 15% | ENH | ENH | MDT | MDT |
| 30% | ENH | MDT | HIGH | HIGH |
| 45% | ENH | MDT | HIGH | HIGH |
| 60% | ENH | HIGH | HIGH | HIGH |

**Wind**

| p(wind) | <CIG1 | CIG1 | CIG2 | CIG3 |
|---|---|---|---|---|
| 2% | TSTM | MRGL | MRGL | SLGT |
| 5% | MRGL | MRGL | SLGT | SLGT |
| 15% | SLGT | SLGT | ENH | ENH |
| 30% | SLGT | ENH | ENH | MDT |
| 45% | ENH | ENH | MDT | HIGH |
| 60% | ENH | MDT | HIGH | HIGH |
| 75% | ENH | MDT | HIGH | HIGH |
| 90% | ENH | MDT | HIGH | HIGH |

**Hail**

| p(hail) | <CIG1 | CIG1 | CIG2 | CIG3 |
|---|---|---|---|---|
| 2% | TSTM | MRGL | MRGL | SLGT |
| 5% | MRGL | MRGL | SLGT | SLGT |
| 15% | SLGT | SLGT | ENH | ENH |
| 30% | SLGT | ENH | ENH | MDT |
| 45% | ENH | ENH | MDT | HIGH |
| 60% | ENH | MDT | MDT | HIGH |

## Input parameters (per forecast hour)

≈6,100 features. Most are **spatially smoothed**: the key fields are averaged at
25/50/100 mi and given forward/leftward/straddling gradients, in two tiers, so the
model sees neighborhood context (not just noisy point values). `--temporal` adds
previous/following-hour copies (~3×).

**Tier 1 — non-motion scalars** (means + gradients vs **all 5** storm motions;
48 cols each, ~70 fields → ~3,400):

| Group | # fields | Description |
|---|---:|---|
| Thermo | 14 | SBCAPE/CIN + 0–90/0–180 mb partials, MLCAPE/CIN (100 mb parcel) + 0–90/0–180 partials, DCAPE, PWAT |
| Kinematics | 13 | div / conv / abs-vort @ 925/850/500/250 mb + differential divergence (250−850) |
| TMP/DPT at levels | 8 | T and Td interpolated to 925/850/500/250 mb |
| Lapse rates | 5 | −dT/dz (K/km) over 0–500 m, 500–1000 m, 1–3 km, 3–6 km (SR-wind layers) + 6–9 km |
| Surface boundaries | 15 | gradients flagging boundaries (outflow/front/dryline) for deviant supercells: signed ∂/∂x,∂/∂y + magnitude of 2 m T & 2 m Td; the four 10 m wind-vector gradient components (∂u/∂x,∂u/∂y,∂v/∂x,∂v/∂y); 10 m convergence, vorticity, stretching/shearing/total deformation |
| Ensemble probability | 15 | REFC ≥{10,20,30,40,50}, REFD ≥{30,40,50}, UH 0–3 km ≥{25,75,150}, UH 2–5 km ≥{25,75,150,300} — fraction of members exceeding |

**Tier 2 — motion-relative scalars** (means + gradients in their **own** frame
only; 12 cols each, ~190 fields → ~2,280):

| Group | # fields | Description |
|---|---:|---|
| SR scalars | 80 | SRH, SR-wind speed, streamwise vorticity, streamwise fraction × 5 motions × 4 layers (0–500 m, 500–1000 m, 1–3 km, 3–6 km) |
| Rotated hodograph winds | 110 | storm-relative (u,v) at 11 AGL heights, rotated into each of the 5 motion frames |

**Raw point fields** (~400, unsmoothed): fine TMP/DPT/T−Td/VVEL profiles (16
surface-relative levels: 25 mb ≤150, 50 mb ≤500, 100 mb above), SR-wind vector
components, motion vectors (u/v/spd of the 5 motions), mandatory-level HGT
(925→100 mb), surface scalars (2 m T/Td + rotated 10 m wind, SOILW, TSOIL, PWAT,
cloud, precip type, RELV 0–1/0–2 km), and the T/Td-at-level point values.

5 storm motions = Bunkers RM/LM, 0–6 km mean wind, Corfidi up/downshear. The
fetched isobaric set is **19 fixed levels** (1000→100 mb, 25/50/100 mb spacing;
650/550/450/250/150 dropped for storage — features "at" those are interpolated).

## Running it

```bash
conda activate met

# 1. Build the training matrix from severe-report hours (deterministic-from-main
#    ensemble; one sample per lead band per severe hour). Streams to part files
#    under Data/processed/<name>_parts/.  Env knobs control storage/memory/speed.
FIGS_MEMBER_WORKERS=6 FIGS_MAX_TASKS_PER_CHILD=16 \
python -m figs.cli build-data \
  --start 2020-12-02 --end 2023-12-31 --out Data/processed/figs_2020_2023.parquet \
  --members 6 --bands 6,12,18,24,30,36,42,48 \
  --min-reports 10 --workers 8 --flush-every 10 --min-free-gb 660

# 1b. (optional) Add NEW feature families to an EXISTING matrix without rebuilding.
#     Reuses the local GRIB cache (cache-only, no re-download, no remote file checks)
#     and recomputes ONLY the new columns from the main member — existing columns are
#     left untouched. Re-run training afterward to pick them up.
python -m figs.cli augment-data --parquet Data/processed/figs_2020_2023.parquet --workers 8

# 2. Train lead-banded hazard + conditional-intensity models + calibrators
#    (LightGBM backend; bagging + regularization; calibrator on the held-out split).
python -m figs.cli train --parquet Data/processed/figs_2020_2023.parquet \
  --backend lightgbm --calibrator logistic --bags 3 \
  --depth 6 --num-leaves 48 --min-child-samples 300 --min-child-weight 20 \
  --colsample 0.5 --reg-lambda 5 --max-bin 63 --trees 600 --lr 0.04 \
  --max-rows-per-band 1000000 --val-rows 250000

# 3. Forecast a HRRR cycle -> netCDF + SPC-style plots/animations + cumulative daily.
#    Forecast hours are downloaded + computed across worker processes.
python -m figs.cli predict --run 2024-05-21T12 --fmax 36 --workers 4
#    --nc PATH  custom netCDF path   |   --no-plots  netCDF only

# Analysis notebooks (run with the met kernel):
#   notebooks/01_case_analysis  02_training_progress  03_feature_dependence
#   04_csi_accuracy  05_report_database
```

### build-data flags & env vars

| flag / env | default | effect |
|---|---|---|
| `--members N` | 6 | time-lagged ensemble members (probability fields) |
| `--bands h1,h2,…` | 6,12,18,24,30,36,42,48 | one sample per lead hour, per severe hour |
| `--min-reports N` | 1 | only build hours with ≥ N severe reports |
| `--workers N` | 1 | samples built concurrently (separate processes) |
| `--flush-every N` | 25 | write a parquet part-file every N samples |
| `--min-free-gb N` | 50 | hard-stop (+checkpoint) when free disk drops below |
| `--temporal` | off | add previous/following-hour fields (~3× size) |

### augment-data flags

Adds the lapse-rate + surface-boundary feature columns (Tier-1; ~980 columns with
their spatial expansion) to an existing parquet **in place**, recomputing only those
from the cached GRIB. Idempotent (re-running overwrites the same columns).

| flag | default | effect |
|---|---|---|
| `--parquet PATH` | — | existing dataset (parquet file or `_parts` dir) |
| `--members N` | 6 | ensemble members used to pick the main member (matches build) |
| `--workers N` | 1 | (valid_time, fxx) samples computed concurrently (separate processes) |
| `--temporal` | off | also add `_prev`/`_next` variants — **must match how the parquet was built** |
| `FIGS_MEMBER_WORKERS` | 6 | concurrent member downloads within a sample |
| `FIGS_MAX_TASKS_PER_CHILD` | 16 | recycle a worker after N samples (caps eccodes memory) |

### train flags

| flag | default | effect |
|---|---|---|
| `--backend` | mlx | `lightgbm` (fast multithreaded CPU, native sample weights) or `mlx` |
| `--calibrator` | logistic | probability calibrator (`logistic` robust to sparse bins; `isotonic`) |
| `--bags N` | 1 | bagging: N hazard models/band, each all-positives + a disjoint 1/N negative fold, averaged |
| `--max-rows-per-band N` | none | cap a band's (per-bag) TRAIN rows at read time (bounds RAM) |
| `--val-rows N` | 1,000,000 | cap validation rows read for metrics/calibration |
| `--max-bin N` | 255 | LightGBM feature bins; lower (63/127) cuts histogram RAM at wide feature counts |
| `--num-leaves N` | 2^depth | leaves/tree; set below the ceiling to regularize |
| `--min-child-samples` / `--min-child-weight` | 20 / 1.0 | min rows / min sum-hessian per leaf (raise for rare classes) |
| `--colsample` / `--subsample` / `--reg-lambda` | 0.7 / 0.7 / 1.0 | per-tree feature & row fractions, L2 |
| `--trees` / `--depth` / `--lr` | 300 / 6 / 0.05 | boosting rounds, max depth, learning rate |

LightGBM needs ~2.5× the feature matrix transiently (≈20 KB/row at this width), so
`--max-rows-per-band` is the RAM dial; `--bags` extends negative coverage past it.

### predict flags

| flag | default | effect |
|---|---|---|
| `--fmax N` | 36 | forecast through hour N |
| `--workers N` | 4 | forecast hours assembled/predicted concurrently (own process each) |
| `--members N` | 6 | time-lagged ensemble members |
| `--temporal` | off | include prev/next-hour fields (must match training) |
| `--no-plots` | off | write only the netCDF |

**Sizing:** deterministic-from-main makes each sample ~100 MB (coarse). 2 TB
holds ~20k samples; `samples = severe_hours × bands`. Download time ≈ total ÷
bandwidth; gigabit + `--workers ≈ cores` is CPU-bound at ~`cores` parallel
feature computes. The `--min-free-gb` guard makes overruns safe.

## Status

| Milestone | State |
|---|---|
| M1 grid + HRRR store | ✅ done |
| M2 report DB + labels (SVRGIS EF + tornado **tracks**; UNK-wind & EFU-tor dropped) | ✅ done |
| M3 feature engine (validated vs MetPy) | ✅ done |
| M4 issuance-capped time-lagged ensemble | ✅ done |
| M5 real fetch + dataset builder + mlx-boosting GBDT | ✅ done |
| M6 lead-banded training (LightGBM, bagging, regularization) + calibration + band-aware all-band predict | ✅ done |
| M7 products: SPC plots/animations, cumulative daily, netCDF, CIG mapping (hatched, outlined), smooth-median postprocess, IEM SPC-outlook vector contours (legacy SIGN + 2026 CIG tiers) | ✅ done |
| M8 full 2020–2023 build → train → backtest | ⏳ running |

Validation (vs MetPy, single column): Bunkers RM/LM, per-layer SRH (exact for
same motion), SBCAPE/MLCAPE within ~7%, DCAPE within ~25% (wet-bulb-seeded
descent). mlx-boosting verified for binary + multiclass.

## Environment

Conda env `met` (Python 3.14). Installed: numpy, scipy, pandas, xarray,
matplotlib, cartopy, metpy, requests, pyarrow, mlx, **mlx-boosting**,
**herbie-data**, **cfgrib**, **eccodes**.

> `data/grid.py` implements the spherical Lambert Conformal transform directly in
> NumPy (no pyproj dependency — PROJ resolution is fragile in this conda env).

## References reused
`Reference-ReportDB` (IEM LSR + SPC + SVRGIS report DB & tornado tracks),
`Reference-SWIPR` (events/labels parquet pattern). HRRR via the NOAA
`noaa-hrrr-bdp-pds` S3 bucket.
