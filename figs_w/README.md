# FIGS-W — Forecasting Intensity Guidance System: Wildfires

A sibling of FIGS (same repo, `figs_w/`) that **reuses the FIGS grid, HRRR
retrieval, feature engine, GBDT wrapper, calibration and plotting**, but retargets
to wildfires. Per forecast hour and ~15 km cell (25 mi SPC-style neighborhood):

- **p(wildfire)** — a new fire start nearby;
- **conditional size** distribution (0–25 / 25–100 / 100–250 / 250–1000 / 1000+ ac) → CIG.

> **Deadliness is not modeled.** The only casualty source covering the 2021+
> HRRRv4 era (NCEI Storm Events) geocodes wildfires only to large NWS *forecast
> zones*, too coarse to attribute deaths/damage to a specific fire. Size is the
> intensity target.

Products mirror the **SPC fire-weather outlook**: probability + CIG → **NONE /
ELEVATED / CRITICAL / EXTREME**.

## What changed vs FIGS (lighter weight)

| | FIGS | FIGS-W |
|---|---|---|
| reference motions | 5 (Bunkers RM/LM, mean, Corfidi up/down) | **2** (none, 0–6 km mean wind) |
| ensemble prob fields (REFC/REFD/UH exceedance) | yes | **no** (single deterministic run) |
| deterministic REFC/REFD/UH | yes | **kept** |
| time-lagged members | up to 6 | **1** (the run itself) |

**Added inputs, all full Tier-1 (means + gradients vs both motions):** SOILW,
TSOIL, cloud cover (low/mid/high/total), surface T/Td/T−Td/wind-speed, and the
**static geography** — terrain (elev/slope/aspect), **terrain texture**
(TRI/TPI/elev-std/slope-std), land use, population density.

Together (5→2 motions, no ensemble prob) this is a much smaller feature matrix
than FIGS.

## Data sources

- **HRRR** — shared GRIB cache with FIGS (same `Data/hrrr/`); just a smaller field
  subset + 2 motions.
- **Terrain + texture** — SRTM DEM via the **`elevation`** library, reprojected to
  the FIGS grid (`rasterio`); elevation/slope/aspect + ruggedness (TRI/TPI, elev/slope
  neighborhood std). Falls back to HRRR `zsfc` if those optional deps are absent.
- **Land use** — USGS **NLCD** GeoTIFF → per-cell fractional cover of forest / shrub /
  grass / crop / urban / water (each class reprojected as a 0/1 mask, average resampling).
- **Population density** — **WorldPop** USA GeoTIFF (people·km⁻²) → area-averaged per
  cell (+ log1p companion).

  `pip install elevation rasterio`; download the NLCD + WorldPop rasters once, then
  `python -m figs_w.cli build-static --nlcd nlcd_conus.tif --worldpop worldpop_usa.tif`.
  Cached to `Data/figs_w/static/{terrain,landuse,popdensity}.npz`; missing land-use /
  population fall back to zeros (warned) until built.
- **Fire reports** — a merged **fire catalog**, where each fire is an *interval*
  (large fires burn across many HRRR cycles), not a point:
  - **NIFC incident locations** → IRWIN id, discovery + containment time, point,
    preliminary size;
  - **NIFC perimeters / NIFS** → authoritative **final acres** (joined by IRWIN id,
    overrides the preliminary incident size);
  - **NIFC fire progression** → time-stamped perimeters → the fire's **footprint
    over time**, so it's stamped at *every valid hour it's active* using its
    perimeter as of that hour;
  - **IEM wildfire LSRs** → near-real-time fill, **deduplicated** against NIFC by
    space (~10 mi) + time (~24 h); IEM-only fires are kept occurrence-only (size
    unknown → excluded from the CIG/size fit).

  Labels (`data/labels.build_labels`) stamp the 25 mi neighborhood around the active
  footprint and store RAW values — occurrence + the largest nearby fire's **final**
  size (acres), binned at train/CIG time. Only the NIFC service URLs/field names
  still need confirming (below).

## Probability + CIG → fire-weather categorical

Categories: `NONE`(0) `ELEVATED`(1) `CRITICAL`(2) `EXTREME`(3). The CIG column is
the size-distribution category (`<CIG1..CIG3`). The mapping (per request):

| prob level | <CIG1 | CIG1 | CIG2 | CIG3 |
|---|---|---|---|---|
| L1 | NONE | ELEVATED | ELEVATED | CRITICAL |
| L2 | ELEVATED | ELEVATED | CRITICAL | CRITICAL |
| L3 | ELEVATED | CRITICAL | CRITICAL | EXTREME |
| L4 | CRITICAL | CRITICAL | EXTREME | EXTREME |
| L5 | CRITICAL | EXTREME | EXTREME | EXTREME |

The five **probability thresholds** (`config.PROB_LEVELS_PCT`) are **placeholders**
— set them from model inference / the practically-perfect forecast (PPF) of the
combined report set. The **CIG reference size distributions** (`config.CIG_REFERENCE`)
are placeholders too; fit them from the training size distribution with
`products.cig.fit_cig_reference(marginal)` (`marginal_size_distribution(parquet)`
reads the empirical marginal off a built matrix).

## Pipeline

```bash
conda activate met
python -m figs_w.cli build-terrain                                   # one-time static fields
python -m figs_w.cli build-data --start 2021-05-01 --end 2021-10-31 \
    --out Data/figs_w/processed/fw.parquet --min-fires 1
python -m figs_w.cli train --parquet Data/figs_w/processed/fw.parquet --backend lightgbm
python -m figs_w.cli predict --run 2024-08-15T18 --fmax 24
```

## Reuse map (imports from `figs`)

`figs.data.grid` (grid/regrid/neighborhood) · `figs.data.hrrr_store` (HRRR) ·
`figs.features.{profiles,thermo,sr_params,hodograph,kinematics,spatial}` +
`assemble` helpers · `figs.model.{wrapper,calibrate}` · `figs.products.plots`
(base axis/extent/CIG hatch). FIGS-W only adds the 2-motion frame set, the static
geography, the wildfire labels/targets, and the fire-weather color/category tables.

## Open items (need confirmation / later detail)

1. **NIFC service URLs + field names** — confirm `NIFC_INCIDENT_SERVICE` /
   `NIFC_PERIMETER_SERVICE` / `NIFC_PROGRESSION_SERVICE` and the
   `INCIDENT_FIELDS` / `PERIMETER_FIELDS` / `PROGRESSION_FIELDS` maps in
   `data/fire_reports.py` against the linked datasets (the query + merge logic is
   source-agnostic once the join keys/URLs are right).
2. **Merge / dedup — DONE** (NIFC-authoritative final size via IRWIN-id perimeter
   join, fire-progression footprint over the fire's lifetime, IEM space+time dedup).
   Tune `_DEDUP_MI` / `_DEDUP_HR` / `_DEFAULT_DURATION_HR` once validated on data.
3. **Land use + population density ingest — DONE** (NLCD fractional cover +
   WorldPop density via `rasterio` reprojection; SRTM terrain via `elevation`).
   Just needs the rasters downloaded + `build-static` run.
4. **Probability levels** — calibrate `PROB_LEVELS_PCT` from inference / PPF.
5. **CIG reference size distributions** — fit from training (`fit_cig_reference`);
   the 5 size bins may change once the data is in hand.
```
