"""FIGS-W configuration. Reuses the FIGS grid, vertical levels and HRRR retrieval
config wholesale (``from figs import config as F``); only the wildfire-specific
pieces — motions, the curated field set, the static-geography fields, targets,
size/CIG bins and the fire-weather categorical mapping — are (re)defined here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from figs import config as F

# --------------------------------------------------------------------------- #
# Paths — share the HRRR GRIB cache with FIGS (same model data!); keep FIGS-W's
# processed matrices / models / products under their own Data/figs_w/* dirs.
# --------------------------------------------------------------------------- #
HRRR_CACHE = F.HRRR_CACHE
GRID_CACHE = F.GRID_CACHE
DATA_ROOT = F.DATA_ROOT
W_ROOT = DATA_ROOT / "figs_w"
PROCESSED = W_ROOT / "processed"
MODELS = W_ROOT / "models"
PRODUCTS = W_ROOT / "products"
STATIC_CACHE = W_ROOT / "static"        # regridded terrain / land-use / pop-density
REPORTS_CACHE = W_ROOT / "reports"      # NIFC + IEM fire report cache
for _p in (PROCESSED, MODELS, PRODUCTS, STATIC_CACHE, REPORTS_CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# Reused grid / vertical sampling (identical to FIGS so the GRIB cache + regrid
# weights are shared).
FIGS_NY, FIGS_NX = F.FIGS_NY, F.FIGS_NX
HRRR_GRID = F.HRRR_GRID
FETCH_ISOBARIC_LEVELS = F.FETCH_ISOBARIC_LEVELS
PROFILE_DEPTHS = F.PROFILE_DEPTHS
KINEMATIC_LEVELS = F.KINEMATIC_LEVELS
DIFFERENTIAL_DIVERGENCE = F.DIFFERENTIAL_DIVERGENCE
MANDATORY_HGT_LEVELS = F.MANDATORY_HGT_LEVELS
TMP_DPT_SPATIAL_LEVELS = F.KINEMATIC_LEVELS

# --------------------------------------------------------------------------- #
# Motions — wildfire spread is wind-aligned, not storm-relative; we keep just two
# reference frames (vs FIGS's five): zero-motion (ground-relative) and the 0–6 km
# mean wind. Every per-motion feature (SR winds, rotated hodographs, spatial
# gradients) is computed for these two only → a large weight cut vs FIGS.
# --------------------------------------------------------------------------- #
MOTION_VECTORS = ("none", "mean_wind")
SR_LAYERS = F.SR_LAYERS                 # reuse the AGL layers for SR winds

# --------------------------------------------------------------------------- #
# Spatial smoothing (same machinery / radii as FIGS).
# --------------------------------------------------------------------------- #
SPATIAL_MEAN_RADII_MI = F.SPATIAL_MEAN_RADII_MI
GRADIENT_TYPES = F.GRADIENT_TYPES
PREDICT_SMOOTH_RADII_MI = F.PREDICT_SMOOTH_RADII_MI

# Tier-1 NON-motion scalars (means + gradients vs BOTH motions). Thermo + kinematics
# carry over; the wildfire additions (SOILW/TSOIL/cloud/surface T-Td) and the static
# geography fields join this set so they get the full treatment.
SPATIAL_BASE_FIELDS = (
    # thermodynamics (same as FIGS)
    "sbcape", "mlcape", "dcape", "sbcin", "mlcin",
    "sbcape_0_90", "sbcape_0_180", "mlcape_0_90", "mlcape_0_180",
    "sbcin_0_90", "sbcin_0_180", "mlcin_0_90", "mlcin_0_180", "pwat",
    # kinematics
    "mb925_div", "mb925_conv", "mb925_absvort",
    "mb850_div", "mb850_conv", "mb850_absvort",
    "mb500_div", "mb500_conv", "mb500_absvort",
    "mb250_div", "mb250_conv", "mb250_absvort",
    "diffdiv_250_850",
    # WILDFIRE additions — fuel/moisture/insolation state, full Tier-1 treatment
    "soilw", "tsoil", "lcdc", "mcdc", "hcdc", "tcdc",
    "sfc_t2m", "sfc_td2m", "sfc_tdspread2m", "sfc_wspd10",
    # deterministic reflectivity / updraft helicity (kept; ensemble prob dropped)
    "refc", "refd", "uh03", "uh25",
)

# Static geography fields (regridded once to the FIGS grid, broadcast to every
# valid time). ALL get the full Tier-1 treatment. Land-use and population-density
# specifics are TBD ("detailed later"); terrain + terrain-texture are derived from
# the HRRR surface geopotential we already fetch (see data/static.py).
STATIC_TERRAIN_FIELDS = ("elev", "slope", "elev_gradx", "elev_grady", "aspect_sin", "aspect_cos")
STATIC_TERRAIN_TEXTURE_FIELDS = ("tri", "tpi", "slope_std", "elev_std")  # ruggedness/texture
STATIC_LANDUSE_FIELDS = ("lu_forest", "lu_shrub", "lu_grass", "lu_crop", "lu_urban", "lu_water")
STATIC_POPDENSITY_FIELDS = ("popdens", "popdens_log")
STATIC_FIELDS = (STATIC_TERRAIN_FIELDS + STATIC_TERRAIN_TEXTURE_FIELDS
                 + STATIC_LANDUSE_FIELDS + STATIC_POPDENSITY_FIELDS)

SR_SCALARS = ("srh", "srw_spd", "swfrac", "swv")   # Tier-2 own-frame (2 motions)
INCLUDE_HODOGRAPH = True

# Surface scalar point fields carried straight through (native CAPE/CIN, soil, cloud,
# moisture). Reflectivity/UH are also kept as point fields and Tier-1 fields above.
SURFACE_POINT_FIELDS = (
    "hrrr_sbcape", "hrrr_sbcin", "hrrr_mlcape90", "hrrr_mlcin90",
    "hrrr_mlcape180", "hrrr_mlcin180",
    "soilw", "tsoil", "pwat", "lcdc", "mcdc", "hcdc", "tcdc",
    "refc", "refd", "uh03", "uh25",
)

# --------------------------------------------------------------------------- #
# Targets and conditional size bins
# --------------------------------------------------------------------------- #
HAZARDS = ("wildfire",)                 # single "hazard" → reuses the FIGS model loop
NEIGHBORHOOD_RADIUS_MI = 25.0           # SPC-style neighborhood (probability + deadliness)
NEIGHBORHOOD_TIME_MIN = F.NEIGHBORHOOD_TIME_MIN

# A fire counts as "deadly"/significant if it caused a fatality or destroyed
# structures within the neighborhood (drives the deadliness model). TBD as the
# combined report schema firms up; structures-destroyed threshold is a placeholder.
DEADLY_STRUCTURES_THRESHOLD = 1

# Conditional SIZE distribution (the CIG target): final-size acreage bins.
INTENSITY_BINS = {
    "wildfire": dict(
        labels=("0-25", "25-100", "100-250", "250-1000", "1000+"),
        edges=(0.0, 25.0, 100.0, 250.0, 1000.0),   # acres; >= edge → that/next bin
        kind="acres",
    ),
}

# --------------------------------------------------------------------------- #
# CIG (Conditional Intensity Guidance) reference SIZE distributions.
#
# Per CIG category, the frequency of each size bin (ordered as in INTENSITY_BINS).
# Unlike FIGS's hand-set charts, these are intended to be FIT FROM THE TRAINING
# SIZE DISTRIBUTION (see products/cig.fit_cig_reference) — the values below are a
# monotone-by-tail placeholder so the pipeline runs before the fit is available.
# --------------------------------------------------------------------------- #
CIG_CATEGORIES = ("<CIG1", "CIG1", "CIG2", "CIG3")
CIG_REFERENCE = {
    "wildfire": {
        "<CIG1": (70.0, 20.0, 6.0, 3.0, 1.0),     # mostly small fires
        "CIG1":  (45.0, 30.0, 15.0, 7.0, 3.0),
        "CIG2":  (25.0, 28.0, 24.0, 15.0, 8.0),
        "CIG3":  (10.0, 18.0, 24.0, 26.0, 22.0),  # heavy large-fire tail
    },
}

# --------------------------------------------------------------------------- #
# Probability + CIG → SPC FIRE-WEATHER categorical (per the requested mapping).
# Categories: 0 NONE, 1 ELEVATED, 2 CRITICAL, 3 EXTREME.
# Rows are (probability % threshold, (<CIG1, CIG1, CIG2, CIG3) → category), ascending.
#
# The five probability thresholds are PLACEHOLDERS — the real levels are TBD and
# should be set from model inference / the practically-perfect-forecast (PPF) of
# the combined fire report dataset (see README). The CATEGORY tuples ARE the
# requested level-1..5 mapping and should not change:
#   L1: NONE, ELEVATED, ELEVATED, CRITICAL
#   L2: ELEVATED, ELEVATED, CRITICAL, CRITICAL
#   L3: ELEVATED, CRITICAL, CRITICAL, EXTREME
#   L4: CRITICAL, CRITICAL, EXTREME, EXTREME
#   L5: CRITICAL, EXTREME, EXTREME, EXTREME
# --------------------------------------------------------------------------- #
CATEGORY_NAMES = {0: "NONE", 1: "ELEVATED", 2: "CRITICAL", 3: "EXTREME"}

# placeholder probability thresholds (%) for levels 1..5 — RECALIBRATE from PPF.
PROB_LEVELS_PCT = (2.0, 5.0, 10.0, 20.0, 35.0)

CIG_CONVERSION = {
    "wildfire": [
        (PROB_LEVELS_PCT[0], (0, 1, 1, 2)),   # L1: NONE / ELEVATED / ELEVATED / CRITICAL
        (PROB_LEVELS_PCT[1], (1, 1, 2, 2)),   # L2
        (PROB_LEVELS_PCT[2], (1, 2, 2, 3)),   # L3
        (PROB_LEVELS_PCT[3], (2, 2, 3, 3)),   # L4
        (PROB_LEVELS_PCT[4], (2, 3, 3, 3)),   # L5
    ],
}

# Fire-weather probability fill levels (fractions) for the probability product —
# the same five levels, as a probability contour set. RECALIBRATE with PROB_LEVELS_PCT.
SPC_PROB_LEVELS = {"wildfire": tuple(p / 100.0 for p in PROB_LEVELS_PCT)}

# --------------------------------------------------------------------------- #
# Lead-time bands + weekly split — reuse FIGS.
# --------------------------------------------------------------------------- #
LEAD_BANDS = F.LEAD_BANDS
bands_for_fxx = F.bands_for_fxx
split_for_date = F.split_for_date
HRRRV4_START = F.HRRRV4_START
