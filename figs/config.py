"""Central configuration for FIGS: paths, grid definition, vertical levels,
motion vectors, storm-relative layers, hazard/intensity bins, thresholds, and
lead-time bands.

Everything that another module needs to agree on lives here so the data,
feature, model, and product layers stay consistent. No heavy dependencies are
imported at module load (only ``numpy`` and stdlib); HRRR/Herbie/mlx imports are
deferred to the modules that use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"

HRRR_CACHE = DATA_ROOT / "hrrr"          # raw/subset GRIB cache (Herbie)
GRID_CACHE = DATA_ROOT / "grid"          # regrid weights + neighborhood indices
PROCESSED = DATA_ROOT / "processed"      # feature/label parquet
REPORTS_CACHE = DATA_ROOT / "reports"    # combined report DB cache
MODELS = DATA_ROOT / "models"            # trained GBDT models + calibrators
PRODUCTS = DATA_ROOT / "products"        # output plots / animations

for _p in (HRRR_CACHE, GRID_CACHE, PROCESSED, REPORTS_CACHE, MODELS, PRODUCTS):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# HRRR native grid (CONUS, Lambert Conformal Conic) and FIGS output grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LambertGrid:
    """HRRR CONUS Lambert Conformal grid parameters (operational v4)."""

    nx: int = 1799
    ny: int = 1059
    dx: float = 3000.0          # m
    dy: float = 3000.0          # m
    lat_1: float = 38.5         # standard parallels
    lat_2: float = 38.5
    lat_0: float = 38.5         # projection origin latitude
    lon_0: float = -97.5        # central meridian
    # Lower-left corner (first grid point) lat/lon:
    sw_lat: float = 21.138123
    sw_lon: float = -122.719528
    earth_radius: float = 6371229.0

    def proj_params(self) -> dict:
        """pyproj/cartopy Lambert Conformal parameter dict (spherical earth)."""
        return dict(
            proj="lcc",
            lat_1=self.lat_1,
            lat_2=self.lat_2,
            lat_0=self.lat_0,
            lon_0=self.lon_0,
            R=self.earth_radius,
        )


HRRR_GRID = LambertGrid()

# FIGS predicts on HRRR downsampled by this factor (nadocast-style ~15 km).
BLOCK = 5                                    # 3 km * 5 = 15 km
FIGS_DX_KM = HRRR_GRID.dx * BLOCK / 1000.0   # 15.0 km nominal cell size
FIGS_NX = HRRR_GRID.nx // BLOCK              # 359
FIGS_NY = HRRR_GRID.ny // BLOCK              # 211


# --------------------------------------------------------------------------- #
# Vertical sampling for fine TMP/DPT/wind profiles.
#
# Levels are defined as PRESSURE DEPTH BELOW THE SURFACE (mb), so the actual
# target pressure for a column is  p_target = p_surface - depth. Spacing is finest
# in the boundary layer and coarsens aloft:
#   * every 25 mb in the lowest 150 mb above the surface,
#   * every 50 mb from 150 to 500 mb above the surface,
#   * every 100 mb above that, up to PROFILE_TOP_DEPTH.
# --------------------------------------------------------------------------- #
PROFILE_TOP_DEPTH = 700.0  # mb above surface (≈ up to ~300 mb for a 1000 mb sfc)


def profile_depths() -> np.ndarray:
    """Pressure depths (mb below surface) at which to sample the profiles."""
    low = np.arange(25.0, 150.0 + 1e-6, 25.0)                # 25..150 by 25
    mid = np.arange(200.0, 500.0 + 1e-6, 50.0)               # 200..500 by 50
    high = np.arange(600.0, PROFILE_TOP_DEPTH + 1e-6, 100.0)  # 600..700 by 100
    return np.concatenate([[0.0], low, mid, high])           # include the surface


PROFILE_DEPTHS = profile_depths()

# Full HRRR isobaric levels (mb) available in wrfprsf (reference / synthetic tests).
HRRR_ISOBARIC_LEVELS = np.array(
    [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
     675, 650, 625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375,
     350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100],
    dtype=float,
)

# Levels we actually DOWNLOAD. To save storage/bandwidth the fetched set mirrors
# the profile spacing (25 mb low, 50 mb mid, ~100 mb aloft) and DROPS the
# 650/550/450/250/150 mb levels — an intentional, fixed space saving (~84 MB per
# member). Features nominally "at" the dropped pressures (e.g. 250 mb divergence,
# 250/150 mb HGT, 250 mb T/Td) are still produced — they're interpolated in log-p
# from the neighboring retained levels. This is THE level set; it must stay fixed
# so the Herbie subset cache (keyed by the search string) keeps matching.
FETCH_ISOBARIC_LEVELS = np.array(
    [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
     600, 500, 400, 300, 200, 100],
    dtype=float,
)

# Levels (mb) used for kinematic fields (divergence/convergence/abs vorticity).
KINEMATIC_LEVELS = (925.0, 850.0, 500.0, 250.0)
DIFFERENTIAL_DIVERGENCE = (250.0, 850.0)  # upper minus lower

# Mandatory pressure levels (mb) at which to include geopotential height as a
# feature (nadocast convention).
MANDATORY_HGT_LEVELS = (925.0, 850.0, 700.0, 500.0, 300.0, 250.0, 200.0, 150.0, 100.0)

# Pressure levels at which TMP/DPT are added as point features AND given the full
# tier-1 spatial treatment (means + all-motion gradients). Reuse the levels we
# already carry (the kinematic levels) rather than introducing new ones.
TMP_DPT_SPATIAL_LEVELS = KINEMATIC_LEVELS  # (925, 850, 500, 250) mb

# Surface scalar fields passed straight through as point features.
SURFACE_POINT_FIELDS = (
    "hrrr_sbcape", "hrrr_sbcin", "hrrr_mlcape90", "hrrr_mlcin90",
    "hrrr_mlcape180", "hrrr_mlcin180",
    "soilw", "tsoil", "pwat", "lcdc", "mcdc", "hcdc", "tcdc",
    "crain", "cfrzr", "cicep", "csnow",
    "relv01", "relv02",  # HRRR low-level (0-1 km, 0-2 km) relative vorticity
)


# --------------------------------------------------------------------------- #
# Storm-motion / reference vectors and storm-relative layers
# --------------------------------------------------------------------------- #
# Each motion vector gets: hodograph rotation, forward/leftward/straddling
# spatial gradients, and per-layer SR wind / streamwise vorticity / SRH.
MOTION_VECTORS = ("bunkers_rm", "bunkers_lm", "mean_0_6km", "corfidi_up", "corfidi_down")

# AGL layers (m) for SR wind, streamwise vorticity, SRH.
SR_LAYERS = (
    ("0_500m", 0.0, 500.0),
    ("500_1000m", 500.0, 1000.0),
    ("1_3km", 1000.0, 3000.0),
    ("3_6km", 3000.0, 6000.0),
)


# --------------------------------------------------------------------------- #
# Spatial means and gradients
# --------------------------------------------------------------------------- #
SPATIAL_MEAN_RADII_MI = (25.0, 50.0, 100.0)
GRADIENT_TYPES = ("forward", "leftward", "straddling")

# Output post-processing: before any prob/CIG/categorical product, each raw model
# prediction grid is Gaussian-smoothed at these radii (miles) and the per-cell
# MEDIAN across the set is taken as the final prediction. 0 mi = the raw grid, so
# the median of {raw, 25 mi, 50 mi} tempers single-cell noise while preserving
# real maxima (a robust de-speckle that a single heavy smooth would wash out).
PREDICT_SMOOTH_RADII_MI = (0.0, 25.0, 50.0)

# Spatial smoothing scope (features/assemble.py applies it). Two tiers:
#
# 1. SPATIAL_BASE_FIELDS — NON-motion environmental scalars. These get the full
#    treatment: 25/50/100 mi means + forward/leftward/straddling gradients of
#    each mean relative to ALL motion vectors (they aren't tied to any one
#    frame). Each expands to len(RADII)*(1 + len(MOTIONS)*len(GRADIENT_TYPES))
#    columns (= 48 at 5 motions).
#
# 2. SPATIAL_SR_SCALARS — motion-relative SR scalar components. Every computed
#    feature named ``{motion}_{layer}_{component}`` for these components is
#    smoothed (25/50/100 mi means) and gradient-ed ONLY in its OWN motion frame
#    (a cross-frame gradient of, e.g., an RM-frame SRH would be meaningless).
#    Each expands to len(RADII)*(1 + len(GRADIENT_TYPES)) columns (= 12).
#    The raw wind components (rotated hodograph u/v, srw_u/srw_v) are NOT smoothed.
SPATIAL_BASE_FIELDS = (
    # thermodynamics
    "sbcape", "mlcape", "dcape", "sbcin", "mlcin",
    "sbcape_0_90", "sbcape_0_180", "mlcape_0_90", "mlcape_0_180",
    "sbcin_0_90", "sbcin_0_180", "mlcin_0_90", "mlcin_0_180", "pwat",
    # kinematics / dynamics
    "mb925_div", "mb925_conv", "mb925_absvort",
    "mb850_div", "mb850_conv", "mb850_absvort",
    "mb500_div", "mb500_conv", "mb500_absvort",
    "mb250_div", "mb250_conv", "mb250_absvort",
    "diffdiv_250_850",
)
SPATIAL_SR_SCALARS = ("srh", "srw_spd", "swfrac", "swv")
# Also give the rotated hodograph wind components ('{motion}_u{h}'/'{motion}_v{h}')
# the tier-2 own-frame treatment. Large add (~110 fields x 12 cols), but they are
# already rotated into each motion frame so own-frame gradients are meaningful.
SPATIAL_INCLUDE_HODOGRAPH = True


# --------------------------------------------------------------------------- #
# Ensemble (time-lagged HRRR)
# --------------------------------------------------------------------------- #
ENSEMBLE_MAX_MEMBERS = 6     # most-recent runs reaching a given valid time
HRRR_LONG_CYCLES = (0, 6, 12, 18)   # 48-h runs (reach Day 2)
HRRR_SHORT_LEN = 18          # forecast length for non-48h cycles
HRRR_LONG_LEN = 48

# Member-exceedance probability fields are limited to reflectivity + UH.
REFC_THRESHOLDS_DBZ = (10, 20, 30, 40, 50)
REFD_THRESHOLDS_DBZ = (30, 40, 50)          # 1 km AGL reflectivity
UH_03KM_THRESHOLDS = (25, 75, 150)          # m^2/s^2, 0–3 km updraft helicity
UH_25KM_THRESHOLDS = (25, 75, 150, 300)     # m^2/s^2, 2–5 km updraft helicity


# --------------------------------------------------------------------------- #
# Hazards and conditional-intensity bins
# --------------------------------------------------------------------------- #
HAZARDS = ("tor", "wind", "hail")

# Severe / significant-severe report thresholds (nadocast convention).
SEVERE_THRESHOLDS = dict(tor_ef=0, wind_kt=50.0, hail_in=1.0)
SIGNIFICANT_THRESHOLDS = dict(tor_ef=2, wind_kt=65.0, hail_in=2.0)

# Conditional-intensity bin edges. Bins are labeled by index 0..n-1 and match
# the FIGS output spec and the CIG chart categories.
INTENSITY_BINS = {
    # tornado: EF rating buckets
    "tor": dict(
        labels=("EF0", "EF1", "EF2", "EF3", "EF4+"),
        # bin by integer EF; EF4+ catches 4 and 5
        edges=(0, 1, 2, 3, 4),  # >= edge assigns to that/next bucket; see labels.py
        kind="ef",
    ),
    # wind: estimated/measured gust in knots
    "wind": dict(
        labels=("50-55", "56-64", "65-73", "74-82", "83+"),
        edges=(50.0, 56.0, 65.0, 74.0, 83.0),
        kind="kt",
    ),
    # hail: max diameter in inches
    "hail": dict(
        labels=("1-1.49", "1.5-1.99", "2-3.49", "3.5+"),
        edges=(1.0, 1.5, 2.0, 3.5),
        kind="in",
    ),
}


# --------------------------------------------------------------------------- #
# CIG (Conditional Intensity Guidance) reference distributions.
#
# For each hazard and CIG category, the historical frequency (percent) of each
# conditional-intensity bin (ordered as in INTENSITY_BINS[...]["labels"]). These
# define the CIG categories: products/cig.py maps a predicted conditional-
# intensity distribution to a CIG category by comparison against these.
# Values are raw chart percentages and are normalized to sum to 1 where used.
#
# Hail CIG3 is a user-specified extension (the source chart only shows hail to
# CIG2); the other rows come directly from the CIG charts.
# --------------------------------------------------------------------------- #
CIG_CATEGORIES = ("<CIG1", "CIG1", "CIG2", "CIG3")

CIG_REFERENCE = {
    "tor": {
        "<CIG1": (61.0, 32.0, 6.0, 1.0, 0.1),
        "CIG1": (42.0, 37.0, 14.0, 5.0, 1.0),
        "CIG2": (36.0, 33.0, 18.0, 9.0, 3.0),
        "CIG3": (31.0, 28.0, 22.0, 14.0, 5.0),
    },
    "wind": {
        "<CIG1": (74.0, 20.0, 5.0, 1.0, 0.1),
        "CIG1": (55.0, 31.0, 10.0, 2.0, 1.0),
        "CIG2": (44.0, 33.0, 17.0, 4.0, 2.0),
        "CIG3": (33.0, 34.0, 23.0, 7.0, 4.0),
    },
    "hail": {
        "<CIG1": (65.0, 27.0, 8.0, 0.1),
        "CIG1": (51.0, 32.0, 15.0, 2.0),
        "CIG2": (40.0, 35.0, 20.0, 5.0),
        "CIG3": (29.0, 38.0, 25.0, 8.0),  # user-specified extension
    },
}


# --------------------------------------------------------------------------- #
# Probability -> SPC categorical-risk conversion tables.
#
# Given a hazard probability and the derived CIG category, look up the SPC
# categorical outlook level. Categories: 0 none/TSTM, 1 MRGL, 2 SLGT, 3 ENH,
# 4 MDT, 5 HIGH. Each row is (prob_threshold_percent, categories_by_cig_index)
# with CIG index 0=<CIG1, 1=CIG1, 2=CIG2, 3=CIG3; None = "not used" (capped).
# Rows are ascending in probability; a value maps to the highest row whose
# threshold it meets. Below the lowest threshold -> category 0.
# --------------------------------------------------------------------------- #
CATEGORY_NAMES = {0: "TSTM", 1: "MRGL", 2: "SLGT", 3: "ENH", 4: "MDT", 5: "HIGH"}

CIG_CONVERSION = {
    "tor": [
        (2.0, (1, 1, 2, None)),
        (5.0, (2, 2, 3, None)),
        (10.0, (2, 3, 3, 3)),
        (15.0, (3, 3, 4, 4)),
        (30.0, (3, 4, 5, 5)),
        (45.0, (3, 4, 5, 5)),
        (60.0, (3, 5, 5, 5)),
    ],
    "wind": [
        (5.0, (1, 1, 2, None)),
        (15.0, (2, 2, 3, None)),
        (30.0, (2, 3, 3, None)),
        (45.0, (3, 3, 4, 5)),
        (60.0, (3, 4, 5, 5)),
        (75.0, (3, 4, 5, 5)),
        (90.0, (3, 4, 5, 5)),
    ],
    "hail": [  # CIG3 (last column) is a user-specified extension
        (5.0, (1, 1, 2, 2)),
        (15.0, (2, 2, 3, 3)),
        (30.0, (2, 3, 3, 4)),
        (45.0, (3, 3, 4, 5)),
        (60.0, (3, 4, 4, 5)),
    ],
}


# --------------------------------------------------------------------------- #
# Labeling neighborhood (nadocast convention)
# --------------------------------------------------------------------------- #
NEIGHBORHOOD_RADIUS_MI = 25.0
NEIGHBORHOOD_TIME_MIN = 30.0


# --------------------------------------------------------------------------- #
# Lead-time bands (forecast hours): 6-hour, non-overlapping. Each band trains a
# separate model on the samples whose forecast hour falls in its range (the
# sampled leads 6/12/18/24/30/36/42/48 each land in exactly one band). At
# prediction time EVERY forecast hour ensembles ALL band models (a lead-diverse
# ensemble), so finer bands add members rather than narrowing coverage.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeadBand:
    name: str
    fmin: int
    fmax: int


LEAD_BANDS = (
    LeadBand("f1_6", 1, 6),
    LeadBand("f7_12", 7, 12),
    LeadBand("f13_18", 13, 18),
    LeadBand("f19_24", 19, 24),
    LeadBand("f25_30", 25, 30),
    LeadBand("f31_36", 31, 36),
    LeadBand("f37_42", 37, 42),
    LeadBand("f43_48", 43, 48),
)


def bands_for_fxx(fxx: int) -> list[LeadBand]:
    """Return the lead-time band(s) covering forecast hour ``fxx`` (used for
    training-sample assignment; prediction ensembles all bands)."""
    return [b for b in LEAD_BANDS if b.fmin <= fxx <= b.fmax]


# --------------------------------------------------------------------------- #
# Training span and weekly split
# --------------------------------------------------------------------------- #
HRRRV4_START = "2020-12-02"   # HRRRv4 operational

# Official SPC probabilistic-outlook levels (fractions). Tornado uses a 2% floor
# and a 10% level; wind and hail start at 5% and omit 2%/10%.
SPC_PROB_LEVELS = {
    "tor": (0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60),
    "wind": (0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90),  # SPC added 75/90%
    "hail": (0.05, 0.15, 0.30, 0.45, 0.60),
}


def split_for_date(dt) -> str:
    """nadocast weekly split: weekday->train, Saturday->validation, Sunday->test.
    ``dt`` is a date/datetime; uses ``weekday()`` (Mon=0 .. Sun=6)."""
    wd = dt.weekday()
    if wd == 5:
        return "validation"
    if wd == 6:
        return "test"
    return "train"
