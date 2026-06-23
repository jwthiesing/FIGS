"""FIGS-W — Forecasting Intensity Guidance System: Wildfires.

A sibling of FIGS that reuses the FIGS grid, HRRR retrieval, profile/thermo/
kinematic/spatial feature machinery, GBDT wrapper, calibration and plotting, but
swaps the target (severe hazards → wildfire) and the inputs:

  * a **lighter** HRRR feature set — only **2** reference motions (no-motion +
    mean wind) instead of 5, and **no** time-lagged ensemble probability fields
    (the deterministic REFC/REFD/UH are kept);
  * **added** environment: SOILW, TSOIL, cloud cover and surface T/Td get the full
    Tier-1 spatial treatment;
  * **static** geography: terrain, land use and population density (regridded to
    the FIGS grid, cached once) get the full Tier-1 treatment.

Targets (per forecast hour, per ~15 km cell, within a 25 mi neighborhood):
  * ``p(wildfire)``           — a new wildfire start nearby;
  * conditional **size** distribution (5 acreage bins) → CIG categories.

Products mirror the SPC **fire-weather** outlook: probability + CIG → NONE /
ELEVATED / CRITICAL / EXTREME.
"""
