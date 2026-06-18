"""FIGS — Forecasting Intensity Guidance for Severe weather.

A gradient-boosted-tree severe-weather model in the style of nadocast, trained on
time-lagged HRRR ensembles. Predicts per-forecast-hour hazard probabilities and
conditional intensity distributions for tornado / wind / hail, then renders
SPC-style probability + Conditional Intensity Guidance (CIG) products.

See plan: forecasting intensity guidance overview in the project plan file.
"""

__version__ = "0.0.1"
