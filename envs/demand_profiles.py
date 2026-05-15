"""
envs/demand_profiles.py
=======================
Alternative BASE_DEMAND profiles for robustness experiments.

Each profile is a 24-element numpy array (one value per hour, 00–23).
Values are in the same normalised kWh units as the original environment.

Profiles
--------
RESIDENTIAL   : original profile — morning + evening peaks (urban household mix)
INDUSTRIAL    : flat high daytime load, drops at night (factories / offices)
SUMMER_AC     : midday peak driven by air-conditioning, hot-climate residential
WEEKEND       : no morning commute spike, late afternoon leisure peak
"""

import numpy as np

# ---------------------------------------------------------------------------
# Original residential profile (kept here for reference / import convenience)
# ---------------------------------------------------------------------------
RESIDENTIAL = np.array([
    0.40, 0.35, 0.30, 0.28, 0.28, 0.32,   # 00–05  night valley
    0.50, 0.70, 0.85, 0.80, 0.75, 0.70,   # 06–11  morning ramp + peak
    0.65, 0.60, 0.58, 0.60, 0.65, 0.75,   # 12–17  midday plateau
    0.90, 1.00, 0.95, 0.80, 0.65, 0.50,   # 18–23  evening peak
])

# ---------------------------------------------------------------------------
# Industrial: high flat load 06–20, low at night
# Typical of manufacturing / commercial districts
# ---------------------------------------------------------------------------
INDUSTRIAL = np.array([
    0.35, 0.32, 0.30, 0.30, 0.32, 0.38,   # 00–05  skeleton crew / HVAC base
    0.60, 0.80, 0.90, 0.92, 0.92, 0.90,   # 06–11  shift start, ramp up
    0.88, 0.88, 0.87, 0.88, 0.90, 0.85,   # 12–17  sustained production
    0.70, 0.55, 0.45, 0.40, 0.37, 0.35,   # 18–23  shift end, wind down
])

# ---------------------------------------------------------------------------
# Summer / AC-heavy: midday spike due to cooling loads, hot climate
# Morning and evening are more moderate than residential
# ---------------------------------------------------------------------------
SUMMER_AC = np.array([
    0.38, 0.34, 0.31, 0.29, 0.29, 0.33,   # 00–05  cool night, low load
    0.45, 0.58, 0.70, 0.82, 0.92, 0.98,   # 06–11  temperature rises, AC kicks in
    1.00, 0.98, 0.96, 0.93, 0.88, 0.80,   # 12–17  peak heat → peak AC load
    0.72, 0.65, 0.58, 0.52, 0.46, 0.41,   # 18–23  evening cool-down
])

# ---------------------------------------------------------------------------
# Weekend / leisure: no commute spike, later start, prolonged evening
# Lower overall magnitude, but shifted later in the day
# ---------------------------------------------------------------------------
WEEKEND = np.array([
    0.42, 0.38, 0.34, 0.31, 0.30, 0.31,   # 00–05  late-night socialising tail
    0.33, 0.38, 0.48, 0.58, 0.65, 0.68,   # 06–11  slow morning, brunch load
    0.65, 0.63, 0.62, 0.65, 0.70, 0.75,   # 12–17  midday leisure plateau
    0.82, 0.90, 0.92, 0.88, 0.75, 0.58,   # 18–23  prime-time / dinner peak
])

# ---------------------------------------------------------------------------
# Registry — used by the multi-demand wrapper and experiment script
# ---------------------------------------------------------------------------
ALL_PROFILES: dict[str, np.ndarray] = {
    "Residential": RESIDENTIAL,
    "Industrial":  INDUSTRIAL,
    "Summer_AC":   SUMMER_AC,
    "Weekend":     WEEKEND,
}