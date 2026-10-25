"""Air Quality Index (AQI) calculation using US EPA breakpoints."""
import numpy as np
import pandas as pd

# (conc_low, conc_high, index_low, index_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

CATEGORY_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def sub_index(conc, breakpoints):
    """Convert a concentration to its AQI sub-index using linear interpolation."""
    conc = np.asarray(conc, dtype=float)
    out = np.full(conc.shape, np.nan)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = (conc >= c_lo) & (conc <= c_hi)
        out[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (conc[mask] - c_lo) + i_lo
    out[conc > breakpoints[-1][1]] = 500
    return out


def categorize(aqi):
    bins = [-np.inf, 50, 100, 200, 300, 400, np.inf]
    return pd.cut(aqi, bins=bins, labels=CATEGORY_LABELS, right=True)


def compute_aqi(df):
    """Add AQI sub-indices, the overall AQI and its category to the table."""
    out = df.copy()
    # EPA truncates PM2.5 to one decimal and PM10 to a whole number before lookup
    out["AQI_PM2.5"] = sub_index(np.floor(out["PM2.5"] * 10) / 10, PM25_BREAKPOINTS)
    out["AQI_PM10"] = sub_index(np.floor(out["PM10"]), PM10_BREAKPOINTS)
    out["AQI"] = np.fmax(out["AQI_PM2.5"], out["AQI_PM10"])
    out["AQI_category"] = categorize(out["AQI"])
    return out