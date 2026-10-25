"""Loading and cleaning of hourly air quality records."""
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
WEATHER = []  # this dataset has no weather columns


def load_data(path):
    """Read the CSV and sort it by station and time."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


def _iqr_filter(series, k=3.0):
    """Replace values outside Q1 - k*IQR and Q3 + k*IQR with NaN."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.where(series.between(q1 - k * iqr, q3 + k * iqr))


def clean_data(df):
    """Remove duplicates, invalid values and outliers, then fill short gaps."""
    df = df.drop_duplicates(subset=["station", "datetime"]).copy()
    cols = POLLUTANTS + WEATHER

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Negative concentrations are invalid
    df[POLLUTANTS] = df[POLLUTANTS].where(df[POLLUTANTS] >= 0)

    # Outliers are judged per station
    for col in POLLUTANTS:
        df[col] = df.groupby("station")[col].transform(_iqr_filter)

    # Fill gaps of up to 6 hours within each station
    parts = []
    for station, group in df.groupby("station"):
        group = group.set_index("datetime").sort_index()
        group[cols] = group[cols].interpolate(method="time", limit=6, limit_direction="both")
        group["station"] = station
        parts.append(group.reset_index())

    cleaned = pd.concat(parts, ignore_index=True)
    return cleaned.dropna(subset=POLLUTANTS).reset_index(drop=True)