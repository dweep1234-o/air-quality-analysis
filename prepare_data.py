"""Convert the city hourly dataset into the format used by this project."""
from pathlib import Path

import pandas as pd

SOURCE = Path("data/city_hour.csv")
OUTPUT = Path("data/air_quality.csv")

df = pd.read_csv(SOURCE, parse_dates=["Datetime"])
df = df.rename(columns={"Datetime": "datetime", "City": "station"})
df = df[["datetime", "station", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
df.to_csv(OUTPUT, index=False)
print(f"Wrote {len(df):,} rows to {OUTPUT}")  