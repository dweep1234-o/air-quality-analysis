"""Check whether PM2.5 values are predictable from earlier hours."""
from src.preprocessing import load_data

df = load_data("data/air_quality.csv")

print(f"{'City':<12}{'Lag-1 corr':>12}{'Lag-24 corr':>14}")
for station, group in df.groupby("station"):
    series = group.sort_values("datetime")["PM2.5"].dropna()
    print(f"{station:<12}{series.autocorr(1):>12.3f}{series.autocorr(24):>14.3f}")