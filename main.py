"""Run the full air quality analysis pipeline.

Usage:
    python main.py                      # uses data/air_quality.csv
    python main.py --data my_data.csv   # use a different dataset
"""
import argparse
from pathlib import Path

from src import analysis
from src.aqi import compute_aqi
from src.forecasting import run_forecast
from src.preprocessing import clean_data, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Air quality analysis pipeline")
    parser.add_argument("--data", default="data/air_quality.csv", help="path to the input CSV")
    parser.add_argument("--output", default="outputs", help="folder for results")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run prepare_data.py first.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("1. Loading and cleaning data")
    raw = load_data(data_path)
    print(f"   {len(raw):,} raw records from {raw['station'].nunique()} cities")
    clean = compute_aqi(clean_data(raw))
    print(f"   {len(clean):,} records after cleaning")
    clean.to_csv(out / "cleaned_air_quality.csv", index=False)

    print("2. Exploratory and trend analysis")
    analysis.summary_statistics(clean).to_csv(out / "summary_statistics.csv")
    analysis.station_comparison(clean).to_csv(out / "city_comparison.csv")
    analysis.monthly_trend(clean).to_csv(out / "monthly_trend.csv")
    categories = analysis.category_distribution(clean)
    categories.to_csv(out / "aqi_category_distribution.csv")
    analysis.save_figures(clean, out / "figures")
    print("   AQI category distribution (% of hours):")
    print(categories.to_string())

    print("3. Forecasting PM2.5 (this can take a few minutes)")
    metrics, predictions, importances = run_forecast(clean)
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    predictions.to_csv(out / "forecast_predictions.csv", index=False)
    importances.to_csv(out / "feature_importance.csv", header=["importance"])
    print(metrics.to_string(index=False))

    print(f"\nDone. Results saved to {out.resolve()}")


if __name__ == "__main__":
    main()