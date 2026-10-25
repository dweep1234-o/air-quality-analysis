"""Streamlit web page for the air quality analysis project."""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path("outputs")
FIGURES = OUT / "figures"

st.set_page_config(page_title="Air Quality Analysis", layout="wide")
st.title("Air Quality Analysis")
st.caption("Hourly PM2.5, PM10 and other pollutants for five Indian cities, 2015 to 2024")


@st.cache_data
def load_outputs():
    cleaned = pd.read_csv(OUT / "cleaned_air_quality.csv", parse_dates=["datetime"])
    metrics = pd.read_csv(OUT / "forecast_metrics.csv")
    categories = pd.read_csv(OUT / "aqi_category_distribution.csv", index_col=0)
    importances = pd.read_csv(OUT / "feature_importance.csv", index_col=0)
    return cleaned, metrics, categories, importances


if not (OUT / "cleaned_air_quality.csv").exists():
    st.error("No results found. Run `python main.py` first.")
    st.stop()

cleaned, metrics, categories, importances = load_outputs()

tab_overview, tab_charts, tab_forecast = st.tabs(["Overview", "Charts", "Forecast"])

with tab_overview:
    st.subheader("Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(cleaned):,}")
    col2.metric("Cities", cleaned["station"].nunique())
    col3.metric("Mean PM2.5 (µg/m³)", f"{cleaned['PM2.5'].mean():.1f}")

    st.subheader("AQI category distribution")
    st.bar_chart(categories["percent"])

    st.subheader("Explore a city")
    city = st.selectbox("City", sorted(cleaned["station"].unique()))
    pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])
    subset = cleaned[cleaned["station"] == city].set_index("datetime")
    st.line_chart(subset[pollutant].resample("D").mean())

with tab_charts:
    st.subheader("Saved figures")
    for name, caption in [
        ("monthly_trend.png", "Monthly average concentration"),
        ("daily_cycle.png", "Average daily cycle"),
        ("correlation_heatmap.png", "Correlation matrix"),
        ("aqi_by_station.png", "AQI distribution by city"),
        ("aqi_categories.png", "Hours spent in each AQI category"),
    ]:
        path = FIGURES / name
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)

with tab_forecast:
    st.subheader("Model comparison")
    st.dataframe(metrics, hide_index=True)
    st.info(
        "The Random Forest does not beat the average value (R² ≈ 0). "
        "The data shows no time pattern, so this result is expected. "
        "See REPORT.md for details."
    )
    st.subheader("Feature importance")
    st.bar_chart(importances["importance"])