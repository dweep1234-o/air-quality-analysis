"""Exploratory analysis, trend analysis and figure generation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .aqi import CATEGORY_LABELS  # noqa: E402
from .preprocessing import POLLUTANTS, WEATHER  # noqa: E402


def summary_statistics(df):
    return df[POLLUTANTS + WEATHER].describe().T.round(2)


def monthly_trend(df):
    return df.groupby(df["datetime"].dt.month)[POLLUTANTS].mean().round(2)


def hourly_profile(df):
    return df.groupby(df["datetime"].dt.hour)[["PM2.5", "NO2"]].mean().round(2)


def station_comparison(df):
    return df.groupby("station")[POLLUTANTS + ["AQI"]].mean().round(2)


def correlation_matrix(df):
    return df[POLLUTANTS + WEATHER].corr().round(2)


def category_distribution(df):
    counts = df["AQI_category"].value_counts().reindex(CATEGORY_LABELS).fillna(0).astype(int)
    return (counts / counts.sum() * 100).round(2).rename("percent")


def save_figures(df, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_trend(df)[["PM2.5", "PM10", "NO2"]].plot(marker="o", ax=ax)
    ax.set(title="Monthly average concentration", xlabel="Month", ylabel="µg/m³")
    fig.tight_layout()
    fig.savefig(out / "monthly_trend.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_profile(df).plot(marker="o", ax=ax)
    ax.set(title="Average daily cycle", xlabel="Hour of day", ylabel="µg/m³")
    fig.tight_layout()
    fig.savefig(out / "daily_cycle.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix(df), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="station", y="AQI", ax=ax)
    ax.set_title("AQI distribution by station")
    fig.tight_layout()
    fig.savefig(out / "aqi_by_station.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    hours_per_category = df["AQI_category"].value_counts().reindex(CATEGORY_LABELS).fillna(0)
    hours_per_category.plot.bar(ax=ax, color="steelblue")
    ax.set(title="Hours spent in each AQI category", xlabel="", ylabel="Hours")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(out / "aqi_categories.png", dpi=120)
    plt.close(fig)