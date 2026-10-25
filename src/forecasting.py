"""Short-term PM2.5 forecasting with a persistence baseline for comparison."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import WEATHER

LAGS = (1, 2, 3, 6, 12, 24)


def build_features(df, target="PM2.5"):
    """Create lagged and calendar features without using future values."""
    frames = []
    for _, group in df.groupby("station"):
        g = group.sort_values("datetime").copy()
        g["site"] = g["station"]  # keep the city name before one-hot encoding
        for lag in LAGS:
            g[f"lag_{lag}"] = g[target].shift(lag)
        g["hour"] = g["datetime"].dt.hour
        g["month"] = g["datetime"].dt.month
        g["dayofweek"] = g["datetime"].dt.dayofweek
        frames.append(g)
    feat = pd.concat(frames, ignore_index=True)
    feat = pd.get_dummies(feat, columns=["station"], dtype=int)
    return feat.dropna(subset=[f"lag_{lag}" for lag in LAGS]).reset_index(drop=True)


def feature_columns(feat):
    lag_cols = [c for c in feat.columns if c.startswith("lag_")]
    station_cols = [c for c in feat.columns if c.startswith("station_")]
    return lag_cols + station_cols + ["hour", "month", "dayofweek"] + WEATHER


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def run_forecast(df, target="PM2.5", test_fraction=0.2, random_state=42):
    """Train on the earlier 80% of the timeline and test on the latest 20%."""
    feat = build_features(df, target)

    times = feat["datetime"].sort_values().reset_index(drop=True)
    cutoff = times.iloc[int(len(times) * (1 - test_fraction))]
    train = feat[feat["datetime"] < cutoff]
    test = feat[feat["datetime"] >= cutoff]

    cols = feature_columns(feat)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(train[cols], train[target])
    predictions = model.predict(test[cols])

    metrics = pd.DataFrame([
        {"model": "Random Forest", **evaluate(test[target], predictions)},
        {"model": "Persistence baseline (lag-1)", **evaluate(test[target], test["lag_1"])},
    ]).round(3)

    results = test[["datetime", "site", target]].rename(columns={"site": "station"})
    results = results.assign(predicted=predictions)
    importances = (
        pd.Series(model.feature_importances_, index=cols)
        .sort_values(ascending=False)
        .round(4)
    )
    return metrics, results, importances