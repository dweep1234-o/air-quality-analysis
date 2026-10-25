import numpy as np
import pandas as pd

from src.aqi import PM10_BREAKPOINTS, PM25_BREAKPOINTS, compute_aqi, sub_index


def test_pm25_breakpoint_values():
    assert np.isclose(sub_index([12.0], PM25_BREAKPOINTS)[0], 50)
    assert np.isclose(sub_index([35.4], PM25_BREAKPOINTS)[0], 100)
    assert np.isclose(sub_index([150.4], PM25_BREAKPOINTS)[0], 200)


def test_pm10_breakpoint_values():
    assert np.isclose(sub_index([154], PM10_BREAKPOINTS)[0], 100)


def test_overall_aqi_is_maximum_sub_index():
    df = pd.DataFrame({"PM2.5": [35.4], "PM10": [254]})  # PM2.5 gives 100, PM10 gives 150
    out = compute_aqi(df)
    assert out["AQI"].iloc[0] == 150
    assert out["AQI_category"].iloc[0] == "Moderate"