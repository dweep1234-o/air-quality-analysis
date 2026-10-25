# Air Quality Analysis: Results

## Data
The dataset contains hourly air quality records for five Indian cities
(Bangalore, Chennai, Delhi, Kolkata, Mumbai) from 2015 to 2024, totalling
438,245 records. After cleaning, all records were retained.

## Air Quality Index
Using US EPA breakpoints for PM2.5 and PM10, the overall AQI shows that
the recorded hours were spread across categories as follows:

| Category     | Share of hours |
|--------------|----------------|
| Good         | 0.23%          |
| Satisfactory | 1.61%          |
| Moderate     | 16.07%         |
| Poor         | 17.69%         |
| Very Poor    | 23.56%         |
| Severe       | 40.83%         |

## Forecasting PM2.5
A Random Forest model was trained on the first 80% of the timeline and tested
on the last 20%, using lagged values, calendar features and city.

| Model                       | MAE     | RMSE    | R²     |
|-----------------------------|---------|---------|--------|
| Random Forest               | 125.09  | 144.41  | ≈ 0.00 |
| Persistence baseline (lag-1)| 166.77  | 204.19  | -1.00  |

The model explains none of the variation in PM2.5 and performs no better
than predicting the average value.

## Predictability check
[Paste the table from check_predictability.py here]

Lag-1 and lag-24 autocorrelations are close to zero in every city. This means
a PM2.5 value gives almost no information about the next hour or the same
hour the next day.

## Limitations
- In real air quality data, hourly PM2.5 is strongly autocorrelated, usually
  around 0.9. The absence of this pattern here suggests the values may not be
  genuine monitoring readings.
- Daily data shows the same pattern, so changing the time resolution does not
  solve the problem.
- The AQI shares reflect the dataset's unusually high average (about 250 µg/m³)
  and should not be read as typical conditions in these cities.

## Conclusion
The pipeline correctly cleans the data, calculates AQI, and evaluates
forecasts. The main finding is that this dataset contains no time-dependent
pattern, so forecasting it is not possible. Future work should use verified
monitoring data, such as OpenAQ or CPCB records, to test the same pipeline.