"""Construct a leakage-safe multi-station PM2.5 forecasting table.

All predictors are contemporaneous or lagged at issue time t; targets are
future PM2.5 at t+h. Rolling summaries are shifted before calculation.
"""
from pathlib import Path

import pandas as pd

PANEL = Path('delhi_pm25_hourly_panel_v2')
OUT = Path('delhi_pm25_forecasting_features.csv.gz')
SUMMARY = Path('forecasting_feature_construction_summary.csv')
TARGET = 'PM2.5 (µg/m³)'

coverage = pd.read_csv(PANEL / 'hourly_panel_coverage_by_station.csv')
core = set(coverage.loc[coverage['pm25_hour_missing_pct'] <= 10, 'station'])
frames = []
for path in sorted(PANEL.glob('*_hourly.csv.gz')):
    x = pd.read_csv(path, parse_dates=['hour_utc'])
    x = x.loc[x['station'].isin(core)].copy()
    frames.append(x)

df = pd.concat(frames, ignore_index=True).sort_values(['station', 'hour_utc'])
df = df.drop_duplicates(['station', 'hour_utc'], keep='first')
df['year'] = df['hour_utc'].dt.year
df['month'] = df['hour_utc'].dt.month
df['hour'] = df['hour_utc'].dt.hour
df['day_of_week'] = df['hour_utc'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')

# Cyclical calendar representation avoids treating 23:00 and 00:00 as far apart.
import numpy as np
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

by_station = df.groupby('station', group_keys=False)
for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
    df[f'pm25_lag_{lag}h'] = by_station[TARGET].shift(lag)

shifted = by_station[TARGET].shift(1)
for window in [6, 24, 72, 168]:
    df[f'pm25_rollmean_{window}h'] = (
        shifted.groupby(df['station']).rolling(window, min_periods=max(3, window // 2)).mean()
        .reset_index(level=0, drop=True)
    )
    df[f'pm25_rollsd_{window}h'] = (
        shifted.groupby(df['station']).rolling(window, min_periods=max(3, window // 2)).std()
        .reset_index(level=0, drop=True)
    )

for horizon in [1, 6, 24]:
    df[f'target_pm25_t_plus_{horizon}h'] = by_station[TARGET].shift(-horizon)

# Retain target, counts, all contemporaneous measurements, lags, calendar fields,
# and future targets. Modelling-time imputation/scaling is intentionally not done here.
df.to_csv(OUT, index=False, compression='gzip')
summary = pd.DataFrame({
    'horizon_hours': [1, 6, 24],
    'records_with_target': [int(df[f'target_pm25_t_plus_{h}h'].notna().sum()) for h in [1, 6, 24]],
    'core_stations': [len(core)] * 3,
    'first_issue_time_utc': [df['hour_utc'].min()] * 3,
    'last_issue_time_utc': [df['hour_utc'].max()] * 3,
})
summary.to_csv(SUMMARY, index=False)
print(summary.to_string(index=False))
