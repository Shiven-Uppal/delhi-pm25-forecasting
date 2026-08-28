"""Create an auditable hourly panel from the raw 15-minute Delhi files.

PM2.5 is retained as an hourly target only when at least 3 of 4 quarter-hour
readings are present. Other variables are hourly means; their observation
counts are retained so later modelling can apply training-only imputation.
"""
from pathlib import Path

import pandas as pd

RAW = Path('audit_work_20260828/delhi_pm25_multiyear_raw/raw')
OUT = Path('delhi_pm25_hourly_panel_v2')
OUT.mkdir(exist_ok=True)

target = 'PM2.5 (µg/m³)'
base_columns = ['Station ID', 'State', 'City', 'Station Name', 'Timestamp']
files = sorted(RAW.glob('*15_minute*.csv'))
station_summaries = []

for path in files:
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
    df = df.loc[df['Timestamp'].notna()].copy()
    station = df['Station Name'].dropna().iloc[0]
    df['hour_utc'] = df['Timestamp'].dt.floor('h')

    numeric = [c for c in df.columns if c not in base_columns + ['hour_utc']]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    grouped = df.groupby('hour_utc', sort=True)
    hourly = grouped[numeric].mean()
    counts = grouped[numeric].count().add_suffix('__n15')
    hourly = hourly.join(counts)
    hourly.insert(0, 'station', station)
    hourly.insert(1, 'station_id', df['Station ID'].dropna().iloc[0])
    hourly = hourly.reset_index()
    hourly['pm25_hour_valid'] = hourly[f'{target}__n15'] >= 3
    hourly.loc[~hourly['pm25_hour_valid'], target] = pd.NA

    safe_name = path.stem.replace('_15_minute_AQI_Data_for_2024-25', '')
    final_path = OUT / f'{safe_name}_hourly.csv.gz'
    temporary_path = OUT / f'{safe_name}_hourly.csv.gz.part'
    hourly.to_csv(temporary_path, index=False, compression='gzip')
    temporary_path.replace(final_path)
    station_summaries.append({
        'station': station,
        'station_id': df['Station ID'].dropna().iloc[0],
        'hourly_rows': len(hourly),
        'valid_hourly_pm25': int(hourly['pm25_hour_valid'].sum()),
        'pm25_hour_missing_pct': 100 * (1 - hourly['pm25_hour_valid'].mean()),
        'first_hour_utc': hourly['hour_utc'].min(),
        'last_hour_utc': hourly['hour_utc'].max(),
    })

summary = pd.DataFrame(station_summaries).sort_values('station')
summary.to_csv(OUT / 'hourly_panel_coverage_by_station.csv', index=False)
print(summary.to_string(index=False))
print(f'\nTotal valid hourly PM2.5 targets: {summary.valid_hourly_pm25.sum():,}')
