from pathlib import Path

import pandas as pd

raw = Path('audit_work_20260828/delhi_pm25_multiyear_raw/raw')
records = []
for path in sorted(raw.glob('*15_minute*.csv')):
    df = pd.read_csv(path, usecols=['Station Name', 'Timestamp', 'PM2.5 (µg/m³)'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
    df['PM2.5 (µg/m³)'] = pd.to_numeric(df['PM2.5 (µg/m³)'], errors='coerce')
    station = df['Station Name'].dropna().iloc[0] if df['Station Name'].notna().any() else path.stem
    valid_time = df['Timestamp'].notna()
    duplicate = df.loc[valid_time].duplicated(['Station Name', 'Timestamp']).sum()
    records.append({
        'station': station,
        'file': path.name,
        'records_15_min': len(df),
        'first_timestamp_utc': df.loc[valid_time, 'Timestamp'].min(),
        'last_timestamp_utc': df.loc[valid_time, 'Timestamp'].max(),
        'valid_pm25_15_min': int(df['PM2.5 (µg/m³)'].notna().sum()),
        'pm25_missing_pct': 100 * df['PM2.5 (µg/m³)'].isna().mean(),
        'duplicate_station_timestamps': int(duplicate),
    })

out = pd.DataFrame(records).sort_values('station')
out.to_csv('pm25_2024_25_coverage_audit.csv', index=False)
print(out.to_string(index=False))
print(f'\nStations: {len(out)}')
print(f'Rows: {out.records_15_min.sum():,}')
print(f'Valid PM2.5 readings: {out.valid_pm25_15_min.sum():,}')
