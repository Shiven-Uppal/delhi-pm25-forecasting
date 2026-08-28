"""Time-safe benchmark suite for 1, 6 and 24-hour PM2.5 forecasts."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = 'delhi_pm25_forecasting_features.csv.gz'
OUT = 'fair_benchmark_results_2025.csv'
df = pd.read_csv(DATA, parse_dates=['hour_utc'])

target_now = 'PM2.5 (µg/m³)'
target_cols = [c for c in df if c.startswith('target_pm25_')]
exclude = {'hour_utc', 'station_id', 'pm25_hour_valid'} | set(target_cols)
features = [c for c in df.columns if c not in exclude]
categorical = ['station']
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
    ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
])

def subset_within(start, end, horizon):
    # both the issue time and the target time must remain inside the named period
    target_time = df['hour_utc'] + pd.Timedelta(hours=horizon)
    return df.loc[(df['hour_utc'] >= start) & (target_time <= end)].copy()

periods = {
    'train': ('2024-01-01 00:00:00+00:00', '2024-09-30 23:00:00+00:00'),
    'validation': ('2024-10-01 00:00:00+00:00', '2024-11-30 23:00:00+00:00'),
    'test_2025': ('2025-01-01 00:00:00+00:00', '2025-12-31 23:00:00+00:00'),
}

def score(y, pred):
    return {
        'n': len(y),
        'mae': mean_absolute_error(y, pred),
        'rmse': mean_squared_error(y, pred) ** 0.5,
        'r2': r2_score(y, pred),
        'bias': float(np.mean(pred - y)),
    }

all_results = []
for horizon in [1, 6, 24]:
    ycol = f'target_pm25_t_plus_{horizon}h'
    data = {name: subset_within(start, end, horizon).dropna(subset=[ycol])
            for name, (start, end) in periods.items()}
    train, val, test = data['train'], data['validation'], data['test_2025']

    # Naive forecasts require a currently observed PM2.5 value at issue time.
    for model_name, predictor in [
        ('persistence', target_now),
        ('seasonal_persistence_7d', 'pm25_lag_168h'),
    ]:
        for split_name, part in [('validation', val), ('test_2025', test)]:
            valid = part.dropna(subset=[predictor])
            row = score(valid[ycol], valid[predictor])
            row.update({'horizon_hours': horizon, 'model': model_name, 'split': split_name})
            all_results.append(row)

    # Linear and nonlinear models have all preprocessing fitted on train only.
    for model_name, estimator in [
        ('ridge', Ridge(alpha=10.0)),
        ('hist_gradient_boosting', HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            max_iter=250, random_state=2026)),
    ]:
        pipe = Pipeline([('preprocess', preprocessor), ('model', estimator)])
        training = train.dropna(subset=[ycol])
        pipe.fit(training[features], training[ycol])
        for split_name, part in [('validation', val), ('test_2025', test)]:
            pred = pipe.predict(part[features])
            row = score(part[ycol], pred)
            row.update({'horizon_hours': horizon, 'model': model_name, 'split': split_name})
            all_results.append(row)

results = pd.DataFrame(all_results)
results.to_csv(OUT, index=False)
print(results.round(4).to_string(index=False))
