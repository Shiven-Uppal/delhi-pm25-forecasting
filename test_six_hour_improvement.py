"""Blocked-bootstrap CI and HAC Diebold--Mariano test for 6-hour forecasts."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(20260828)
H = 6
df = pd.read_csv('delhi_pm25_forecasting_features.csv.gz', parse_dates=['hour_utc'])
ycol = f'target_pm25_t_plus_{H}h'
target_now = 'PM2.5 (µg/m³)'
target_cols = [c for c in df if c.startswith('target_pm25_')]
exclude = {'hour_utc', 'station_id', 'pm25_hour_valid'} | set(target_cols)
features = [c for c in df if c not in exclude]
categorical = ['station']
numeric = [c for c in features if c not in categorical]

def select(start, end):
    target_time = df.hour_utc + pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc >= start) & (target_time <= end)].dropna(subset=[ycol]).copy()

train = select('2024-01-01 00:00:00+00:00', '2024-09-30 23:00:00+00:00')
test = select('2025-01-01 00:00:00+00:00', '2025-12-31 23:00:00+00:00').dropna(subset=[target_now])

pre = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
    ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
])
model = Pipeline([('preprocess', pre), ('model', HistGradientBoostingRegressor(
    learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., max_iter=250, random_state=2026))])
model.fit(train[features], train[ycol])
test['hgb_prediction'] = model.predict(test[features])
test['loss_difference'] = np.abs(test[target_now] - test[ycol]) - np.abs(test.hgb_prediction - test[ycol])

# One value per issue hour: avoids treating station forecasts within the same hour as independent.
d = test.groupby('hour_utc', sort=True)['loss_difference'].mean().to_numpy()
T = len(d)
mean_improvement = float(d.mean())

# Moving-block bootstrap: seven-day blocks preserve short-term temporal dependence.
block = 24 * 7
starts = np.arange(0, T - block + 1)
boot = np.empty(2000)
for b in range(len(boot)):
    chunks = []
    size = 0
    while size < T:
        x = d[RNG.choice(starts):][:block]
        chunks.append(x)
        size += len(x)
    boot[b] = np.concatenate(chunks)[:T].mean()
ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

# Diebold--Mariano statistic with Bartlett/Newey--West HAC covariance (24-hour lag).
lag = 24
centered = d - mean_improvement
gamma0 = np.mean(centered * centered)
lrv = gamma0
for k in range(1, lag + 1):
    gamma = np.mean(centered[k:] * centered[:-k])
    lrv += 2 * (1 - k / (lag + 1)) * gamma
se = np.sqrt(lrv / T)
dm_stat = mean_improvement / se
p_value = 2 * norm.sf(abs(dm_stat))

out = pd.DataFrame([{
    'horizon_hours': H,
    'common_station_hour_forecasts': len(test),
    'hour_level_observations': T,
    'mean_mae_improvement_ug_m3': mean_improvement,
    'bootstrap_block_hours': block,
    'bootstrap_replicates': len(boot),
    'bootstrap_ci_95_low': ci_low,
    'bootstrap_ci_95_high': ci_high,
    'dm_hac_lag_hours': lag,
    'dm_statistic': dm_stat,
    'dm_two_sided_p_value': p_value,
}])
out.to_csv('six_hour_inference_vs_persistence.csv', index=False)
print(out.round(6).to_string(index=False))
