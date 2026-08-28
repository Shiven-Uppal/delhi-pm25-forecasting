"""Publication figures for the final PM10-free PM2.5 forecasting study."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path('final_paper_figures')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

# Figure 2: final model trace for two representative core stations.
H = 6
df = pd.read_csv('delhi_pm25_forecasting_features.csv.gz', parse_dates=['hour_utc'])
ycol = f'target_pm25_t_plus_{H}h'
targets = [c for c in df if c.startswith('target_pm25_')]
exclude = {'hour_utc', 'station_id', 'pm25_hour_valid'} | set(targets)
features = [c for c in df if c not in exclude and 'PM10' not in c]
numeric = [c for c in features if c != 'station']
def select(start, end):
    target_time = df.hour_utc + pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc >= start) & (target_time <= end)].dropna(subset=[ycol]).copy()
train = select('2024-01-01 00:00:00+00:00', '2024-09-30 23:00:00+00:00')
test = select('2025-01-01 00:00:00+00:00', '2025-12-31 23:00:00+00:00').dropna(subset=['PM2.5 (µg/m³)'])
pre = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
    ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['station']),
])
model = Pipeline([('preprocess', pre), ('model', HistGradientBoostingRegressor(
    learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., max_iter=250, random_state=2026))])
model.fit(train[features], train[ycol])
test['prediction'] = model.predict(test[features])
test['target_time'] = test.hour_utc + pd.Timedelta(hours=H)

stations = ['Alipur, Delhi - DPCC', 'Wazirpur, Delhi - DPCC']
window_start = pd.Timestamp('2025-12-01 00:00:00+00:00')
window_end = pd.Timestamp('2025-12-08 00:00:00+00:00')
fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
for ax, station in zip(axes, stations):
    x = test.loc[(test.station == station) & (test.target_time >= window_start) & (test.target_time < window_end)]
    ax.plot(x.target_time, x[ycol], color='#1f2937', lw=1.3, label='Observed PM₂.₅')
    ax.plot(x.target_time, x.prediction, color='#2563eb', lw=1.1, label='PM₁₀-free HGB')
    ax.plot(x.target_time, x['PM2.5 (µg/m³)'], color='#dc2626', lw=0.9, alpha=.85, label='Persistence')
    ax.set_ylabel('µg/m³')
    ax.set_title(station.replace(', Delhi - ', ' — '), loc='left', fontsize=10, weight='bold')
    ax.grid(axis='y', alpha=.25)
axes[0].legend(ncol=3, fontsize=8, frameon=False, loc='upper right')
axes[-1].set_xlabel('Forecast target time (UTC), 1–7 December 2025')
fig.tight_layout()
fig.savefig(OUT / 'figure2_six_hour_forecast_traces.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 3: station-heldout gains.
held = pd.read_csv('final_pm10_free_station_heldout_six_hour_results.csv')
held = held.loc[held.fold != 'pooled_station_heldout'].copy()
held['fold_label'] = [f'Fold {i}' for i in range(1, len(held)+1)]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bars = ax.barh(held.fold_label, held.mae_improvement, color='#2563eb')
ax.axvline(0, color='black', lw=.8)
ax.set_xlabel('MAE improvement over persistence (µg/m³)')
ax.set_title('Six-hour performance at entirely unseen stations', loc='left', weight='bold')
ax.grid(axis='x', alpha=.25)
for bar, value in zip(bars, held.mae_improvement):
    ax.text(value + .12, bar.get_y() + bar.get_height()/2, f'{value:.2f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'figure3_station_heldout_improvement.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Figure 4: grouped permutation importance.
imp = pd.read_csv('final_pm10_free_grouped_permutation_importance.csv').sort_values('mean_mae_increase')
labels = {
    'PM2.5 rolling summaries': 'PM₂.₅ rolling summaries',
    'current PM2.5': 'Current PM₂.₅',
    'PM2.5 lag features': 'PM₂.₅ lag features',
}
imp['display_group'] = imp['predictor_group'].replace(labels)
fig, ax = plt.subplots(figsize=(7, 4.1))
err_low = imp.mean_mae_increase - imp.ci_95_low
err_high = imp.ci_95_high - imp.mean_mae_increase
ax.barh(imp.display_group, imp.mean_mae_increase, xerr=np.vstack([err_low, err_high]), color='#0f766e', capsize=3)
ax.set_xlabel('Increase in 2025 MAE after within-station permutation (µg/m³)')
ax.set_title('Model reliance of the final six-hour model', loc='left', weight='bold')
ax.grid(axis='x', alpha=.25)
fig.tight_layout()
fig.savefig(OUT / 'figure4_grouped_permutation_importance.png', dpi=300, bbox_inches='tight')
plt.close(fig)
