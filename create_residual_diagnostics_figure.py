"""Residual diagnostics for the locked PM10-free six-hour HGB model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path('final_paper_figures/figure5_six_hour_residual_diagnostics.png')
OUT.parent.mkdir(exist_ok=True)
RNG_SEED = 20260828
H = 6

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

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

preprocess = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
    ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['station']),
])
model = Pipeline([('preprocess', preprocess), ('model', HistGradientBoostingRegressor(
    learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., max_iter=250, random_state=2026))])
model.fit(train[features], train[ycol])

sample = test.sample(n=10_000, random_state=RNG_SEED).copy()
sample['prediction'] = model.predict(sample[features])
sample['observed'] = sample[ycol]
sample['residual'] = sample['observed'] - sample['prediction']

limit = float(np.ceil(max(sample.observed.max(), sample.prediction.max()) / 50) * 50)
pred_limit = float(np.ceil(sample.prediction.max() / 50) * 50)
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

ax = axes[0]
ax.scatter(sample.observed, sample.prediction, s=6, alpha=.16, linewidths=0, color='#2563eb', rasterized=True)
ax.plot([0, limit], [0, limit], '--', color='#111827', lw=1.1, label='Perfect agreement')
ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='Observed PM₂.₅ (µg/m³)', ylabel='Predicted PM₂.₅ (µg/m³)')
ax.set_title('a  Predicted versus observed', loc='left', weight='bold')
ax.legend(frameon=False, loc='upper left', fontsize=8.5)
ax.grid(alpha=.18)

ax = axes[1]
ax.scatter(sample.prediction, sample.residual, s=6, alpha=.16, linewidths=0, color='#0f766e', rasterized=True)
ax.axhline(0, color='#111827', lw=1.1)
# Binned residual mean shows any systematic departure without claiming a fitted physical relationship.
bins = np.linspace(0, pred_limit, 31)
group = sample.groupby(pd.cut(sample.prediction, bins, include_lowest=True), observed=True).residual.agg(['mean', 'count'])
centres = np.array([(i.left + i.right) / 2 for i in group.index])
valid = group['count'].to_numpy() >= 20
ax.plot(centres[valid], group['mean'].to_numpy()[valid], color='#b91c1c', lw=1.8, label='Binned mean residual')
ax.set(xlim=(0, pred_limit), xlabel='Predicted PM₂.₅ (µg/m³)', ylabel='Residual: observed − predicted (µg/m³)')
ax.set_title('b  Residuals against prediction', loc='left', weight='bold')
ax.legend(frameon=False, loc='upper right', fontsize=8.5)
ax.grid(alpha=.18)

fig.savefig(OUT, dpi=320, facecolor='white', bbox_inches='tight')
print(f'Saved {OUT}; sample size={len(sample)}, random seed={RNG_SEED}')
