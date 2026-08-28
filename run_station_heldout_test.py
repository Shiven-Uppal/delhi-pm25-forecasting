"""Five-fold station-heldout transfer test for the frozen 6-hour HGB model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

H = 6
df = pd.read_csv('delhi_pm25_forecasting_features.csv.gz', parse_dates=['hour_utc'])
ycol = f'target_pm25_t_plus_{H}h'
current = 'PM2.5 (µg/m³)'
targets = [c for c in df if c.startswith('target_pm25_')]
# Station identity is deliberately omitted: a held-out station has no learned station effect.
exclude = {'hour_utc', 'station', 'station_id', 'pm25_hour_valid'} | set(targets)
features = [c for c in df if c not in exclude and 'PM10' not in c]

def select(start, end):
    target_time = df.hour_utc + pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc >= start) & (target_time <= end)].dropna(subset=[ycol]).copy()

train_all = select('2024-01-01 00:00:00+00:00', '2024-09-30 23:00:00+00:00')
test_all = select('2025-01-01 00:00:00+00:00', '2025-12-31 23:00:00+00:00').dropna(subset=[current])
stations = np.array(sorted(train_all.station.unique()))
splitter = GroupKFold(n_splits=5)
rows=[]
pooled=[]
for fold, (train_idx, held_idx) in enumerate(splitter.split(stations, groups=stations), start=1):
    train_stations = set(stations[train_idx])
    held_stations = set(stations[held_idx])
    train = train_all.loc[train_all.station.isin(train_stations)]
    test = test_all.loc[test_all.station.isin(held_stations)]
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', HistGradientBoostingRegressor(learning_rate=.05, max_leaf_nodes=31,
         l2_regularization=1., max_iter=250, random_state=2026)),
    ])
    model.fit(train[features], train[ycol])
    prediction = model.predict(test[features])
    persistence = test[current].to_numpy()
    y = test[ycol].to_numpy()
    rows.append({
      'fold': fold,
      'heldout_stations': ' | '.join(sorted(held_stations)),
      'heldout_station_count': len(held_stations),
      'test_records': len(test),
      'hgb_mae': mean_absolute_error(y,prediction),
      'persistence_mae': mean_absolute_error(y,persistence),
      'mae_improvement': mean_absolute_error(y,persistence)-mean_absolute_error(y,prediction),
      'hgb_rmse': mean_squared_error(y,prediction)**.5,
      'hgb_r2': r2_score(y,prediction),
    })
    pooled.append(pd.DataFrame({'y':y,'hgb':prediction,'persistence':persistence}))

out=pd.DataFrame(rows)
pool=pd.concat(pooled,ignore_index=True)
overall=pd.DataFrame([{
  'fold':'pooled_station_heldout', 'heldout_stations':'All 31 stations across five folds',
  'heldout_station_count':31,'test_records':len(pool),
  'hgb_mae':mean_absolute_error(pool.y,pool.hgb),
  'persistence_mae':mean_absolute_error(pool.y,pool.persistence),
  'mae_improvement':mean_absolute_error(pool.y,pool.persistence)-mean_absolute_error(pool.y,pool.hgb),
  'hgb_rmse':mean_squared_error(pool.y,pool.hgb)**.5,'hgb_r2':r2_score(pool.y,pool.hgb)}])
pd.concat([out,overall],ignore_index=True).to_csv('final_pm10_free_station_heldout_six_hour_results.csv',index=False)
print(pd.concat([out,overall],ignore_index=True).round(4).to_string(index=False))
