"""Validation-selected final models evaluated fairly on the 2025 test year."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('delhi_pm25_forecasting_features.csv.gz', parse_dates=['hour_utc'])
target_now = 'PM2.5 (µg/m³)'
target_cols = [c for c in df if c.startswith('target_pm25_')]
exclude = {'hour_utc', 'station_id', 'pm25_hour_valid'} | set(target_cols)
features = [c for c in df if c not in exclude]
categorical = ['station']
numeric = [c for c in features if c not in categorical]

def period(start, end, h):
    target_time = df['hour_utc'] + pd.Timedelta(hours=h)
    ycol = f'target_pm25_t_plus_{h}h'
    return df.loc[(df.hour_utc >= start) & (target_time <= end)].dropna(subset=[ycol]).copy()

def metrics(y, p):
    return dict(n=len(y), mae=mean_absolute_error(y,p),
                rmse=mean_squared_error(y,p)**0.5, r2=r2_score(y,p), bias=float(np.mean(p-y)))

rows=[]
for h, selected in [(1, 'persistence'), (6, 'hist_gradient_boosting'), (24, 'persistence')]:
    ycol=f'target_pm25_t_plus_{h}h'
    test=period('2025-01-01 00:00:00+00:00','2025-12-31 23:00:00+00:00',h)
    common=test.dropna(subset=[target_now]).copy()
    y=common[ycol]
    p_persist=common[target_now]
    base=metrics(y,p_persist)
    if selected == 'persistence':
        chosen=base
    else:
        train=period('2024-01-01 00:00:00+00:00','2024-09-30 23:00:00+00:00',h)
        pre=ColumnTransformer([
          ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
          ('station',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
        pipe=Pipeline([('preprocess',pre),('model',HistGradientBoostingRegressor(
          learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.,max_iter=250,random_state=2026))])
        pipe.fit(train[features],train[ycol])
        chosen=metrics(y,pipe.predict(common[features]))
    rows.append({
      'horizon_hours':h,'validation_selected_model':selected,
      'test_records_common_support':chosen['n'],
      'selected_mae':chosen['mae'],'selected_rmse':chosen['rmse'],
      'selected_r2':chosen['r2'],'selected_bias':chosen['bias'],
      'persistence_mae':base['mae'],'mae_improvement_vs_persistence':base['mae']-chosen['mae'],
      'mae_improvement_pct':100*(base['mae']-chosen['mae'])/base['mae']})
out=pd.DataFrame(rows)
out.to_csv('final_2025_evaluation_validation_selected.csv',index=False)
print(out.round(4).to_string(index=False))
