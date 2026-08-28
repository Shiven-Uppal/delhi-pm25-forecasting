"""Grouped, within-station permutation importance for frozen 6-hour HGB."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(20260828)
H = 6
REPEATS = 20
df = pd.read_csv('delhi_pm25_forecasting_features.csv.gz', parse_dates=['hour_utc'])
ycol=f'target_pm25_t_plus_{H}h'
target_cols=[c for c in df if c.startswith('target_pm25_')]
exclude={'hour_utc','station_id','pm25_hour_valid'}|set(target_cols)
features=[c for c in df if c not in exclude and 'PM10' not in c]

def select(start,end):
    target_time=df.hour_utc+pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc>=start)&(target_time<=end)].dropna(subset=[ycol]).copy()

train=select('2024-01-01 00:00:00+00:00','2024-09-30 23:00:00+00:00')
test=select('2025-01-01 00:00:00+00:00','2025-12-31 23:00:00+00:00').dropna(subset=['PM2.5 (µg/m³)'])

# All-missing columns cannot contribute and are explicitly removed before fitting.
features=[c for c in features if c == 'station' or train[c].notna().any()]
numeric=[c for c in features if c != 'station']
preprocessor=ColumnTransformer([
    ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
    ('station',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['station']),
])
model=Pipeline([('preprocess',preprocessor),('model',HistGradientBoostingRegressor(
    learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.,max_iter=250,random_state=2026))])
model.fit(train[features],train[ycol])
baseline=mean_absolute_error(test[ycol],model.predict(test[features]))

groups={
 'current PM2.5':['PM2.5 (µg/m³)'],
 'PM2.5 lag features':[c for c in features if c.startswith('pm25_lag_')],
 'PM2.5 rolling summaries':[c for c in features if c.startswith('pm25_roll')],
 'gaseous pollutants':[c for c in features if c in ['NO (µg/m³)','NO2 (µg/m³)','NOx (ppb)','NH3 (µg/m³)','SO2 (µg/m³)','CO (mg/m³)','Ozone (µg/m³)','Benzene (µg/m³)','Toluene (µg/m³)','Xylene (µg/m³)','Eth-Benzene (µg/m³)','MP-Xylene (µg/m³)']],
 'meteorology':[c for c in features if c in ['AT (°C)','RH (%)','WS (m/s)','WD (deg)','RF (mm)','TOT-RF (mm)','SR (W/mt2)','BP (mmHg)','VWS (m/s)']],
 'calendar':[c for c in features if c in ['year','month','hour','day_of_week','is_weekend','hour_sin','hour_cos','month_sin','month_cos']],
 'availability counts':[c for c in features if c.endswith('__n15')],
}
rows=[]
station_indices=[x.index.to_numpy() for _,x in test.groupby('station',sort=False)]
for group, cols in groups.items():
    cols=[c for c in cols if c in features]
    if not cols:
        continue
    increases=[]
    for _ in range(REPEATS):
        perm=test[features].copy()
        for idx in station_indices:
            shuffled=RNG.permutation(idx)
            perm.loc[idx,cols]=test.loc[shuffled,cols].to_numpy()
        increases.append(mean_absolute_error(test[ycol],model.predict(perm))-baseline)
    rows.append({'predictor_group':group,'n_features':len(cols),'baseline_mae':baseline,
      'mean_mae_increase':np.mean(increases),'ci_95_low':np.quantile(increases,.025),
      'ci_95_high':np.quantile(increases,.975),'permutations':REPEATS})
out=pd.DataFrame(rows).sort_values('mean_mae_increase',ascending=False)
out.to_csv('final_pm10_free_grouped_permutation_importance.csv',index=False)
print(out.round(4).to_string(index=False))
