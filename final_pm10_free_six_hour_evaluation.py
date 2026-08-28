"""Final pre-specified PM10-free 6-hour model: test metrics and inference."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG=np.random.default_rng(20260828)
H=6
df=pd.read_csv('delhi_pm25_forecasting_features.csv.gz',parse_dates=['hour_utc'])
ycol=f'target_pm25_t_plus_{H}h'
targets=[c for c in df if c.startswith('target_pm25_')]
exclude={'hour_utc','station_id','pm25_hour_valid'}|set(targets)
features=[c for c in df if c not in exclude and 'PM10' not in c]
numeric=[c for c in features if c!='station']

def select(start,end):
    tt=df.hour_utc+pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc>=start)&(tt<=end)].dropna(subset=[ycol]).copy()

train=select('2024-01-01 00:00:00+00:00','2024-09-30 23:00:00+00:00')
test=select('2025-01-01 00:00:00+00:00','2025-12-31 23:00:00+00:00').dropna(subset=['PM2.5 (µg/m³)'])
pre=ColumnTransformer([
 ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
 ('station',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['station'])])
model=Pipeline([('preprocess',pre),('model',HistGradientBoostingRegressor(
 learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.,max_iter=250,random_state=2026))])
model.fit(train[features],train[ycol])
pred=model.predict(test[features]); persist=test['PM2.5 (µg/m³)'].to_numpy(); y=test[ycol].to_numpy()
lossdiff=np.abs(persist-y)-np.abs(pred-y)

# Test metrics on the identical persistence-supported station-hour records.
metrics=pd.DataFrame([{
 'horizon_hours':H,'final_model':'PM10-free histogram gradient boosting','test_records':len(test),
 'mae':mean_absolute_error(y,pred),'rmse':mean_squared_error(y,pred)**.5,'r2':r2_score(y,pred),
 'bias':float(np.mean(pred-y)),'persistence_mae':mean_absolute_error(y,persist),
 'mae_improvement':mean_absolute_error(y,persist)-mean_absolute_error(y,pred),
 'mae_improvement_pct':100*(mean_absolute_error(y,persist)-mean_absolute_error(y,pred))/mean_absolute_error(y,persist)}])
metrics.to_csv('final_pm10_free_six_hour_test_metrics.csv',index=False)

# Hour-level dependence-aware inference.
d=pd.DataFrame({'hour':test.hour_utc,'d':lossdiff}).groupby('hour',sort=True).d.mean().to_numpy()
T=len(d); mean=float(d.mean()); block=168; starts=np.arange(T-block+1); boot=[]
for _ in range(2000):
 chunks=[]; n=0
 while n<T:
  c=d[RNG.choice(starts):][:block]; chunks.append(c); n+=len(c)
 boot.append(np.concatenate(chunks)[:T].mean())
z=d-mean; lag=24; lrv=np.mean(z*z)
for k in range(1,lag+1): lrv+=2*(1-k/(lag+1))*np.mean(z[k:]*z[:-k])
dm=mean/np.sqrt(lrv/T)
inference=pd.DataFrame([{
 'horizon_hours':H,'hour_level_observations':T,'mean_hour_level_mae_improvement':mean,
 'bootstrap_block_hours':block,'bootstrap_replicates':2000,
 'bootstrap_ci_95_low':np.quantile(boot,.025),'bootstrap_ci_95_high':np.quantile(boot,.975),
 'dm_hac_lag_hours':lag,'dm_statistic':dm,'dm_two_sided_p_value':2*norm.sf(abs(dm))}])
inference.to_csv('final_pm10_free_six_hour_inference.csv',index=False)
print(metrics.round(5).to_string(index=False)); print(inference.round(6).to_string(index=False))
