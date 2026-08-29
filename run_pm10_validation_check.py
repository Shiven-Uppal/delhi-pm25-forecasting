"""Validation-only PM10 ablation for pre-specifying the final six-hour model."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

H=6
df=pd.read_csv('delhi_pm25_forecasting_features.csv.gz',parse_dates=['hour_utc'])
ycol=f'target_pm25_t_plus_{H}h'
targets=[c for c in df if c.startswith('target_pm25_')]
exclude={'hour_utc','station_id','pm25_hour_valid'}|set(targets)
all_features=[c for c in df if c not in exclude]

def select(start,end):
    tt=df.hour_utc+pd.Timedelta(hours=H)
    return df.loc[(df.hour_utc>=start)&(tt<=end)].dropna(subset=[ycol]).copy()

train=select('2024-01-01 00:00:00+00:00','2024-09-30 23:00:00+00:00')
validation=select('2024-10-01 00:00:00+00:00','2024-11-30 23:00:00+00:00')

def score(include_pm10):
    features=[c for c in all_features if include_pm10 or 'PM10' not in c]
    numeric=[c for c in features if c!='station']
    pre=ColumnTransformer([
      ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
      ('station',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['station'])])
    model=Pipeline([('preprocess',pre),('model',HistGradientBoostingRegressor(
      learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.,max_iter=250,random_state=2026))])
    model.fit(train[features],train[ycol])
    p=model.predict(validation[features])
    return {'specification':'with PM10' if include_pm10 else 'without PM10',
      'validation_records':len(validation),'mae':mean_absolute_error(validation[ycol],p),
      'rmse':mean_squared_error(validation[ycol],p)**.5,'r2':r2_score(validation[ycol],p),
      'bias':float(np.mean(p-validation[ycol]))}

out=pd.DataFrame([score(True),score(False)])
out['mae_change_vs_with_pm10']=out.mae-out.loc[0,'mae']
out.to_csv('six_hour_pm10_validation_ablation.csv',index=False)
print(out.round(5).to_string(index=False))
