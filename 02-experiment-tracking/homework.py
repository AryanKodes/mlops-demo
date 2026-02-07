import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import mlflow
import pickle

df = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df['duration'] = df.duration.dt.total_seconds() / 60

df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

categorical = ['PULocationID', 'DOLocationID']

df[categorical] = df[categorical].astype(str)

train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].astype('str')
    
    return df

df_val = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet')

val_dicts = df_val[categorical].to_dict(orient='records')

X_val = dv.transform(val_dicts) 
y_val = df_val.duration.values

y_pred = lr.predict(X_val)

with open('mlruns/models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)

for x in range(15):
        with mlflow.start_run():
                        alpha = x/100

                        mlflow.set_tag("developer", "aryan")
                        mlflow.log_param("alpha", alpha)

                        lr = Lasso(alpha)
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_val)
                        
                        rmse = mean_squared_error(y_val, y_pred, squared=False)
                        mlflow.log_metric("rmse", rmse)
                        print("alpha : ", x)
        