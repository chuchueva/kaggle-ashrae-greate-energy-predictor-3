import gc
import pandas as pd
import numpy as np
import math as math
import constants as c
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def get_error(v1, v2):

    return sqrt(mean_squared_error(v1, v2))


def prepare_data_glb(x, building_data, weather_data, site=None, meter=None):

    df = x.merge(building_data, on='building_id', how='left')
    df = df.merge(weather_data, on=['site_id', 'timestamp'], how='left')

    df.index = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M:%S')

    drop_features = ['timestamp']
    df.drop(drop_features, axis=1, inplace=True)

    if 'meter_reading' in df:
        df['meter_reading'] = np.log1p(df['meter_reading'])

    return df


def feature_engineering(df):

    df['weekday'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['season'] = np.zeros(len(df))
    df.loc[df.index.month.isin([3, 4, 5]), 'season'] = 1
    df.loc[df.index.month.isin([6, 7, 8]), 'season'] = 2
    df.loc[df.index.month.isin([9, 10, 11]), 'season'] = 3

    settings = [1, 5]

    for w in settings:
        df['sin_' + str(w)] = np.square(np.sin(df.index.dayofyear.astype('float64') / 365 * w * math.pi).values)

    df['primary_use'] = df['primary_use'].map(c.PRIMARY_USE_MAP).astype(np.uint8)

    print('Weekday-hour-sins is done!')

    return df


def create_lags(df):

    feature_cols = ['air_temperature', 'dew_temperature', 'wind_speed']
    settings = [24, 168]

    for site_id in range(16):

        mask = df['site_id'] == site_id

        for feature in feature_cols:
            col_names_lags = [feature + '_lag_' + str(shift) for shift in settings]

            for idx in range(0, len(settings)):
                df.loc[mask, col_names_lags[idx]] = df.loc[mask, feature].shift(settings[idx])

    return df


def create_window_averages(df, window):

    feature_cols = ['air_temperature', 'dew_temperature', 'wind_speed']
    df_site = df.groupby('site_id')

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)
    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f'{feature}_mean_window_{window}'] = df_mean[feature]
        df[f'{feature}_median_window_{window}'] = df_median[feature]

    return df


def get_seasonality_model(df, model=None):

    settings = {
        'col_name_x':   ['sin_1', 'air_temperature'],
        'col_name_y':   ['meter_reading'],
        'months_bundle': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        'days_bundle':  [[0, 1, 2, 3, 4, 5, 6]],
        'hours_bundle': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
        'model': dict()
    }

    sins = [1]
    for w in sins:
        df.loc[:, 'sin_' + str(w)] = np.square(np.sin(df.index.dayofyear.astype('float64') / 365 * w * math.pi).values)

    y_pred = np.zeros(len(df))
    x_train = df[settings['col_name_x']]

    if model is None:
        y_train = df[settings['col_name_y']]

    for m in range(len(settings['months_bundle'])):
        for d in range(len(settings['days_bundle'])):
            for h in range(len(settings['hours_bundle'])):

                mask = (np.isin(x_train.index.month,  settings['months_bundle'][m])) & \
                       (np.isin(x_train.index.dayofweek,  settings['days_bundle'][d])) & \
                       (np.isin(x_train.index.hour, settings['hours_bundle'][h]))

                if any(mask):

                    if model is None:
                        model = LinearRegression()
                        model.fit(x_train[mask].values, y_train[mask].values)

                    y_m = model.predict(x_train[mask].values)
                    y_pred[mask] = y_m[:, 0]

    return model, y_pred