import gc
import pandas as pd
import numpy as np
import math as math
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


def reduce_mem_usage(df, use_float16=False):

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def get_error(v):

    v.dropna(inplace=True)
    vv = v.values
    rmse = sqrt(mean_squared_log_error(vv[:, 0], vv[:, 1]))

    return rmse


def prepare_data_glb(x, building_data, weather_data):

    df = x.merge(building_data, on='building_id', how='left')
    df = df.merge(weather_data, on=['site_id', 'timestamp'], how='left')

    df.index = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M:%S')

    drop_features = ['timestamp']
    df.drop(drop_features, axis=1, inplace=True)

    if 'meter_reading' in df:
        mask = (df['site_id'] == 0) & (df['meter'] == 0)
        df.loc[mask, 'meter_reading'] = df.loc[mask, 'meter_reading'] * 0.2931
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

    encounter_features = ['primary_use']
    le = LabelEncoder()
    for ef in encounter_features:
        df[ef] = le.fit_transform(df[ef])

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
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
    df_std = df_rolled.std().reset_index().astype(np.float16)
    df_skew = df_rolled.skew().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f'{feature}_mean_window_{window}'] = df_mean[feature]
        df[f'{feature}_median_window_{window}'] = df_median[feature]
        df[f'{feature}_min_window_{window}'] = df_min[feature]
        df[f'{feature}_max_window_{window}'] = df_max[feature]
        df[f'{feature}_std_window_{window}'] = df_std[feature]
        df[f'{feature}_skew_window_{window}'] = df_skew[feature]

    return df


def filter2(df_train, building_id, meter, min_length, plot=False, verbose=False):

    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/122471

    if verbose:
        print("building_id: {}, meter: {}".format(building_id, meter))

    temp_df = df_train[(df_train['building_id'] == building_id) & (df_train['meter'] == meter)]
    target = temp_df['meter_reading'].values
    building_idx = []

    if any(target):

        splitted_target = np.split(target, np.where(target[1:] != target[:-1])[0] + 1)
        splitted_date = np.split(temp_df.index.values, np.where(target[1:] != target[:-1])[0] + 1)

        for i, x in enumerate(splitted_date):
            if len(x) > min_length:
                start = x[0]
                end = x[-1]
                value = splitted_target[i][0]
                idx = df_train.query('(@start <= index <= @end) and meter_reading == @value and meter == @meter '
                                     'and building_id == @building_id', engine='python').index.tolist()
                building_idx.extend(idx)
                if verbose:
                    print('Length: {},\t{}  -  {},\tvalue: {}'.format(len(x), start, end, value))

        building_idx = pd.Int64Index(building_idx)

        if plot:
            fig, axes = plt.subplots(nrows=2, figsize=(16, 18), dpi=100)

            temp_df.set_index('timestamp')['meter_reading'].plot(ax=axes[0])
            temp_df.drop(building_idx, axis=0).set_index('timestamp')['meter_reading'].plot(ax=axes[1])

            axes[0].set_title(f'Building {building_id} raw meter readings')
            axes[1].set_title(f'Building {building_id} filtered meter readings')

            plt.show()

    return building_idx


def sigma_filter(df, tolerance=3):

    df['meter_reading_ln'] = np.log1p(df.meter_reading)
    stats = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter'])\
                    .rolling(24*7, min_periods=2, center=True).meter_reading_ln.agg(['median'])
    std = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter']).meter_reading_ln.std()
    stats['max'] = np.expm1(stats['median'] + tolerance*std)
    stats['min'] = np.expm1(stats['median'] - tolerance*std)
    stats['median'] = np.expm1(stats['median'])
    df = df.merge(stats[['median', 'min', 'max']], left_on=['building_id', 'meter', 'timestamp'], right_index=True)

    return df[(df.meter_reading <= df['max']) & (df.meter_reading >= df['min'])]