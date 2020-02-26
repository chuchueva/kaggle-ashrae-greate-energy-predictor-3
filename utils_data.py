import gc
import inspect
import pandas as pd
import numpy as np
import math as math
import constants as c
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from math import sqrt


def read_train_data(site_id_list, meter_type_list, train_flag=True):

    df_tr_total = list()
    df_w = list()

    df_building_out = pd.read_csv(c.SOURCE_FOLDER + c.BUILDING_FILE)

    for site_id in site_id_list:

        weather_data = c.WEATHER_FILE_TEMPLATE % site_id
        df_w.append(pd.read_feather(c.CLEAN_FOLDER + weather_data))

        for meter_type in meter_type_list:

            if train_flag:
                data_file = c.TRAIN_FILE_TEMPLATE % (site_id, meter_type)
            else:
                data_file = c.TEST_FILE_TEMPLATE % (site_id, meter_type)

            try:
                df_raw = pd.read_feather(c.SPLIT_FOLDER + data_file)
                print('File %s is read' % (c.SPLIT_FOLDER + data_file))
            except:
                print('File %s does not exist' % (c.SPLIT_FOLDER + data_file))
                continue

            if train_flag:
                df_clean = manual_filtering(df_raw)
                df_tr_total.append(df_clean)
            else:
                df_tr_total.append(df_raw)

    df_tr_total_out = pd.concat(df_tr_total)
    df_w_out = pd.concat(df_w)

    return df_w_out, df_tr_total_out, df_building_out


def manual_filtering(df_train):

    filters_data = pd.read_csv(c.FILTER_FILE)

    df_train['IsFiltered'] = 0

    # Special treatment for site_0
    df_train.loc[df_train.query('building_id <= 104 and meter == 0 and timestamp < "2016-05-21 00:00:00"').index,
                 'IsFiltered'] = 1

    # General treatment for the rest
    meter_list = np.unique(df_train['meter'])
    building_list = np.unique(df_train['building_id'])

    for meter in meter_list:
        for building_id in building_list:
            build_settings = filters_data.query('meter == @meter and building_id == @building_id')
            if len(build_settings) == 1:
                min_edge = build_settings['min_edge'].values[0]
                max_edge = build_settings['max_edge'].values[0]
                if np.isnan(min_edge):
                    min_edge = -1
                if np.isnan(max_edge):
                    max_edge = 1e+10
                df_train.loc[df_train.query('building_id == @building_id and meter == @meter and '
                                            '(meter_reading <= @min_edge or meter_reading >= @max_edge)').index,
                             'IsFiltered'] = 1

    print('Filtered values num is %d' % np.sum(df_train['IsFiltered']))

    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)
    df_train.drop(columns=['IsFiltered'], inplace=True)

    return df_train.reset_index(drop=True)


def get_error(v1, v2):

    return sqrt(mean_squared_error(v1, v2))


def prepare_data(x, building_data, weather_data, make_log=True):

    df = x.merge(building_data, on='building_id', how='left')
    df = df.merge(weather_data, on=['site_id', 'timestamp'], how='left')

    df.index = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M:%S')

    drop_features = ['timestamp']
    df.drop(drop_features, axis=1, inplace=True)

    if ('meter_reading' in df) & make_log:
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

    cf = ['building_id', 'primary_use', 'hour', 'weekday', 'month', 'season']

    return df, cf


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


def get_seasonality_model(df, settings, regress_model=None, prophet_model=None):

    sins = [1]
    for w in sins:
        col_name = 'sin_' + str(w)
        df[col_name] = np.square(np.sin(df.index.dayofyear.astype('float64') / 365 * w * math.pi).values)

    y_pred_regress = np.zeros(len(df))
    y_pred_prophet = np.zeros(len(df))
    x_train = df[settings['col_name_x']]

    if regress_model is None:
        y_train = df[settings['col_name_y']]

    for m in range(len(settings['months_bundle'])):
        for d in range(len(settings['days_bundle'])):
            for h in range(len(settings['hours_bundle'])):

                mask = (np.isin(x_train.index.month,  settings['months_bundle'][m])) & \
                       (np.isin(x_train.index.dayofweek,  settings['days_bundle'][d])) & \
                       (np.isin(x_train.index.hour, settings['hours_bundle'][h]))

                if any(mask):

                    if regress_model is None:
                        regress_model = LinearRegression()
                        regress_model.fit(x_train[mask].values, y_train[mask].values)

                    if prophet_model is None:
                        prophet_model = Prophet()
                        df_train = pd.DataFrame()
                        df_train['ds'] = x_train[mask].index
                        df_train['y'] = y_train.loc[mask, settings['col_name_y']].values.ravel()
                        for f in settings['col_name_x']:
                            df_train[f] = x_train.loc[mask, f].values
                        prophet_model.fit(df_train, verbose=False)

                if type(regress_model) == LinearRegression:
                    y_m = regress_model.predict(x_train[mask].values)
                    y_pred_regress[mask] = y_m[:, 0]

                if type(prophet_model) == Prophet:
                    df_pred = pd.DataFrame()
                    df_pred['ds'] = x_train[mask].index
                    for f in settings['col_name_x']:
                        df_pred[f] = x_train.loc[mask, f].values
                    y_m = prophet_model.predict(df_pred)
                    y_pred_prophet[mask] = y_m['yhat'].values

    return regress_model, prophet_model, y_pred_regress, y_pred_prophet


def get_normalisation_index(predictors, df_columns):

    predictors = list(predictors)
    df_columns = list(df_columns)

    mask = np.empty(len(df_columns))
    mask.fill(np.nan)

    for i in range(len(df_columns)):
        if isinstance(df_columns[i], list):
            idx = predictors.index(df_columns[i][0])
        else:
            if len(df_columns[i]) == 1:
                idx = predictors.index(df_columns[i][0])
            else:
                idx = predictors.index(df_columns[i])
        mask[i] = idx

    mask = mask.astype('int')

    return mask


def do_normalisation(df, scaler):

    x = df.values.astype('float64')

    if any(scaler):
        mask = get_normalisation_index(scaler['predictors'], list(df.columns.values))
        set_p1 = np.array(scaler['p1'][mask])
        set_p2 = np.array(scaler['p2'][mask])
    else:
        set_p1 = x.mean(axis=0)
        scaler['p1'] = set_p1
        set_p2 = x.std(axis=0)
        scaler['p2'] = set_p2
        scaler['predictors'] = df.columns

    x -= set_p1
    x /= set_p2

    df_out = pd.DataFrame(x, columns=df.columns, index=df.index)

    return df_out, scaler


def undo_normalisation(df, scaler):

    mask = get_normalisation_index(scaler['predictors'], list(df.columns.values))
    x = df.values
    y = x * scaler['p2'][mask] + scaler['p1'][mask]
    df_out = pd.DataFrame(y, columns=df.columns, index=df.index)

    return df_out


def blend(df_p, b_list):

    x = np.nanmean(df_p[b_list], axis=1)
    x = np.expm1(x)
    y = df_p['row_id']
    xy = np.column_stack((y, x))

    return xy


def flat_list(input_list):

    f_list = []

    for sublist_1 in input_list:
        if isinstance(sublist_1, list):
            for sublist_2 in sublist_1:
                if isinstance(sublist_2, list):
                    for item in sublist_2:
                        f_list.append(item)
                else:
                    f_list.append(sublist_2)
        else:
            f_list.append(sublist_1)

    f_list = list(np.unique(f_list))

    return f_list


def get_name(model_type, site=None, meter=None, building=None, cv=None):

    # '%s_site_%s_meter_%s_building_%s_cv_%s'
    site = str(np.unique(site))[1:-1].replace(' ', '_')
    meter = str(np.unique(meter))[1:-1].replace(' ', '_')

    n = c.MODEL_NAME_TEMPLATE % (model_type, site, meter, building, cv)
    n = n.replace('_site_None', '')
    n = n.replace('_meter_None', '')
    n = n.replace('_building_None', '')
    n = n.replace('_cv_None', '')

    return n