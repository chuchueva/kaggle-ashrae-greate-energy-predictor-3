import gc
import inspect
import pandas as pd
import numpy as np
import math as math
import constants as c
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import Bounds
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from math import sqrt


def get_regress(df, settings, model=None):

    x_train = df[settings['col_name_x']]
    y_pred_regress = np.zeros(len(df))

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

                    if type(model) == LinearRegression:
                        y_m = model.predict(x_train[mask].values)
                        y_pred_regress[mask] = y_m[:, 0]

    return model, y_pred_regress


def get_prophet(df, settings, model=None):

    sins = [1]
    for w in sins:
        col_name = 'sin_' + str(w)
        df[col_name] = np.square(np.sin(df.index.dayofyear.astype('float64') / 365 * w * math.pi).values)

    y_pred_prophet = np.zeros(len(df))
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
                        model = Prophet()
                        df_train = pd.DataFrame()
                        df_train['ds'] = x_train[mask].index
                        df_train['y'] = y_train.loc[mask, settings['col_name_y']].values.ravel()
                        for f in settings['col_name_x']:
                            df_train[f] = x_train.loc[mask, f].values
                        model.fit(df_train, verbose=False)

                    if type(model) == Prophet:
                        df_pred = pd.DataFrame()
                        df_pred['ds'] = x_train[mask].index
                        for f in settings['col_name_x']:
                            df_pred[f] = x_train.loc[mask, f].values
                        y_m = model.predict(df_pred)
                        y_pred_prophet[mask] = y_m['yhat'].values

    return model, y_pred_prophet


def model(x, u):
    return x[0] * u[:, 0] + x[1] * u[:, 1] + x[2] * u[:, 2]


def deviation(x, u, y):
    return model(x, u) - y


def get_smart_blend(x_test, x_train, actuals):

    mask = np.invert(np.isnan(x_train[:, 0]))
    u = x_train[mask]
    y = actuals[mask]
    x0 = np.array([0.334, 0.333, 0.333])
    res = least_squares(deviation, x0, bounds=(0.1, 0.5), args=(u, y), verbose=1)
    print(np.round(res.x, 4))
    pred = res.x[0] * x_test[:, 0] + res.x[1] * x_test[:, 1] + res.x[2] * x_test[:, 2]

    return pred

