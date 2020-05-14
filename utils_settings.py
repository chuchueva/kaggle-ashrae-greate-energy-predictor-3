import constants as c
import numpy as np


def get_regress_settings():

    regress_settings = {
        'col_name_x': ['sin_1', 'air_temperature'],
        'col_name_y': ['meter_reading'],
        'months_bundle': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        'days_bundle': [[0, 1, 2, 3, 4, 5, 6]],
        'hours_bundle': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
    }

    return regress_settings


def get_trees_settings(setting_type, site_id=None):

    settings = {

        'cv': 2,

        'lgb_params': {

            # option 1:
            'objective': 'regression',
            'num_leaves': 10,
            'learning_rate': 0.01,
            # 'num_boost_round': lgboost_num_boost_round(site_id),
            'num_boost_round': 400,
            'metric': 'rmse'

            # option 2 (looks overfitted)
            # 'num_leaves': 2 ** 9,
            # 'learning_rate': 0.05,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 5,
            # 'feature_fraction': 0.8,
            # 'feature_fraction_bynode': 0.7,
            # 'min_data_in_leaf': 1000,
            # 'max_depth': -1,
            # 'objective': 'regression',
            # 'seed': c.FAVOURITE_NUMBER,
            # 'feature_fraction_seed': c.FAVOURITE_NUMBER,
            # 'bagging_seed': c.FAVOURITE_NUMBER,
            # 'drop_seed': c.FAVOURITE_NUMBER,
            # 'data_random_seed': c.FAVOURITE_NUMBER,
            # 'boosting_type': 'gbdt',
            # 'verbose': 1,
            # 'metric': 'rmse'
        },

        'xgb_params': {
            'max_depth': 35,
            'learning_rate': 0.05,
            'n_estimators': 35
        },

        'cat_params': {
            'depth': 10,
            'learning_rate': 1,
            'iterations': 15,
            'eval_metric': 'RMSE',
            'random_seed': c.FAVOURITE_NUMBER,
            'verbose': True
        },

        'network_params': {
            'horison': 1,
            'neuron_number': 24,
            'epochs': 30,
            'batch_size': 1000,
            'learning_rate': 0.00002
        }
    }

    ss = settings.get(setting_type)

    return ss


def lgboost_num_boost_round(site_id):

    settings = {
        0: 180,
        1: 210,
        2: 720,
        3: 75,
        4: 90,
        5: 120,
        6: 1500,
        7: 1500,
        8: 90,
        9: 1500,
        10: 1500,
        11: 240,
        12: 60,
        13: 1500,
        14: 1500,
        15: 375

    }

    ss = settings.get(site_id)
    if ss is not None:
        ss = ss

    return ss


def get_feature_settings():

    fs = dict()

    fs['do_humidity'] = False
    fs['do_holidays'] = False
    fs['do_building_meter_reading_count'] = False

    fs['weather_lag_vars'] = ['air_temperature']
    fs['weather_lag_values'] = [1, 2, 3, 4, 5, 6, 12, 18, 24]

    fs['weather_average_vars'] = []
    fs['weather_average_window'] = 24

    fs['consumption_sins'] = [1, 5]

    return fs
