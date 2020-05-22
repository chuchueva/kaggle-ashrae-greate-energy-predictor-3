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

        'lgb_params': {

            'objective': 'regression',
            'num_leaves': 10,
            'learning_rate': 0.01,
            'num_boost_round': boost_settings('lgb_params', 'num_boost_round', site_id),
            'metric': 'rmse'

        },

        'xgb_params': {
            'max_depth': 10,
            'learning_rate': 0.15,
            'n_estimators': 20
        },

        'cat_params': {
            'depth': 10,
            'learning_rate': 0.7,
            'iterations': 25,
            'eval_metric': 'RMSE',
            'verbose': True
        },

        'network_params': {
            'horison': 1,
            'neuron_number': 24,
            'epochs': 35,
            'batch_size': 1000,
            'learning_rate': 0.0005
        }
    }

    ss = settings.get(setting_type)

    return ss


def boost_settings(model, setting_name, site_id):

    settings = {
        'lgb_params': {
            'num_boost_round': {
                0: 400,
                1: 400,
                2: 400,
                3: 400,
                4: 400,
                5: 400,
                6: 1000,
                7: 1000,
                8: 400,
                9: 1000,
                10: 1000,
                11: 400,
                12: 400,
                13: 1000,
                14: 1000,
                15: 400
            }
        }
    }

    ss = settings.get(model).get(setting_name).get(site_id)

    return ss


def get_feature_settings():
    fs = dict()

    fs['do_humidity'] = True
    fs['do_holidays'] = True
    fs['do_building_meter_reading_count'] = False

    fs['weather_lag_vars'] = ['air_temperature']
    fs['weather_lag_values'] = [1, 2, 3, 4, 5, 6, 12, 18, 24]

    fs['weather_average_vars'] = []
    fs['weather_average_window'] = 24

    fs['consumption_sins'] = [1, 5]

    return fs
