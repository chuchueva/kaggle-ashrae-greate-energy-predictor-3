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
        },
        'xgb_params': {
            'n_estimators': {
                0: 20,
                1: 20,
                2: 20,
                3: 20,
                4: 20,
                5: 20,
                6: 25,
                7: 25,
                8: 20,
                9: 25,
                10: 25,
                11: 20,
                12: 20,
                13: 25,
                14: 25,
                15: 20
            }
        }
    }

    ss = settings.get(model).get(setting_name).get(site_id)
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
