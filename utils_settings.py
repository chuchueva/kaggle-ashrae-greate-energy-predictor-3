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


def get_trees_settings(site_id):

    settings = {

        'cv': 2,

        'lgb_params': {
            'objective': 'regression',
            'num_leaves': 10,
            'learning_rate': 0.01,
            'num_boost_round': 400,
            'metric': 'rmse'
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

    ss = settings.get(str(site_id))

    return ss