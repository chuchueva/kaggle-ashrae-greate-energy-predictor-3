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

        # Option 1: by site, meters joined
        # site_id_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
        # meter_list = [[0, 1, 2, 3]]

        # 'lgb_params': {
        #     'objective': 'regression',
        #     'num_leaves': 10,
        #     'learning_rate': 0.01,
        #     'num_boost_round': boost_settings('lgb_params', 'num_boost_round', site_id),
        #     'metric': 'rmse'
        # },
        #
        'xgb_params': {
            'max_depth': 10,
            'learning_rate': 0.15,
            'n_estimators': 20
        }
        #
        # 'cat_params': {
        #     'depth': 10,
        #     'learning_rate': 0.7,
        #     'iterations': 25,
        #     'eval_metric': 'RMSE',
        #     'verbose': True
        # },
        #
        # 'network_params': {
        #     'horison': 1,
        #     'neuron_number': 60,
        #     'epochs': 35,
        #     'batch_size': 1000,
        #     'learning_rate': 0.001
        # }

        # Option 2: by site bunches, meters separate
        # meter_list = [[0], [1], [2], [3]]

        # 'lgb_params': {

            # meter 0, site_id_list = [[0, 8], [1, 5, 12], [3, 6, 7], [11, 14, 15], [9, 13], [4, 10, 2]],
            # edge cv = 0.265
            # 'objective': 'regression',
            # 'num_leaves': 30,
            # 'learning_rate': 0.01,
            # 'num_boost_round': 1200,
            # 'metric': 'rmse'

            # meter 1, site_id_list = [[0, 8], [1, 5, 12], [3, 6, 7], [11, 14, 15], [9, 13], [4, 10, 2]],
            # edge cv = 0.937
            # meter 2, site_id_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], edge cv = 1.125
            # meter 3, site_id_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], edge cv = 1.232
            # 'objective': 'regression',
            # 'num_leaves': 10,
            # 'learning_rate': 0.01,
            # 'num_boost_round': 1200,
            # 'metric': 'rmse'

        # },

        # 'xgb_params': {
        #     'max_depth': 10,
        #     'learning_rate': 0.15,
        #     'n_estimators': 30
        # },
        #
        # 'network_params': {
        #     'horison': 1,
        #     'neuron_number': 144,
        #     'epochs': 35,
        #     'batch_size': 750,
        #     'learning_rate': 0.001
        # }
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

    fs['do_humidity'] = False
    fs['do_holidays'] = False
    fs['do_building_meter_reading_count'] = False

    fs['weather_lag_vars'] = ['air_temperature']
    fs['weather_lag_values'] = [1, 2, 3, 4, 5, 6, 12, 18, 24]

    fs['weather_average_vars'] = []
    fs['weather_average_window'] = 24

    fs['consumption_sins'] = [1, 5]

    return fs
