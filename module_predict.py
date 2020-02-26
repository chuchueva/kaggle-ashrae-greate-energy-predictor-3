import gc as gc
import time as time
import random as random
import numpy as np
import pandas as pd
import utils_data as ud
import utils_settings as us
import lightgbm as lgb
import pickle as pickle
import constants as c
import itertools as itertools
import keras.optimizers as opt
from keras.models import Sequential
from keras import layers
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()

start_time = time.time()

random.seed(c.FAVOURITE_NUMBER)

result_to_update_file = 'late_model_01.csv'
result_new_file = 'late_model_05.csv'

model_files = ['model_regress.pickle', 'model_trees.pickle']
model_list = []

for f in model_files:
    model_load = open(c.MODEL_FOLDER + f, 'rb')
    model_list.append(pickle.load(model_load))
    print('Model settings are uploaded from ' + c.MODEL_FOLDER + f)

'''

Read data

'''

site_id_list = []
meter_list = []

for model in model_list:
    site_id_list.append(model['site_id_list'])
    meter_list.append(model['meter_list'])

meter_list = ud.flat_list(meter_list)
site_id_list = ud.flat_list(site_id_list)

df_weather, df_predict, df_building = ud.read_train_data(site_id_list, meter_list, train_flag=False)
df_predict = ud.prepare_data(df_predict, df_building, df_weather)
df_predict, categorical_features = ud.feature_engineering(df_predict)

print('Data is ready')

for model in model_list:

    site_id_list = model['site_id_list']
    meter_list = model['meter_list']

    '''
        
        Predict with trees
        
    '''

    if model['type'] == 'tree':

        features_list = df_predict.columns[np.invert(df_predict.columns.isin(['row_id']))]

        for site_id in site_id_list:

            for meter in meter_list:

                mask = (df_predict['site_id'].isin(site_id)) & (df_predict['meter'].isin(meter))

                if any(mask):

                    name = ud.get_name('feature_list', site=site_id, meter=meter)
                    features_list = model[name]
                    X_predict = df_predict.loc[mask, features_list].reset_index(drop=True)

                    for fold in range(us.get_trees_settings('cv')):

                        # LGBoost
                        model_name = ud.get_name('lgb', site=site_id, meter=meter, cv=str(fold))
                        y_pred = model[model_name].predict(X_predict, num_iteration=model[model_name].best_iteration)
                        df_predict.loc[mask, 'lgb_model_%d' % fold] = y_pred
                        print('LGBoost for fold %d is done' % fold)

                        # XGBoost
                        model_name = ud.get_name('xgb', site=site_id, meter=meter, cv=str(fold))
                        y_pred = model[model_name].predict(X_predict, ntree_limit=model[model_name].best_ntree_limit)
                        df_predict.loc[mask, 'xgb_model_%d' % fold] = y_pred
                        print('XGBoost for fold %d is done' % fold)

                        # CatBoost
                        model_name = ud.get_name('cat', site=site_id, meter=meter, cv=str(fold))
                        categorical_features_indices = np.where(X_predict.dtypes != np.float)[0]
                        y_pred = model[model_name].predict(X_predict)
                        df_predict.loc[mask, 'cat_model_%d' % fold] = y_pred
                        print('CatBoost for fold %d is done' % fold)

                    del X_predict
                    gc.collect()

    '''

        Predict with regress & prophet

    '''

    if model['type'] == 'seasonality':

        settings = us.get_regress_settings()
        building_regress = model['building_list'][c.MODEL_TYPE_REGRESS]
        building_prophet = model['building_list'][c.MODEL_TYPE_PROPHET]
        df_predict[c.MODEL_TYPE_REGRESS] = np.nan
        df_predict[c.MODEL_TYPE_PROPHET] = np.nan

        for single_building in building_regress:

            meter = single_building[0]
            building = single_building[1]

            mask = (df_predict['meter'] == meter) & (df_predict['building_id'] == building)
            df_sample_building = df_predict[mask]

            if len(df_sample_building) > 0:
                name = ud.get_name(c.MODEL_TYPE_REGRESS, meter=meter, building=building)
                _, _, pred_r, _ = ud.get_seasonality_model(df_sample_building, settings,
                                                           regress_model=model[name],
                                                           prophet_model=False)
                df_predict.loc[mask, c.MODEL_TYPE_REGRESS] = pred_r

        for single_building in building_prophet:

            meter = single_building[0]
            building = single_building[1]

            mask = (df_predict['meter'] == meter) & (df_predict['building_id'] == building)
            df_sample_building = df_predict[mask]

            if len(df_sample_building) > 0:
                name = ud.get_name(c.MODEL_TYPE_PROPHET, meter=meter, building=building)
                _, _, _, pred_p = ud.get_seasonality_model(df_sample_building, settings,
                                                           regress_model=False,
                                                           prophet_model=model[name])
                df_predict.loc[mask, c.MODEL_TYPE_PROPHET] = pred_p

        print('Regress and prophet models are done!')

'''

    Blending & writing output file

'''

blend_list = np.unique([['lgb_model_' + str(f) for f in range(us.get_trees_settings('cv'))],
                        ['xgb_model_' + str(f) for f in range(us.get_trees_settings('cv'))],
                        ['cat_model_' + str(f) for f in range(us.get_trees_settings('cv'))],
                        ['regress', 'prophet']])

output_array = ud.blend(df_predict, blend_list)
print('Blending is done!')

df_output = pd.read_csv(result_to_update_file)
df_output.index = df_output['row_id'].values
df_output.drop(columns=['row_id'], inplace=True)
df_output.loc[output_array[:, 0], 'meter_reading'] = output_array[:, 1]
df_output.to_csv(result_new_file, index=True, index_label='row_id', float_format='%.2f')
print('File is written, time %.0f sec' % (time.time() - start_time))
