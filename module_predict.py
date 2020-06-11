import os as os
import gc as gc
import time as time
import random as random
import numpy as np
import pandas as pd
import utils_data as ud
import utils_settings as us
import utils_model as um
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

result_to_update_file = 'late_model_nan.csv'
result_new_file = 'late_model_154.csv'
model_folder = 'model_154_new_version_71'

model_files = os.listdir(c.MODEL_FOLDER + model_folder)

model_list = []

for f in model_files:
    model_load = open(c.MODEL_FOLDER + model_folder + '/' + f, 'rb')
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

df_building = ud.read_building_data()
df_weather = ud.read_weather_data(site_id_list)
df_predict = ud.read_consumption_data(site_id_list, meter_list, data_type='test')

if 'meter_reading' in df_predict:
    df_predict.drop(columns=['meter_reading'], inplace=True)

df_weather = ud.weather_feature_engineering(df_weather)
df_predict = ud.consumption_feature_engineering(df_predict)
df_predict = ud.prepare_data(df_predict, df_building, df_weather)

print('****************************************************************************')
print('Data is ready, time %.0f sec' % (time.time() - start_time))

blend_list = []

for model in model_list:

    site_id_list = [model['site_id_list']]
    meter_list = model['meter_list']

    '''
        
        Predict by bunches
        
    '''

    if model['model_type'] in ['lgboost', 'xgboost', 'ctboost', 'network']:

        features_list = df_predict.columns[np.invert(df_predict.columns.isin(['row_id']))]

        for site_id in site_id_list:

            for meter in meter_list:

                mask = (df_predict['site_id'].isin(site_id)) & (df_predict['meter'].isin(meter))

                if any(mask):

                    name = ud.get_name('feature_list', site=site_id, meter=meter)
                    features_list = model[name]
                    X_predict = df_predict.loc[mask, features_list].reset_index(drop=True)

                    for fold in range(c.K_FOLD):

                        col_name = '%s_%d' % (model['model_type'], fold)

                        # LGBoost
                        if model['model_type'] == 'lgboost':
                            model_name = ud.get_name(model['model_type'], site=site_id, meter=meter, cv=str(fold))
                            y_pred = model[model_name].predict(X_predict, num_iteration=model[model_name].best_iteration)

                        # XGBoost
                        if model['model_type'] == 'xgboost':
                            model_name = ud.get_name(model['model_type'], site=site_id, meter=meter, cv=str(fold))
                            y_pred = model[model_name].predict(X_predict, ntree_limit=model[model_name].best_ntree_limit)

                        # CatBoost
                        if model['model_type'] == 'ctboost':
                            model_name = ud.get_name(model['model_type'], site=site_id, meter=meter, cv=str(fold))
                            categorical_features_indices = np.where(X_predict.dtypes != np.float)[0]
                            y_pred = model[model_name].predict(X_predict)

                        # Network
                        if model['model_type'] == 'network':
                            model_name = ud.get_name(model['model_type'], site=site_id, meter=meter, cv=str(fold))
                            scaler_name = ud.get_name('scaler', site=site_id, meter=meter, cv=str(fold))
                            X_predict['floor_count'] = X_predict['floor_count'].fillna(0)
                            X_predict['year_built'] = X_predict['year_built'].fillna(0)
                            X_predict['square_feet'] = X_predict['square_feet'].fillna(0)
                            X_predict_scaled, _ = ud.do_normalisation(X_predict, model[scaler_name])
                            y_pred_scaled = model[model_name].predict(X_predict_scaled)
                            y_pred = ud.undo_normalisation(pd.DataFrame(y_pred_scaled, columns=['meter_reading']),
                                                           model[scaler_name]).values

                        df_predict.loc[mask, col_name] = y_pred

                        if col_name not in blend_list:
                            blend_list.append(col_name)

                        print('%s is done, time %.0f sec' %
                              (ud.get_name(model['model_type'], site_id, meter, cv=fold), time.time() - start_time))

                    del X_predict
                    gc.collect()

    '''
    
        Predict by buildings regress & prophet
    
    '''

    if model['model_type'] in ['regress', 'prophet']:

        settings = us.get_regress_settings()

        if model['model_type'] not in df_predict:
            df_predict[model['model_type']] = np.nan

        for single_building in model['building_list']:

            meter = single_building[0]
            building = single_building[1]

            mask = (df_predict['meter'] == meter) & (df_predict['building_id'] == building)
            df_sample_building = df_predict[mask]

            if len(df_sample_building) > 0:
                name = ud.get_name(model['model_type'], meter=meter, building=building)
                if model['model_type'] == 'regress':
                    _, pred = um.get_regress(df_sample_building, settings, model=model[name])
                elif model['model_type'] == 'prophet':
                    _, pred = um.get_prophet(df_sample_building, settings, model=model[name])

                # plt.plot(np.arange(pred.shape[0]), pred, '*', label='model')
                # plt.show()
                df_predict.loc[mask, model['model_type']] = pred.ravel()

            print('%s model is done for building %d, time %.0f sec' %
                  (model['model_type'], building, time.time() - start_time))

    print('%s: time %.0f sec' % (model['model_type'], time.time() - start_time))
    print('****************************************************************************')

if 'regress' in df_predict:
    blend_list.append('regress')

if 'prophet' in df_predict:
    blend_list.append('prophet')

'''

    Blending & writing output file

'''

print('Blend_list: %s' % blend_list)
output_array = ud.get_average(df_predict, blend_list)

if output_array.shape[1] == 2:
    df_output = pd.read_csv(result_to_update_file)
    df_output.index = df_output['row_id'].values
    df_output.drop(columns=['row_id'], inplace=True)
    df_output.loc[output_array[:, 0], 'meter_reading'] = output_array[:, 1]
    df_output.to_csv(result_new_file, index=True, index_label='row_id', float_format='%.0f')
    print('File is written, time %.0f sec' % (time.time() - start_time))
