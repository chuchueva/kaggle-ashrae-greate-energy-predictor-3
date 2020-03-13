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
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as opt
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()

start_time = time.time()

random.seed(c.FAVOURITE_NUMBER)


'''

Read data

'''
# site_id_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
# meter_list = [[0], [1], [2], [3]]
# site_id_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
# meter_list = [[0, 1, 2, 3]]

site_id_list = [[0]]
meter_list = [[0], [1], [2], [3]]
model_type = 'ctboost'              # xgboost, lgboost, ctboost, network

meter_list_uni = ud.flat_list(meter_list)
site_id_list_uni = ud.flat_list(site_id_list)
df_weather, df_train, df_building = ud.read_train_data(site_id_list_uni, meter_list_uni,
                                                       train_flag=True, folder=c.CLEAN_FOLDER)

df_train = ud.prepare_data(df_train, df_building, df_weather)
df_train, categorical_features = ud.feature_engineering(df_train)

target_name = 'meter_reading'
models = dict()

for site_id in site_id_list:

    for meter in meter_list:

        features_list = df_train.columns[np.invert(df_train.columns.isin(['meter_reading']))]
        kf = KFold(n_splits=us.get_trees_settings('cv'), random_state=c.FAVOURITE_NUMBER, shuffle=True)

        if len(site_id) == 1:
            features_list = features_list[features_list != 'site_id']
        if len(meter) == 1:
            features_list = features_list[features_list != 'meter']

        mask = (df_train['site_id'].isin(site_id)) & (df_train['meter'].isin(meter))

        if any(mask):

            print('****************************************************************************')
            print('Site %d meter %d %s training:' % (site_id[0], meter[0], model_type))

            X_train_site = df_train.loc[mask, features_list].reset_index(drop=True)
            y_train_site = df_train.loc[mask, target_name]
            name = ud.get_name('feature_list', site=site_id, meter=meter)
            models[name] = features_list

            for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):

                print('Site %d meter %d fold %d training:' % (site_id[0], meter[0], fold))

                X_train, X_valid = X_train_site.loc[train_index], X_train_site.loc[valid_index]
                y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]
                y_pred_train_site = np.zeros(len(X_train_site))

                if model_type == 'lgboost':

                    # LGBT (light gradient boost tree, and what are you thinking?)
                    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
                    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
                    watchlist = [d_train, d_valid]
                    model_lgb = lgb.train(us.get_trees_settings('lgb_params'),
                                          train_set=d_train,
                                          valid_sets=watchlist,
                                          verbose_eval=100)
                    name = ud.get_name(model_type, site=site_id, meter=meter, cv=str(fold))
                    models[name] = model_lgb
                    y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
                    y_pred_train_site[valid_index] = y_pred_valid
                    y_pred_train = model_lgb.predict(X_train, num_iteration=model_lgb.best_iteration)
                    y_pred_train_site[train_index] = y_pred_train
                    df_train.loc[mask, '%s_%d' % (model_type, fold)] = y_pred_train_site

                    del y_pred_train_site, y_pred_train, y_pred_valid, model_lgb, watchlist, d_train, d_valid
                    gc.collect()

                if model_type == 'xgboost':

                    # XGBoost
                    model_xgb = XGBRegressor(**us.get_trees_settings('xgb_params'))
                    model_xgb.fit(X_train, y_train,
                                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                                  eval_metric='rmse',
                                  verbose=True,
                                  early_stopping_rounds=5)
                    name = ud.get_name(model_type, site=site_id, meter=meter, cv=str(fold))
                    models[name] = model_xgb
                    y_pred_valid = model_xgb.predict(X_valid, ntree_limit=model_xgb.best_ntree_limit)
                    y_pred_train_site[valid_index] = y_pred_valid
                    y_pred_train = model_xgb.predict(X_train, ntree_limit=model_xgb.best_ntree_limit)
                    y_pred_train_site[train_index] = y_pred_train
                    df_train.loc[mask, '%s_%d' % (model_type, fold)] = y_pred_train_site

                    del y_pred_train_site, y_pred_train, y_pred_valid, model_xgb
                    gc.collect()

                if model_type == 'ctboost':

                    # CatBoost
                    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
                    cat_model = CatBoostRegressor(**us.get_trees_settings('cat_params'))
                    cat_model.fit(X_train, y_train,
                                  eval_set=(X_valid, y_valid),
                                  cat_features=categorical_features_indices)
                    name = ud.get_name(model_type, site=site_id, meter=meter, cv=str(fold))
                    models[name] = cat_model
                    y_pred_valid = cat_model.predict(X_valid)
                    y_pred_train_site[valid_index] = y_pred_valid
                    y_pred_train = cat_model.predict(X_train)
                    y_pred_train_site[train_index] = y_pred_train
                    df_train.loc[mask, '%s_%d' % (model_type, fold)] = y_pred_train_site

                    del y_pred_train_site, y_pred_train, y_pred_valid, cat_model
                    gc.collect()

                if model_type == 'network':

                    settings = us.get_trees_settings('network_params')
                    data = df_train.loc[mask, [*features_list, target_name]].reset_index(drop=True)
                    data.dropna(axis=1, inplace=True)
                    data_scaled, scaler = ud.do_normalisation(data, dict())
                    X_train = data_scaled.loc[train_index, features_list.isin(data.columns)]
                    X_valid = data_scaled.loc[valid_index, features_list.isin(data.columns)]
                    y_train = data_scaled.iloc[train_index, [*df_train.columns].index(target_name)]
                    y_valid = data_scaled.iloc[valid_index, [*df_train.columns].index(target_name)]

                    # Dense
                    network = Sequential()
                    network.add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))
                    network.add(Dense(settings['neuron_number'], activation='relu'))
                    network.add(Dense(settings['horison']))
                    custom_optimiser = opt.Adam(lr=settings['learning_rate'])
                    network.compile(optimizer=custom_optimiser, loss='mse')
                    history = network.fit(X_train, y_train,
                                          epochs=settings['epochs'],
                                          batch_size=settings['batch_size'], verbose=True,
                                          validation_data=(X_valid, y_valid))

                    X_ = data_scaled.loc[train_index, features_list.isin(data.columns)]

                    name = ud.get_name(model_type, site=site_id, meter=meter, cv=str(fold))
                    models[name] = network
                    y_pred_valid = network.predict(X_valid)
                    y_pred_train_site[valid_index] = y_pred_valid.ravel()
                    y_pred_train = network.predict(X_train)
                    y_pred_train_site[train_index] = y_pred_train.ravel()
                    y_pred_train_site = ud.undo_normalisation(pd.DataFrame(y_pred_train_site, columns=[target_name]),
                                                              scaler)
                    df_train.loc[mask, '%s_%d' % (model_type, fold)] = y_pred_train_site.values

            del X_train, X_valid, y_train, y_valid
            gc.collect()

df_train[model_type] = np.nanmean(df_train[[model_type + '_' + str(f) for f in range(us.get_trees_settings('cv'))]],
                                  axis=1)
mask = np.invert(np.isnan(df_train[model_type].values))
print('RMSLE %s: %.2f' % (model_type, ud.get_error(df_train.loc[mask, 'meter_reading'].values,
                                                   df_train.loc[mask, model_type].values)))

models['site_id_list'] = site_id_list
models['meter_list'] = meter_list
models['model_type'] = model_type

model_file = ud.get_name(model_type, site_id_list, meter_list) + '.pickle'
filename = c.MODEL_FOLDER + model_file
model_save = open(filename, 'wb')
pickle.dump(models, model_save)
model_save.close()
print('%s models are saved in %s' %(model_type, filename))
