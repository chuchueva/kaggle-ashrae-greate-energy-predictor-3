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
# Option 1: by site, meter joined
site_id_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
meter_list = [[0, 1, 2, 3]]

# Option 2: by site_bunch, meter separate
# site_id_list = [[0, 8], [1, 5, 12], [4, 10], [3, 6, 7, 11, 14, 15], [9, 13], [2]]
# site_id_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
# site_id_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
# site_id_list = [[0, 2, 6], [7, 9, 10], [11, 13, 14]]
# site_id_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
# site_id_list = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
# meter_list = [[0]]

model_type_list = ['xgboost']           # 'xgboost', 'lgboost', 'ctboost' , 'network'
target_name = 'meter_reading'

for model_type in model_type_list:

    rmsle_total = []

    for site_id in site_id_list:

        meter_list_uni = ud.flat_list(meter_list)
        site_id_list_uni = ud.flat_list(site_id)
        df_building = ud.read_building_data()
        df_weather = ud.read_weather_data(site_id_list_uni)
        df_train = ud.read_consumption_data(site_id_list_uni, meter_list_uni, data_type='train')

        if 'row_id' in df_train:
            df_train.drop(columns=['row_id'], inplace=True)

        if len(df_train) > 0:

            df_weather = ud.weather_feature_engineering(df_weather)
            df_train = ud.consumption_feature_engineering(df_train)
            df_train = ud.prepare_data(df_train, df_building, df_weather)

            categorical_features = ['site_id', 'building_id', 'meter', 'primary_use', 'hour', 'weekday', 'month',
                                    'season']
            features_list = list(df_train.columns)
            features_list.remove('meter_reading')
            models = dict()

            # Option 1: by meter joined
            if len(meter_list[0]) == 4:
                meter_list_by_site = []
                for id in site_id:
                    meter_list_by_site.append(c.SITE_METER.get(id))
                meter_list_by_site = np.unique(meter_list_by_site)

            meter_done = []

            for meter in meter_list:

                if len(site_id) == 1 and 'site_id' in features_list and 'site_id' in categorical_features:
                    features_list.remove('site_id')
                    categorical_features.remove('site_id')
                if len(meter) == 1 and 'meter' in features_list and 'meter' in categorical_features:
                    features_list.remove('meter')
                    categorical_features.remove('meter')

                if len(meter_list[0]) == 4:
                    meter = list(np.array(meter)[np.isin(meter, meter_list_by_site)])

                kf = KFold(n_splits=c.K_FOLD, random_state=c.FAVOURITE_NUMBER, shuffle=True)
                mask = (df_train['site_id'].isin(site_id)) & (df_train['meter'].isin(meter))

                if any(mask):

                    print('Feature list: %s' % features_list)

                    print('Train %s:' % (ud.get_name(model_type, site_id, meter)))

                    X_train_site = df_train.loc[mask, features_list].reset_index(drop=True)
                    y_train_site = df_train.loc[mask, target_name]
                    name = ud.get_name('feature_list', site=site_id, meter=meter)
                    models[name] = features_list

                    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):

                        print('%.2f: %s training:' % (time.time()-start_time,
                                                      ud.get_name(model_type, site_id, meter, cv=fold)))

                        X_train, X_valid = X_train_site.loc[train_index], X_train_site.loc[valid_index]
                        y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]
                        y_pred_train_site = np.zeros(len(X_train_site))

                        if model_type == 'lgboost':

                            # LGBT (light gradient boost tree, and what are you thinking?)
                            d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
                            d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
                            watchlist = [d_train, d_valid]
                            model_lgb = lgb.train(us.get_trees_settings('lgb_params', site_id=site_id[0]),
                                                  train_set=d_train,
                                                  valid_sets=watchlist,
                                                  verbose_eval=30)
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
                            model_xgb = XGBRegressor(**us.get_trees_settings('xgb_params', site_id=site_id[0]))
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
                            data['floor_count'] = data['floor_count'].fillna(0)
                            data['year_built'] = data['year_built'].fillna(0)
                            data['square_feet'] = data['square_feet'].fillna(0)
                            data.dropna(axis=0, inplace=True)
                            data_scaled, scaler = ud.do_normalisation(data, dict())
                            X_train = data_scaled.loc[train_index, features_list]
                            X_valid = data_scaled.loc[valid_index, features_list]
                            y_train = data_scaled.loc[train_index, target_name]
                            y_valid = data_scaled.loc[valid_index, target_name]

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

                            name = ud.get_name(model_type, site=site_id, meter=meter, cv=str(fold))
                            models[name] = network
                            scaler_name = ud.get_name('scaler', site=site_id, meter=meter, cv=str(fold))
                            models[scaler_name] = scaler

                            y_pred_valid = network.predict(X_valid)
                            y_pred_train_site[valid_index] = y_pred_valid.ravel()
                            y_pred_train = network.predict(X_train)
                            y_pred_train_site[train_index] = y_pred_train.ravel()

                            y_pred_train_site = ud.undo_normalisation(pd.DataFrame(y_pred_train_site,
                                                                                   columns=[target_name]), scaler)
                            df_train.loc[mask, '%s_%d' % (model_type, fold)] = y_pred_train_site.values

                    meter_done.append(meter)

                    del X_train, X_valid, y_train, y_valid
                    gc.collect()

                    cols = [model_type + '_' + str(f) for f in range(c.K_FOLD)]
                    df_train[model_type] = np.nanmean(df_train[cols], axis=1)
                    mask = np.invert(np.isnan(df_train[model_type].values))
                    rmsle = ud.get_error(df_train.loc[mask, 'meter_reading'].values, df_train.loc[mask, model_type].values)
                    rmsle_total.append([site_id, rmsle])
                    print('****************************************************************************')
                    print('%.2f: RMSLE site %d model %s = %.2f' % ((time.time() - start_time), site_id[0], model_type, rmsle))
                    print('****************************************************************************')

            models['site_id_list'] = site_id
            models['meter_list'] = meter_done
            models['model_type'] = model_type

            model_file = ud.get_name(model_type, site_id, meter_done) + '.pickle'
            filename = c.MODEL_FOLDER + model_file
            model_save = open(filename, 'wb')
            pickle.dump(models, model_save)
            model_save.close()
            print('%s models are saved in %s' % (model_type, filename))

            del df_train, df_weather, models
            gc.collect()

    if len(rmsle_total) > 0:
        rmsle_total = np.concatenate(rmsle_total, axis=0)
        df_rmsle = pd.DataFrame(rmsle_total.reshape(-1, 2), columns=['site', 'rmsle'])
        df_rmsle.to_csv(model_type + '_rmsle_total.csv', index=False)
        print(df_rmsle)
