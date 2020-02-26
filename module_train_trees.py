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
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()

start_time = time.time()

random.seed(c.FAVOURITE_NUMBER)

model_file = 'model_trees.pickle'

'''

Read data

'''

site_id_list = [[0], [7]]
meter_list = [[0], [1], [2], [3]]

df_weather, df_train, df_building = ud.read_train_data(np.unique(site_id_list), np.unique(meter_list))
df_train = ud.prepare_data(df_train, df_building, df_weather)
df_train, categorical_features = ud.feature_engineering(df_train)

target_name = 'meter_reading'
models_trees = dict()

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

            X_train_site = df_train.loc[mask, features_list].reset_index(drop=True)
            y_train_site = df_train.loc[mask, target_name]
            name = ud.get_name('feature_list', site=site_id, meter=meter)
            models_trees[name] = features_list

            for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):

                X_train, X_valid = X_train_site.loc[train_index], X_train_site.loc[valid_index]
                y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]

                # LGBT (light gradient boost tree, and what are you thinking?)
                y_pred_train_site = np.zeros(len(X_train_site))
                d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
                d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
                watchlist = [d_train, d_valid]
                model_lgb = lgb.train(us.get_trees_settings('lgb_params'),
                                      train_set=d_train,
                                      valid_sets=watchlist,
                                      verbose_eval=100)
                name = ud.get_name('lgb', site=site_id, meter=meter, cv=str(fold))
                models_trees[name] = model_lgb
                y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
                y_pred_train_site[valid_index] = y_pred_valid
                y_pred_train = model_lgb.predict(X_train, num_iteration=model_lgb.best_iteration)
                y_pred_train_site[train_index] = y_pred_train
                df_train.loc[mask, 'lgb_model_%d' % fold] = y_pred_train_site

                del y_pred_train_site, y_pred_train, y_pred_valid, model_lgb, watchlist, d_train, d_valid
                gc.collect()

                # XGBoost
                y_pred_train_site = np.zeros(len(X_train_site))
                model_xgb = XGBRegressor(**us.get_trees_settings('xgb_params'))
                model_xgb.fit(X_train, y_train,
                              eval_set=[(X_train, y_train), (X_valid, y_valid)],
                              eval_metric='rmse',
                              verbose=True,
                              early_stopping_rounds=5)
                name = ud.get_name('xgb', site=site_id, meter=meter, cv=str(fold))
                models_trees[name] = model_xgb
                y_pred_valid = model_xgb.predict(X_valid, ntree_limit=model_xgb.best_ntree_limit)
                y_pred_train_site[valid_index] = y_pred_valid
                y_pred_train = model_xgb.predict(X_train, ntree_limit=model_xgb.best_ntree_limit)
                y_pred_train_site[train_index] = y_pred_train
                df_train.loc[mask, 'xgb_model_%d' % fold] = y_pred_train_site

                del y_pred_train_site, y_pred_train, y_pred_valid, model_xgb
                gc.collect()

                # CatBoost
                y_pred_train_site = np.zeros(len(X_train_site))
                categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
                cat_model = CatBoostRegressor(**us.get_trees_settings('cat_params'))
                cat_model.fit(X_train, y_train,
                              eval_set=(X_valid, y_valid),
                              cat_features=categorical_features_indices)
                name = ud.get_name('cat', site=site_id, meter=meter, cv=str(fold))
                models_trees[name] = cat_model
                y_pred_valid = cat_model.predict(X_valid)
                y_pred_train_site[valid_index] = y_pred_valid
                y_pred_train = cat_model.predict(X_train)
                y_pred_train_site[train_index] = y_pred_train
                df_train.loc[mask, 'cat_model_%d' % fold] = y_pred_train_site

                del y_pred_train_site, y_pred_train, y_pred_valid, cat_model
                gc.collect()

            del X_train, X_valid, y_train, y_valid
            gc.collect()

df_train['lgb_model'] = np.nanmean(df_train[['lgb_model_' + str(f) for f in range(us.get_trees_settings('cv'))]],
                                   axis=1)
df_train['xgb_model'] = np.nanmean(df_train[['xgb_model_' + str(f) for f in range(us.get_trees_settings('cv'))]],
                                   axis=1)
df_train['cat_model'] = np.nanmean(df_train[['cat_model_' + str(f) for f in range(us.get_trees_settings('cv'))]],
                                   axis=1)

mask = np.invert(np.isnan(df_train['lgb_model'].values))
print('RMSLE lgb: %.2f' % ud.get_error(df_train.loc[mask, 'meter_reading'].values,
                                       df_train.loc[mask, 'lgb_model'].values))
print('RMSLE xgb: %.2f' % ud.get_error(df_train.loc[mask, 'meter_reading'].values,
                                       df_train.loc[mask, 'xgb_model'].values))
print('RMSLE cat: %.2f' % ud.get_error(df_train.loc[mask, 'meter_reading'].values,
                                       df_train.loc[mask, 'cat_model'].values))

models_trees['site_id_list'] = site_id_list
models_trees['meter_list'] = meter_list
models_trees['type'] = 'tree'

filename = c.MODEL_FOLDER + model_file
model_save = open(filename, 'wb')
pickle.dump(models_trees, model_save)
model_save.close()
print('Trees models are saved in ' + filename)
