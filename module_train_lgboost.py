import gc as gc
import time as time
import random as random
import numpy as np
import pandas as pd
import utils as utils
import lightgbm as lgb
import pickle as pickle

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

my_favourite_number = 13
seed = my_favourite_number
random.seed(seed)

clean_folder = 'Cleaned/'
source_folder = 'Source/'
model_folder = 'Models/'
building_file = 'building_metadata.csv'
weather_data = 'weather_cleaned.feather'
train_data = 'train_manually_filtered.feather'
model_file = 'model_01.pickle'

'''

Read data

'''

building = pd.read_csv(source_folder + building_file)
df_train_raw = pd.read_feather(clean_folder + train_data)
df_weather = pd.read_feather(clean_folder + weather_data)

print('Data id loaded')

df_weather = utils.create_lags(df_weather)
df_weather = utils.create_window_averages(df_weather, 24)

print('Weather averages are done, time %.0f sec' % (time.time() - start_time))

df_train_raw = utils.prepare_data_glb(df_train_raw, building, df_weather)

del df_weather
gc.collect()

print('Training data is prepared, time %.0f sec' % (time.time() - start_time))

df_train = utils.feature_engineering(df_train_raw)
features_list = df_train.columns[np.invert(df_train.columns.isin(['site_id', 'meter_reading']))]
target_name = 'meter_reading'

print(features_list)
print('Features are prepared, time %.0f sec' % (time.time() - start_time))

categorical_features = ['building_id', 'meter', 'primary_use', 'hour', 'weekday', 'month']

cv = 3
models = {}
cv_scores = {'site_id': [], 'cv_score': []}
params = {'objective': 'regression',
          'num_leaves': 41,
          'learning_rate': 0.049,
          'bagging_freq': 5,
          'bagging_fraction': 0.51,
          'feature_fraction': 0.81,
          'metric': 'rmse'
          }

for site_id in range(16):

    kf = KFold(n_splits=cv, random_state=seed)
    models[site_id] = []

    X_train_site = df_train.loc[df_train['site_id'] == site_id, features_list].reset_index(drop=True)
    y_train_site = df_train.loc[df_train['site_id'] == site_id, target_name]
    y_pred_train_site = np.zeros(len(X_train_site))

    score = 0

    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):

        X_train, X_valid = X_train_site.loc[train_index], X_train_site.loc[valid_index]
        y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]

        d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)

        watchlist = [d_train, d_valid]

        model_lgb = lgb.train(params, train_set=d_train, num_boost_round=999, valid_sets=watchlist,
                              verbose_eval=101, early_stopping_rounds=21)
        models[site_id].append(model_lgb)

        y_pred_valid = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
        y_pred_train_site[valid_index] = y_pred_valid

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
        # print('Site Id:', site_id, ', Fold:', fold + 1, ', RMSE:', rmse)
        score += rmse / cv

    cv_scores['site_id'].append(site_id)
    cv_scores['cv_score'].append(score)

    print('--------------------------------------------------------------------------')
    print('Training for site %d, fold %d is done, RMSE = %.2f, time %.0f sec'
          % (site_id, fold, np.sqrt(mean_squared_error(y_train_site, y_pred_train_site)), time.time() - start_time))

print(pd.DataFrame.from_dict(cv_scores))
print('Training is done!')

del df_train, df_train_raw, X_train, X_valid, y_train, y_valid, d_train, d_valid, watchlist, \
    y_pred_valid, y_pred_train_site
gc.collect()

# Saving models in pickle

result = dict()
result['models'] = models
result['cv'] = cv
result['cv_scores'] = cv_scores
result['feature_list'] = features_list

filename = model_folder + model_file
model_save = open(filename, 'wb')
pickle.dump(result, model_save)
model_save.close()
print('Model is saved in ' + filename)
