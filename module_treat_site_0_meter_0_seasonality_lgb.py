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
weather_data = 'weather_cleaned_site_0.feather'
train_data = 'train_manually_filtered.feather'
test_data = 'test_site_0_meter_0.feather'
model_file = 'model_02.pickle'
result_to_update_file = 'late_model_01.csv'

'''

Read data

'''

building = pd.read_csv(source_folder + building_file)
df_train_raw = pd.read_feather(clean_folder + train_data)
df_weather = pd.read_feather(clean_folder + weather_data)

print('Data id loaded')

# df_weather = utils.create_lags(df_weather)
# df_weather = utils.create_window_averages(df_weather, 24)

print('Weather averages are done, time %.0f sec' % (time.time() - start_time))

site_id = 0
meter_id = 0
df_train_raw = utils.prepare_data_glb(df_train_raw, building, df_weather, site_id, meter_id)

del df_weather
gc.collect()

print('Training data is prepared, time %.0f sec' % (time.time() - start_time))

# Pick up seasonality
seasonality_models = dict()
building_list = np.unique(df_train_raw['building_id'])
meter_type_list = np.unique(df_train_raw['meter'])

for building in building_list:

    for meter_type in meter_type_list:

        mask = (df_train_raw['meter'] == meter_type) & (df_train_raw['building_id'] == building)
        df_sample_building = df_train_raw[mask]

        if any(df_sample_building):

            model, pred = utils.get_seasonality_model(df_sample_building)
            rmse = utils.get_error(df_sample_building['meter_reading'], pred)
            seasonality_models['building_%d_meter_%d' % (building, meter_type)] = model

            # Plot
            # plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, label='actuals')
            # plt.plot(df_sample_building.index, pred, label='meter_modelling')
            # plt.title('Modelling for %d, meter %d, rmse %.2f' % (building, meter_type, rmse), fontsize=20)
            # plt.show()

            # Write to output

            print('Building %s meter %s is done, rmse %.2f, time %.0f sec' %
                  (str(building), str(meter_type), rmse, (time.time() - start_time)))

            df_train_raw.loc[mask, 'seasonality_model'] = pred

        else:

            print('No meter %d data for building %s' % (meter_type, str(building)))

result = dict()
result['seasonality'] = dict()
result['seasonality']['models'] = seasonality_models

df_train_raw['meter_reading_e1'] = df_train_raw['meter_reading'] - df_train_raw['seasonality_model']
df_train = utils.feature_engineering(df_train_raw)
features_list = df_train.columns[np.invert(df_train.columns.isin(['site_id', 'meter_reading', 'meter',
                                                                  'meter_reading_e1', 'seasonality_model']))]
target_name = 'meter_reading_e1'

print(features_list)
print('Features are prepared, time %.0f sec' % (time.time() - start_time))

categorical_features = ['building_id', 'primary_use', 'hour', 'weekday', 'month', 'season']

cv = 2
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

for site_id in range(1):

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
        y_pred_train = model_lgb.predict(X_train, num_iteration=model_lgb.best_iteration)
        y_pred_train_site[train_index] = y_pred_train

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
        df_train_raw['residual_model_%d' % fold] = y_pred_train_site
        score += rmse / cv

    cv_scores['site_id'].append(site_id)
    cv_scores['cv_score'].append(score)

    print('--------------------------------------------------------------------------')
    print('Training for site %d, fold %d is done, RMSE = %.2f, time %.0f sec'
          % (site_id, fold, np.sqrt(mean_squared_error(y_train_site, y_pred_train_site)), time.time() - start_time))

df_train_raw['residual_model'] = np.mean(df_train_raw[['residual_model_' + str(f) for f in range(cv)]], axis=1)

print(pd.DataFrame.from_dict(cv_scores))
print('Training is done!')

del df_train, X_train, X_valid, y_train, y_valid, d_train, d_valid, watchlist, \
    y_pred_valid, y_pred_train_site
gc.collect()

# Saving models in pickle
result['lgb'] = dict()
result['lgb']['models'] = models
result['lgb']['cv'] = cv
result['lgb']['cv_scores'] = cv_scores
result['lgb']['feature_list'] = features_list

filename = model_folder + model_file
model_save = open(filename, 'wb')
pickle.dump(result, model_save)
model_save.close()
print('Model is saved in ' + filename)

'''

Predict

'''

building = pd.read_csv(source_folder + building_file)
df_test = pd.read_feather(clean_folder + test_data)
# df_test = df_test[df_test['building_id'].isin(building_list)]
# df_test = df_test[df_test['meter'].isin(meter_type_list)]
# df_test.reset_index(drop=True).to_feather(clean_folder + 'test_site_0_meter_0.feather')

df_weather_test = pd.read_feather(clean_folder + weather_data)
# df_weather_test = utils.create_window_averages(df_weather_test, 24)
df_test = utils.prepare_data_glb(df_test, building, df_weather_test, site_id, meter_id)
df_test = utils.feature_engineering(df_test)

model_file = open(model_folder + model_file, 'rb')
model = pickle.load(model_file)

del df_weather_test, building
gc.collect()

print('Test data is prepared, time %.0f sec' % (time.time() - start_time))

for site_id in range(1):

    # Seasonality

    for b in building_list:

        mask = df_test['building_id'] == b
        _, pred = utils.get_seasonality_model(df_test[mask],
                                              model['seasonality']['models']['building_%d_meter_%d' % (b, 0)])
        df_test.loc[mask, 'seasonality_model'] = pred

    # Residuals

    X_test_site = df_test.loc[df_test['site_id'] == site_id, model['lgb']['feature_list']]
    y_pred_test_site = np.zeros(len(X_test_site))
    row_id_site = df_test.loc[df_test['site_id'] == site_id, 'row_id']

    for fold in range(model['lgb']['cv']):
        model_lgb = model['lgb']['models'][site_id][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / model['lgb']['cv']
        print('Pred fold %d for site %d is done, time %.0f sec' % (fold, site_id, time.time() - start_time))
        gc.collect()

    df_test['residual_model'] = y_pred_test_site
    df_test['meter_model'] = df_test['seasonality_model'] + df_test['residual_model']

    # col_names_train = ['building_id', 'seasonality_model', 'residual_model', 'meter_reading']
    # col_names_test = ['building_id', 'seasonality_model', 'residual_model', 'meter_model']
    # df_plot = pd.concat([df_train_raw[col_names_train], df_test[col_names_test]], axis=0)
    #
    # for building in building_list:
    #
    #     df_sample_building = df_plot[df_plot['building_id'] == building]
    #     plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, label='meter_reading')
    #     plt.plot(df_sample_building.index, df_sample_building['seasonality_model'].values, label='seasonality_model')
    #     plt.plot(df_sample_building.index, df_sample_building['meter_model'].values, label='meter_model')
    #     plt.title('Modelling for %d, meter %d, rmse %.2f' % (building, meter_type, rmse), fontsize=20)
    #     plt.show()

    if site_id == 0:
        df_test['meter_model'] = np.expm1(df_test['meter_model'])
        df_test['meter_model'] = df_test['meter_model'] * 3.4118

    x = np.column_stack((row_id_site, y_pred_test_site))
    output_array = df_test[['row_id', 'meter_model']].values

    print('Result data for %s is prepared, time %.0f sec' % (site_id, time.time() - start_time))

df_output = pd.read_csv(result_to_update_file)
df_output.index = df_output['row_id'].values
df_output.drop(columns=['row_id'], inplace=True)
df_output.loc[output_array[:, 0], 'meter_reading'] = output_array[:, 1]
df_output.to_csv('late_model_02.csv', index=True, index_label='row_id', float_format='%.2f')
print('File is written, time %.0f sec' % (time.time() - start_time))

