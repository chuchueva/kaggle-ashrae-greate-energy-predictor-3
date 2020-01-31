import gc as gc
import time as time
import random as random
import numpy as np
import pandas as pd
import utils as utils
import utils_clean as uc
import lightgbm as lgb
import pickle as pickle
import constants as c

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

random.seed(c.FAVOURITE_NUMBER)

building_file = 'building_metadata.csv'
weather_data = 'weather_cleaned_site_0.feather'
train_data = 'train_site_0_meter_0.feather'
test_data = 'test_site_0_meter_0.feather'
result_to_update_file = 'late_model_01.csv'
model_file = 'model_03.pickle'

'''

Read data

'''

building = pd.read_csv(c.SOURCE_FOLDER + building_file)
df_train_raw = pd.read_feather(c.SPLIT_FOLDER + train_data)
df_weather = pd.read_feather(c.CLEAN_FOLDER + weather_data)
df_train_raw = uc.manual_filtering_site_0(df_train_raw)
df_train_raw = utils.prepare_data_glb(df_train_raw, building, df_weather)

print('Training data is prepared, time %.0f sec' % (time.time() - start_time))

seasonality_models = dict()
building_list = np.unique(df_train_raw['building_id'])
meter_type_list = [0]

for building in building_list:

    for meter_type in meter_type_list:

        mask = (df_train_raw['meter'] == meter_type) & (df_train_raw['building_id'] == building)
        df_sample_building = df_train_raw[mask]

        if any(df_sample_building):

            model, pred = utils.get_seasonality_model(df_sample_building)
            rmse = utils.get_error(df_sample_building['meter_reading'].values, pred)
            seasonality_models['building_%d_meter_%d' % (building, meter_type)] = model

            # Plot
            # n, bins, patches = plt.hist(df_sample_building['meter_reading'].values, 50, density=True,
            # facecolor='g', alpha=0.75)
            # plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, label='actuals')
            # plt.plot(df_sample_building.index, pred, label='meter_modelling')
            # plt.title('Modelling for %d, meter %d, rmse %.2f' % (building, meter_type, rmse), fontsize=20)
            # plt.show()

            print('Building %s meter %s is done, rmse %.2f, time %.0f sec' %
                  (str(building), str(meter_type), rmse, (time.time() - start_time)))

            df_train_raw.loc[mask, 'seasonality_model'] = pred

        else:

            print('No meter %d data for building %s' % (meter_type, str(building)))

result = dict()
result['seasonality'] = dict()
result['seasonality']['models'] = seasonality_models

print('RMSLE simple: %.2f' % utils.get_error(df_train_raw['meter_reading'].values,
                                             df_train_raw['seasonality_model'].values))

df_train_raw['meter_reading_e1'] = df_train_raw['meter_reading'] - df_train_raw['seasonality_model']
df_train = utils.feature_engineering(df_train_raw)
features_list = df_train.columns[np.invert(df_train.columns.isin(['site_id', 'meter_reading', 'meter',
                                                                  'meter_reading_e1', 'seasonality_model']))]
target_name = 'meter_reading'

print(features_list)
print('Features are prepared, time %.0f sec' % (time.time() - start_time))

categorical_features = ['building_id', 'primary_use', 'hour', 'weekday', 'month', 'season']

cv = 3
models = {}
cv_scores = {'site_id': [], 'cv_score': []}

params = {'objective': 'regression',
          'num_leaves': 50,
          'learning_rate': 0.01,
          'num_boost_round': 1200,
          'metric': 'rmse'
          }

for site_id in range(1):

    kf = KFold(n_splits=cv, random_state=c.FAVOURITE_NUMBER)
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

        model_lgb = lgb.train(params, train_set=d_train, valid_sets=watchlist, verbose_eval=100)
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
df_train_raw['meter_model'] = (df_train_raw['seasonality_model'] + df_train_raw['residual_model']) / 2

print('RMSLE: %.2f' % utils.get_error(df_train_raw['meter_reading'].values, df_train_raw['meter_model'].values))

print(pd.DataFrame.from_dict(cv_scores))
print('Training is done!')

del df_train, X_train, X_valid, y_train, y_valid, d_train, d_valid, watchlist, y_pred_valid, y_pred_train_site
gc.collect()

# Saving models in pickle
result['lgb'] = dict()
result['lgb']['models'] = models
result['lgb']['cv'] = cv
result['lgb']['cv_scores'] = cv_scores
result['lgb']['feature_list'] = features_list

filename = c.MODEL_FOLDER + model_file
model_save = open(filename, 'wb')
pickle.dump(result, model_save)
model_save.close()
print('Model is saved in ' + filename)

'''

Predict

'''

df_test = pd.read_feather(c.SPLIT_FOLDER + test_data)
building = pd.read_csv(c.SOURCE_FOLDER + building_file)
df_test = utils.prepare_data_glb(df_test, building, df_weather)
df_test = utils.feature_engineering(df_test)

model_file = open(c.MODEL_FOLDER + model_file, 'rb')
model = pickle.load(model_file)

del df_weather, building
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
    df_test['meter_model'] = (df_test['seasonality_model'] + df_test['residual_model']) / 2

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

    df_test['meter_model'] = np.expm1(df_test['meter_model'])
    x = np.column_stack((row_id_site, y_pred_test_site))
    output_array = df_test[['row_id', 'meter_model']].values

    print('Result data for %s is prepared, time %.0f sec' % (site_id, time.time() - start_time))
    print('Size of output_array is %d' % output_array.shape[0])

df_output = pd.read_csv(result_to_update_file)
df_output.index = df_output['row_id'].values
df_output.drop(columns=['row_id'], inplace=True)
df_output.loc[output_array[:, 0], 'meter_reading'] = output_array[:, 1]
df_output.to_csv('late_model_03.csv', index=True, index_label='row_id', float_format='%.2f')
print('File is written, time %.0f sec' % (time.time() - start_time))
