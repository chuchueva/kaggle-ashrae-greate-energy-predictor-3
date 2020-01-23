import gc as gc
import time as time
import numpy as np
import pandas as pd
import utils as utils
import lightgbm as lgb
import pickle as pickle

from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

clean_folder = 'Cleaned/'
source_folder = 'Source/'
model_folder = 'Models/'
building_file = 'building_metadata.csv'
weather_data = 'weather_cleaned.feather'
test_data = 'test.csv'
model_file = 'model_01.pickle'

start_time = time.time()

''' 

Main 

'''

TEST_DTYPES = {'row_id': np.uint64, 'building_id': np.uint16, 'meter': np.uint8}
df_test = pd.read_csv(source_folder + test_data, dtype=TEST_DTYPES, parse_dates=['timestamp'])
df_weather = pd.read_feather(clean_folder + weather_data)
building = pd.read_csv(source_folder + building_file)

df_weather = utils.create_window_averages(df_weather, 24)
df_test = utils.prepare_data_glb(df_test, building, df_weather)
df_test = utils.feature_engineering(df_test)

model_file = open(model_folder + model_file, 'rb')
model = pickle.load(model_file)

del df_weather, building
gc.collect()

print('Test data is prepared, time %.0f sec' % (time.time() - start_time))

for site_id in range(16):

    X_test_site = df_test.loc[df_test['site_id'] == site_id, model['feature_list']]
    y_pred_test_site = np.zeros(len(X_test_site))
    row_id_site = df_test.loc[df_test['site_id'] == site_id, 'row_id']

    for fold in range(model['cv']):
        model_lgb = model['models'][site_id][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / model['cv']
        print('Pred fold %d for site %d is done, time %.0f sec' % (fold, site_id, time.time() - start_time))
        gc.collect()

    if site_id == 0:
        y_pred_test_site = np.expm1(y_pred_test_site)
        y_pred_test_site[X_test_site['meter'] == 0] = y_pred_test_site[X_test_site['meter'] == 0] * 3.4118
        x = np.column_stack((row_id_site, y_pred_test_site))
        output_array = x
    else:
        x = np.column_stack((row_id_site, np.expm1(y_pred_test_site)))
        output_array = np.row_stack((output_array, x))

    print('Result data for %s is prepared, time %.0f sec' % (site_id, time.time() - start_time))

df_output = pd.DataFrame()
df_output['meter_reading'] = np.zeros(len(df_test))
df_output.index = df_test['row_id'].values
df_output.loc[output_array[:, 0], 'meter_reading'] = output_array[:, 1]
df_output.to_csv('late_model_01.csv', index=True, index_label='row_id', float_format='%.2f')
print('File is written, time %.0f sec' % (time.time() - start_time))
