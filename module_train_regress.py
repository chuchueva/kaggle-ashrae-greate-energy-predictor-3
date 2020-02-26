import gc as gc
import time as time
import random as random
import numpy as np
import pandas as pd
import utils_data as ud
import utils_settings as us
import pickle as pickle
import constants as c
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()

start_time = time.time()

random.seed(c.FAVOURITE_NUMBER)

model_file = 'model_regress.pickle'

'''

Read data

'''

site_id_list = [[0], [7]]
meter_list = [[0], [1], [2], [3]]

meter_list_uni = ud.flat_list(meter_list)
site_id_list_uni = ud.flat_list(site_id_list)

df_weather, df_train, df_building = ud.read_train_data(site_id_list_uni, meter_list_uni, train_flag=True)
df_train = ud.prepare_data(df_train, df_building, df_weather, make_log=True)

# print('Training data is prepared, time %.0f sec' % (time.time() - start_time))

settings = us.get_regress_settings()
building_list = np.unique(df_train['building_id'])
df_train[c.MODEL_TYPE_REGRESS] = np.nan
df_train[c.MODEL_TYPE_PROPHET] = np.nan
models_seasonality = dict()
rmse_total = list()
building_regress = list()
building_prophet = list()

for meter in meter_list:

    for building in building_list:

        df_sample_building = df_train.query('meter == @meter and building_id == @building')
        mask = (df_train['meter'] == meter[0]) & (df_train['building_id'] == building)

        if len(df_sample_building) > 0:

            model_regress, model_prophet, pred_r, pred_p = ud.get_seasonality_model(df_sample_building, settings)
            rmse_r = ud.get_error(df_sample_building['meter_reading'].values, pred_r)
            rmse_p = ud.get_error(df_sample_building['meter_reading'].values, pred_p)
            rmse_total.append(np.column_stack((building, rmse_r, rmse_p)))

            if rmse_r < c.REGRESS_CV_EDGE:
                name = ud.get_name(c.MODEL_TYPE_REGRESS, meter=meter[0], building=building)
                models_seasonality[name] = model_regress
                df_train.loc[mask, c.MODEL_TYPE_REGRESS] = pred_r
                building_regress.append([meter[0], building])

            if rmse_r < c.REGRESS_CV_EDGE:
                name = ud.get_name(c.MODEL_TYPE_PROPHET, meter=meter[0], building=building)
                models_seasonality[name] = model_prophet
                df_train.loc[mask, c.MODEL_TYPE_PROPHET] = pred_p
                building_prophet.append([meter[0], building])

            # Plot
            # plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, '*', label='actuals')
            # plt.plot(df_sample_building.index, pred_r, label='regress')
            # plt.plot(df_sample_building.index, pred_p, label='prophet')
            # plt.title('Modelling for %d, meter %d, rmse regress %.2f, prophet %.2f'
            #           % (building, meter_type, rmse_r, rmse_p), fontsize=20)
            # plt.show()

        else:

            print('No meter %d data for building %s' % (meter[0], str(building)))

# df_rmse_total = pd.DataFrame(np.concatenate(rmse_total), columns=['building', 'regress', 'prophet'])
# df_rmse_total.to_csv('regress_and_prophet_rmsle.csv')

mask_regress = np.invert(np.isnan(df_train[c.MODEL_TYPE_REGRESS]))
mask_prophet = np.invert(np.isnan(df_train[c.MODEL_TYPE_PROPHET]))

print('RMSLE regress: %.2f' % ud.get_error(df_train.loc[mask_regress, 'meter_reading'].values,
                                           df_train.loc[mask_regress, c.MODEL_TYPE_REGRESS].values))

print('RMSLE prophet: %.2f' % ud.get_error(df_train.loc[mask_prophet, 'meter_reading'].values,
                                           df_train.loc[mask_prophet, c.MODEL_TYPE_PROPHET].values))

models_seasonality['site_id_list'] = site_id_list
models_seasonality['meter_list'] = meter_list
models_seasonality['building_list'] = {'regress': np.array(building_regress), 'prophet': np.array(building_prophet)}
models_seasonality['type'] = 'seasonality'

filename = c.MODEL_FOLDER + model_file
model_save = open(filename, 'wb')
pickle.dump(models_seasonality, model_save)
model_save.close()
print('Regress models are saved in ' + filename)
