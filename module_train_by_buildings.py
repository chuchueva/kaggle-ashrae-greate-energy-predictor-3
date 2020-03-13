import time as time
import random as random
import numpy as np
import pandas as pd
import utils_data as ud
import utils_settings as us
import utils_model as um
import pickle as pickle
import constants as c
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

import warnings
warnings.filterwarnings("ignore")
register_matplotlib_converters()

start_time = time.time()
random.seed(c.FAVOURITE_NUMBER)

'''

Main

'''

site_id_list = [[2]]
meter_list = [[0], [1], [2], [3]]
model_type = 'prophet'          # prophet, regress

meter_list_uni = ud.flat_list(meter_list)
site_id_list_uni = ud.flat_list(site_id_list)

df_weather, df_train, df_building = ud.read_train_data(site_id_list_uni, meter_list_uni,
                                                       train_flag=True, folder=c.CLEAN_FOLDER)
df_train = ud.prepare_data(df_train, df_building, df_weather, make_log=True)

settings = us.get_regress_settings()
building_list = np.unique(df_train['building_id'])
df_train[model_type] = np.nan
models = dict()
rmsle_total = list()
building_regress = list()

for meter in meter_list:

    for building in building_list:

        df_sample_building = df_train.query('meter == @meter and building_id == @building')
        mask = (df_train['meter'] == meter[0]) & (df_train['building_id'] == building)

        if len(df_sample_building) > 0:

            if model_type == 'regress':
                model, pred = um.get_regress(df_sample_building, settings)
            if model_type == 'prophet':
                model, pred = um.get_prophet(df_sample_building, settings)

            rmsle = ud.get_error(df_sample_building['meter_reading'].values, pred)
            rmsle_total.append(np.column_stack((meter[0], building, rmsle)))

            if rmsle < c.REGRESS_PROPHET_CV_EDGE:
                name = ud.get_name(model_type, meter=meter[0], building=building)
                models[name] = model
                df_train.loc[mask, model_type] = pred
                building_regress.append([meter[0], building])

            # Plot
            # plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, '*', label='actuals')
            # plt.plot(df_sample_building.index, pred, label=model_type)
            # plt.title('Modelling for %d, meter %d, rmse %s %.2f'
            #           % (building, meter[0], model_type, rmsle), fontsize=12)
            # plt.show()

            print('Model for building %d meter %d is done, rmsle %.2f, time %.0f sec' %
                  (building, meter[0], rmsle, time.time() - start_time))

        else:

            print('No meter %d data for building %s' % (meter[0], str(building)))

# Save model

models['site_id_list'] = site_id_list
models['meter_list'] = meter_list
models['building_list'] = np.array(building_regress)
models['model_type'] = model_type

model_file = ud.get_name(model_type, site_id_list, meter_list) + '.pickle'
filename = c.MODEL_FOLDER + model_file
model_save = open(filename, 'wb')
pickle.dump(models, model_save)
model_save.close()
print('%s models are saved in %s' %(model_type, filename))

df_rmse_total = pd.DataFrame(np.concatenate(rmsle_total), columns=['meter', 'building', model_type])
df_rmse_total.to_csv('%s_rmsle.csv' % model_file[:-7])

mask = np.invert(np.isnan(df_train[model_type]))

if any(mask):
    print('RMSLE regress: %.2f' % ud.get_error(df_train.loc[mask, 'meter_reading'].values,
                                               df_train.loc[mask, model_type].values))
