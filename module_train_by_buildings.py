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
# site_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
site_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
meter_list = [0, 1, 2, 3]
model_type = 'regress'          # prophet, regress

settings = us.get_regress_settings()
rmsle_total = list()

for site_id in site_id_list:

    df_building = ud.read_building_data()
    df_weather = ud.read_weather_data([site_id])
    df_train = ud.read_consumption_data([site_id], meter_list, data_type='train')
    df_train = ud.consumption_feature_engineering(df_train)
    df_train = ud.prepare_data(df_train, df_building, df_weather)

    building_list = np.unique(df_train['building_id'])
    df_train[model_type] = np.nan
    models = dict()
    building_regress = list()

    meter_done = []

    for meter in c.SITE_METER.get(site_id):

        for building in building_list:

            df_sample_building = df_train.query('meter == @meter and building_id == @building')
            mask = (df_train['meter'] == meter) & (df_train['building_id'] == building)

            if len(df_sample_building) > 0:

                if model_type == 'regress':
                    model, pred = um.get_regress(df_sample_building, settings)
                if model_type == 'prophet':
                    model, pred = um.get_prophet(df_sample_building, settings)

                rmsle = ud.get_error(df_sample_building['meter_reading'].values, pred)
                if rmsle < c.REGRESS_PROPHET_CV_EDGE:
                    name = ud.get_name(model_type, meter=meter, building=building)
                    models[name] = model
                    df_train.loc[mask, model_type] = pred
                    building_regress.append([meter, building])
                    rmsle_total.append(np.column_stack((site_id, meter, building, rmsle)))
                    if meter not in meter_done:
                        meter_done.append(meter)

                # Plot
                # plt.plot(df_sample_building.index, df_sample_building['meter_reading'].values, '*', label='actuals')
                # plt.plot(df_sample_building.index, pred, label=model_type)
                # plt.title('Modelling for %d, meter %d, rmse %s %.2f'
                #           % (building, meter, model_type, rmsle), fontsize=12)
                # plt.show()

                print('Model for building %d meter %d is done, rmsle %.2f, time %.0f sec' %
                      (building, meter, rmsle, time.time() - start_time))

            else:

                print('No meter %d data for building %s' % (meter, str(building)))

    # Save model

    models['site_id_list'] = site_id
    models['meter_list'] = meter_done
    models['building_list'] = np.array(building_regress)
    models['model_type'] = model_type

    model_file = ud.get_name(model_type, site_id, meter_done) + '.pickle'
    filename = c.MODEL_FOLDER + model_file
    model_save = open(filename, 'wb')
    pickle.dump(models, model_save)
    model_save.close()
    print('%s models are saved in %s' % (model_type, filename))

df_rmse_total = pd.DataFrame(np.concatenate(rmsle_total), columns=['site_id', 'meter', 'building', model_type])
df_rmse_sites = df_rmse_total.groupby(['site_id']).mean()
df_rmse_sites.to_csv('%s_rmsle_total_by_sites.csv' % model_type)
df_rmse_total.to_csv('%s_rmsle_total_by_buildings.csv' % model_type)
print(df_rmse_sites)
