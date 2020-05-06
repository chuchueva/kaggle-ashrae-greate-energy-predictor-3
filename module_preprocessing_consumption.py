import time as time
import numpy as np
import pandas as pd
import constants as c
import utils_data as ud

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

'''

Main

'''
site_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
meter_list = np.arange(c.METER_RANGE)
data_type = 'train'                 # test
do_filter = True                    # False (True only for train)
do_average = True                   # if we want to write csv with averages

building = pd.read_csv(c.SOURCE_FOLDER + c.BUILDING_FILE, dtype=c.BUILDING_METADATA_DTYPES)

if data_type == 'train':
    df_train = pd.read_csv(c.SOURCE_FOLDER + c.TRAIN_FILE, dtype=c.TRAIN_DTYPES, parse_dates=['timestamp'])
else:
    df_train = pd.read_csv(c.SOURCE_FOLDER + c.TEST_FILE, dtype=c.TEST_DTYPES, parse_dates=['timestamp'])

site_building_average = pd.DataFrame(np.zeros((c.BUILDING_RANGE, c.METER_RANGE)), columns=[np.arange(c.METER_RANGE)])
site_building_average.index = np.arange(c.BUILDING_RANGE)

for site_id in site_id_list:

    building_list = building['building_id'][building['site_id'] == site_id].values

    for meter in meter_list:

        df_site_meter = df_train.query('building_id in @building_list and meter == @meter')

        if len(df_site_meter) > 0:

            if 'index' in df_site_meter:
                df_site_meter.drop(columns=['index'], inplace=True)

            if do_filter:
                df_site_meter = ud.filter_by_settings(df_site_meter)

            if data_type == 'train':
                file_name = c.CLEAN_FOLDER + c.TRAIN_FILE_TEMPLATE % (site_id, meter)
            else:
                file_name = c.CLEAN_FOLDER + c.TEST_FILE_TEMPLATE % (site_id, meter)

            df_site_meter.reset_index(drop=True).to_feather(file_name)

            print('File %s is written' % file_name)

            # Calculate average
            if do_average:
                x = df_site_meter.groupby(['building_id']).mean()
                site_building_average.iloc[x.index, meter] = x['meter_reading'].values

if do_average:
    site_building_average.to_csv('building-meter-average.csv', index=True)
