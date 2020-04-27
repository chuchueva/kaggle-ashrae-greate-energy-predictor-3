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
site_id_list = np.arange(c.SITE_ID_RANGE)
meter_list = np.arange(c.METER_RANGE)
data_type = 'test'              # test
do_filter = False               # False (True only for train)


building = pd.read_csv(c.SOURCE_FOLDER + c.BUILDING_FILE, dtype=c.BUILDING_METADATA_DTYPES)

if data_type == 'train':
    df_train = pd.read_csv(c.SOURCE_FOLDER + c.TRAIN_FILE, dtype=c.TRAIN_DTYPES, parse_dates=['timestamp'])
else:
    df_train = pd.read_csv(c.SOURCE_FOLDER + c.TEST_FILE, dtype=c.TEST_DTYPES, parse_dates=['timestamp'])


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
