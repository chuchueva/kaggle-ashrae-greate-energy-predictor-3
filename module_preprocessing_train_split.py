import time as time
import numpy as np
import pandas as pd
import constants as c
import utils_data as ud

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

source_folder = 'Source/'
clean_folder = 'Cleaned/'
split_folder = 'Splited/'
train_file = 'train.csv'
building_file = 'building_metadata.csv'
test_file = 'test.feather'

'''

Main

'''

data_type = 'train'          # test
mode = 'cleaned'              # cleaned

if mode == 'source':

    building = pd.read_csv(c.SOURCE_FOLDER + building_file, dtype=c.BUILDING_METADATA_DTYPES)
    if mode == 'train':
        df_train = pd.read_csv(c.SOURCE_FOLDER + train_file, dtype=c.TRAIN_DTYPES, parse_dates=['timestamp'])
    else:
        df_train = pd.read_csv(c.SOURCE_FOLDER + test_file, dtype=c.TEST_DTYPES, parse_dates=['timestamp'])

else:

    site_id_list = [np.arange(c.SITE_ID_RANGE)]
    meter_list = [np.arange(c.METER_RANGE)]
    meter_list_uni = ud.flat_list(meter_list)
    site_id_list_uni = ud.flat_list(site_id_list)
    _, df_train, building = ud.read_train_data(site_id_list_uni, meter_list_uni, train_flag=True)


for site_id in range(c.SITE_ID_RANGE):

    building_list = building['building_id'][building['site_id'] == site_id].values

    for meter in range(c.METER_RANGE):

        df_site_meter = df_train.query('building_id in @building_list and meter == @meter')

        if len(df_site_meter) > 0:

            if data_type == 'train':
                if mode == 'cleaned':
                    file_name = c.CLEAN_FOLDER + c.TRAIN_FILE_TEMPLATE % (site_id, meter)
                else:
                    file_name = c.SPLIT_FOLDER + c.TRAIN_FILE_TEMPLATE % (site_id, meter)
            else:
                file_name = c.SPLIT_FOLDER + c.TEST_FILE_TEMPLATE % (site_id, meter)

            if 'index' in df_site_meter:
                df_site_meter.drop(columns=['index'], inplace=True)

            df_site_meter.reset_index(drop=True).to_feather(file_name)

            print('File %s is written' % file_name)
