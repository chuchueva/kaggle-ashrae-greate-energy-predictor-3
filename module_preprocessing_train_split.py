from collections import defaultdict
import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Read data

BUILDING_METADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.float32,
                            'year_built': np.float32, 'floor_count': np.float32, 'building_eui': np.float32}
building = pd.read_csv(source_folder + building_file, dtype=BUILDING_METADATA_DTYPES)

TRAIN_DTYPES = {'building_id': np.uint16, 'meter': np.uint8, 'meter_reading': np.float32}
df_train = pd.read_csv(source_folder + train_file, dtype=TRAIN_DTYPES, parse_dates=['timestamp'])

# df_test = pd.read_feather(clean_folder + test_file)

# Save data by sites_meter in feather format

for site_id in range(16):

    building_list = building['building_id'][building['site_id'] == site_id].values

    for meter in range(4):

        df_site_meter = df_train.query('building_id in @building_list and meter == @meter', engine='python')
        if len(df_site_meter) > 0:
            file_name = '%strain_site_%d_meter_%d.feather' % (split_folder, site_id, meter)
            if 'index' in df_site_meter:
                df_site_meter.drop(columns=['index'], inplace=True)
            df_site_meter.reset_index(drop=True).to_feather(file_name)
            print('File %s is written' % file_name)
