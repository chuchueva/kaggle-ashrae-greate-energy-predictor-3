from collections import defaultdict
import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants as c

from scipy.interpolate import interp1d
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

source_folder = c.SOURCE_FOLDER
clean_folder = c.CLEAN_FOLDER
train_file = c.WEATHER_TRAIN_FILE
test_file = c.WEATHER_TEST_FILE
building_file = c.BUILDING_FILE

# What to drop
features_to_drop = ['timestamp_aligned', 'time_offset', 'cloud_coverage', 'precip_depth_1_hr',
                    'wind_direction', 'sea_level_pressure']

# What to extrapolate
cols_to_fill = ['air_temperature', 'dew_temperature', 'wind_speed']

'''

Main

'''

# Read data

building = pd.read_csv(source_folder + building_file, dtype=c.BUILDING_METADATA_DTYPES)
df_weather_train = pd.read_csv(source_folder + train_file, dtype=c.WEATHER_DTYPES, parse_dates=['timestamp'])
df_weather_test = pd.read_csv(source_folder + test_file, dtype=c.WEATHER_DTYPES, parse_dates=['timestamp'])

df_weather = pd.concat([df_weather_train, df_weather_test], ignore_index=True)

# 1) Timestamp aligment
# from https://www.kaggle.com/gunesevitan/ashrae-eda-and-preprocessing

time_offset = c.WEATHER_TIME_OFFSET
df_weather['time_offset'] = df_weather['site_id'].map(time_offset)
df_weather['timestamp_aligned'] = df_weather['timestamp'] - pd.to_timedelta(df_weather['time_offset'], unit='H')

# 2) Select features
# from https://www.kaggle.com/gunesevitan/ashrae-eda-and-preprocessing

df_weather['timestamp'] = df_weather['timestamp_aligned']
df_weather.drop(columns=features_to_drop, inplace=True)

# 3) Interpolate nans
# from https://www.kaggle.com/frednavruzov/nan-restoration-techniques-for-weather-data

x_total = pd.date_range(pd.datetime(2015, 12, 31, 0, 0), pd.datetime(2019, 1, 1, 1, 0), freq='H')

result_file_name = "weather_cleaned"

for sid in sorted(df_weather.site_id.unique()):

    print(f'\tfor site_id: "{sid}"')

    df_weather_cleaned = pd.DataFrame()
    df_weather_cleaned['timestamp'] = x_total
    df_weather_cleaned['site_id'] = sid

    for col in cols_to_fill:

        print(f'filling short NaN series in col "{col}"')

        s = df_weather.loc[df_weather.site_id == sid, [col, 'timestamp']].copy()
        s.dropna(inplace=True)
        try:
            int_func = interp1d(s.timestamp.values.astype(np.float64), s[col].values,
                                kind='linear', fill_value='extrapolate')
            s_values_new = int_func(x_total.copy().values.astype(np.float64))
            df_weather_cleaned[col] = s_values_new
        except ValueError:
            print('*** There is no %s data for site %d' % (col, sid))

    file_name = '%s%s_site_%d.feather' % (clean_folder, result_file_name, sid)
    df_weather_cleaned.to_feather(file_name)
    print('File %s is written' % file_name)
