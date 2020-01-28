from collections import defaultdict
import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

source_folder = 'Source/'
clean_folder = 'Cleaned/'
train_file = 'weather_train.csv'
test_file = 'weather_test.csv'
building_file = 'building_metadata.csv'


'''

Main

'''

# Read data

BUILDING_METADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.float32,
                            'year_built': np.float32, 'floor_count': np.float32, 'building_eui': np.float32}
building = pd.read_csv(source_folder + building_file, dtype=BUILDING_METADATA_DTYPES)

WEATHER_DTYPES = {'site_id': np.uint8, 'air_temperature': np.float32, 'cloud_coverage': np.float32,
                  'dew_temperature': np.float32, 'precip_depth_1_hr': np.float32, 'sea_level_pressure': np.float32,
                  'wind_direction': np.float32, 'wind_speed': np.float32}

df_weather_train = pd.read_csv(source_folder + train_file, dtype=WEATHER_DTYPES, parse_dates=['timestamp'])
df_weather_test = pd.read_csv(source_folder + test_file, dtype=WEATHER_DTYPES, parse_dates=['timestamp'])

df_weather = pd.concat([df_weather_train, df_weather_test], ignore_index=True)

# 1) Timestamp aligment
# from https://www.kaggle.com/gunesevitan/ashrae-eda-and-preprocessing

time_offset = {
    0: 5,    1: 0,    2: 9,    3: 6,    4: 8,    5: 0,    6: 6,    7: 6,    8: 5,    9: 7,    10: 8,    11: 6,
    12: 0,    13: 7,    14: 6,    15: 6
}

df_weather['time_offset'] = df_weather['site_id'].map(time_offset)
df_weather['timestamp_aligned'] = df_weather['timestamp'] - pd.to_timedelta(df_weather['time_offset'], unit='H')

# 2) Select features
# from https://www.kaggle.com/gunesevitan/ashrae-eda-and-preprocessing

df_weather['timestamp'] = df_weather['timestamp_aligned']
features_to_drop = ['precip_depth_1_hr', 'cloud_coverage', 'wind_direction', 'sea_level_pressure',
                    'timestamp_aligned', 'time_offset']
df_weather.drop(columns=features_to_drop, inplace=True)

# 3) Interpolate nans
# from https://www.kaggle.com/frednavruzov/nan-restoration-techniques-for-weather-data

cols_to_fill = ['air_temperature', 'dew_temperature', 'wind_speed']
x_total = pd.date_range(pd.datetime(2015, 12, 31, 0, 0), pd.datetime(2019, 1, 1, 1, 0), freq='H')

output_array = np.empty(0)
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
        int_func = interp1d(s.timestamp.values.astype(np.float64), s[col].values,
                            kind='linear', fill_value='extrapolate')
        s_values_new = int_func(x_total.copy().values.astype(np.float64))
        df_weather_cleaned[col] = s_values_new

    # df_weather_cleaned.to_csv(clean_folder + result_file_name + '_site_' + str(sid) + '.csv',
    #                           date_format='%Y%m%d %H:%M', float_format='%.2f', index=False)
    # print('File %s is written' % (clean_folder + result_file_name + '_site_' + str(sid) + '.csv'))

    file_name = '%s%s_site_%d.feather' % (clean_folder, result_file_name, sid)
    df_weather_cleaned.to_feather(file_name)
    print('File %s is written' % file_name)

    if len(output_array) == 0:
        output_array = df_weather_cleaned.values
    else:
        output_array = np.row_stack((output_array, df_weather_cleaned.values))

cols_name = list()
cols_name.append('timestamp')
cols_name.append('site_id')
for n in cols_to_fill:
    cols_name.append(n)

df_weather_updated = pd.DataFrame(output_array, columns=cols_name)

# Save file

result_file_name = "weather_cleaned"
df_weather_updated.to_csv(clean_folder + result_file_name + '.csv', date_format='%Y%m%d %H:%M',
                          float_format='%.2f', index=False)
print('File %s is written' % (clean_folder + result_file_name + '.csv'))

df_weather.to_feather(clean_folder + result_file_name + '.feather')
print('File %s is written' % (clean_folder + result_file_name + '.feather'))
