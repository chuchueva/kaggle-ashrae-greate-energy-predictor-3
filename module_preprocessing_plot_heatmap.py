import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils as utils

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()

source_folder = 'Source/'
cleaned_folder = 'Cleaned/'
train_file = 'train_manually_filtered.csv'
building_file = 'building_metadata.csv'


'''

Main

'''

# Read data

building = pd.read_csv(source_folder + building_file)
building = utils.reduce_mem_usage(building, use_float16=True)

df_train = pd.read_csv(cleaned_folder + train_file)
df_train = utils.reduce_mem_usage(df_train, use_float16=True)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], infer_datetime_format=True)


def get_empty_map(x, y):

    missmap = np.empty((x, len(y)))
    missmap.fill(np.nan)

    return missmap


# Plot missing values per building/meter
time_range = pd.date_range(pd.datetime(2016, 1, 1, 0, 0), pd.datetime(2016, 12, 31, 23, 0), freq='H')
f, a = plt.subplots(1, 4, figsize=(20, 30))

for meter in np.arange(4):

    map_df = get_empty_map(1449, time_range)
    df = df_train[df_train['meter'] == meter].copy()

    for b in range(1449):
        v = df.query('building_id == @b', engine='python')[['timestamp', 'meter_reading']].reset_index()
        if any(v):
            v['meter_reading'] = v['meter_reading'].apply(lambda x: 0 if x == 0 else 1)
            mask = time_range.isin(v['timestamp'])
            if any(mask):
                map_df[b, mask] = v['meter_reading'].values

    a[meter].set_title(f'meter {meter:d}')
    sns.heatmap(map_df, cmap='Paired', ax=a[meter], cbar=False)

plt.show()