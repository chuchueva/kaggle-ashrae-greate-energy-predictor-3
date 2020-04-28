import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants as c
import utils_data as ud
from pandas import datetime
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

start_time = time.time()


'''

Main

'''

site_id_list = [0]

# Read data
meter_list = []
for site_id in site_id_list:
    meter_list.append(c.SITE_METER.get(site_id))
meter_list = ud.flat_list(meter_list)

building = ud.read_building_data()
df_train = ud.read_consumption_data(site_id_list, meter_list, data_type='train')
building = building[building['site_id'].isin(site_id_list)]
building_list = building['building_id']


def get_empty_map(x, y):

    missmap = np.empty((x, len(y)))
    missmap.fill(np.nan)

    return missmap


# Plot missing values per building/meter
time_range = pd.date_range(datetime(2016, 1, 1, 0, 0), datetime(2016, 12, 31, 23, 0), freq='H')
f, a = plt.subplots(1, len(meter_list), figsize=(20, 30))

map_df_collection = dict()
for meter, i in zip(meter_list, range(len(meter_list))):

    map_df = get_empty_map(len(building_list), time_range)
    df = df_train[df_train['meter'] == meter].copy()

    for b in building_list:
        v = df.query('building_id == @b')[['timestamp', 'meter_reading']].reset_index()
        if any(v):
            mask_in = time_range.isin(v['timestamp'])
            mask_out = v['timestamp'].isin(time_range)
            if any(mask_in) and any(mask_out):
                map_df[b, mask_in] = v.loc[mask_out, 'meter_reading']

    map_df_collection['meter_%d' % meter] = map_df

    a[i].set_title('meter %d' % meter)
    sns.heatmap(map_df, cmap='tab20', ax=a[i], cbar=False)

plt.show()