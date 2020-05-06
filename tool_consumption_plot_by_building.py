import utils_data as ud
import pandas as pd
import constants as c
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

site_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
meter_list = [0, 1, 2, 3]

df_building = ud.read_building_data()
df_weather = ud.read_weather_data(site_id_list)
df_consumption = ud.read_consumption_data(site_id_list, meter_list, data_type='train')

'''

 Plot train consumption by meter_building for cleaning purpose

'''

# for site_id in site_id_list:
#
#     building_list = df_building.loc[df_building['site_id'] == site_id, 'building_id'].values
#     # building_list = building_list[building_list >= 7]
#
#     print('Buildings from %d to %d' % (min(building_list), max(building_list)))
#
#     for meter in meter_list:
#         for building in building_list:
#
#             df_sample_building = df_consumption.query('meter == @meter and building_id == @building')
#             df_sample_building_clean = ud.filter_by_settings(df_sample_building.copy())
#
#             if len(df_sample_building) > 0:
#
#                 plt.plot(df_sample_building['timestamp'], df_sample_building['meter_reading'].values, '*',
#                          label='actuals source')
#                 plt.plot(df_sample_building_clean['timestamp'], df_sample_building_clean['meter_reading'].values, '.',
#                          label='actuals cleaned')
#                 plt.title('Consumption for site %d, meter %d, building %d' % (site_id, meter, building), fontsize=12)
#                 # plt.legend()
#                 plt.show()


'''

Plot train consumption by meter_building for checking cleaning settings

'''

filters_data = pd.read_csv(c.FILTER_SETTINGS_FILE)

building_list = filters_data['building_id']
meter_list = filters_data['meter']
site_id_list = filters_data['site_id']

for site_id, building, meter in zip(site_id_list, building_list, meter_list):

    df_sample_building = df_consumption.query('meter == @meter and building_id == @building')
    df_sample_building_clean = ud.filter_by_settings(df_sample_building.copy())

    if len(df_sample_building) > 0:

        plt.plot(df_sample_building['timestamp'], df_sample_building['meter_reading'].values, '*',
                 label='actuals source')
        plt.plot(df_sample_building_clean['timestamp'], df_sample_building_clean['meter_reading'].values, '.',
                 label='actuals cleaned')
        plt.title('Consumption for site %d, meter %d, building %d' % (site_id, meter, building), fontsize=12)
        # plt.legend()
        plt.show()

print('Done!')
