import utils_data as ud
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

site_id_list = [[7]]
meter_list = [[2]]
data_type = 'train'

meter_list_uni = ud.flat_list(meter_list)
site_id_list_uni = ud.flat_list(site_id_list)

df_building = ud.read_building_data()
df_weather = ud.read_weather_data(site_id_list_uni)
df_consumption = ud.read_consumption_data(site_id_list_uni, meter_list_uni, data_type=data_type)

# Plot train consumption by meter_building for cleaning purpose

for site_id in site_id_list_uni:

    building_list = df_building.loc[df_building['site_id'] == site_id, 'building_id'].values
    building_list = building_list[building_list >= 769]

    print('Building from %d to %d' % (min(building_list), max(building_list)))

    for meter in meter_list_uni:
        for building in building_list:

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
