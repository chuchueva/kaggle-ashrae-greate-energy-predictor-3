import utils_data as ud
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

site_id_list = [[0]]
meter_list = [[0], [1], [2], [3]]
is_train = True

meter_list_uni = ud.flat_list(meter_list)
site_id_list_uni = ud.flat_list(site_id_list)

_, df_consumption, df_building = ud.read_train_data(site_id_list_uni, meter_list_uni, train_flag=is_train)

# Plot train consumption by meter_building for cleaning purpose

for site_id in site_id_list_uni:

    building_list = df_building.loc[df_building['site_id'] == site_id, 'building_id'].values
    print('Building from %d to %d' % (min(building_list), max(building_list)))

    for building in building_list:
        for meter in meter_list_uni:

            df_sample_building = df_consumption.query('meter == @meter and building_id == @building')
            df_sample_building_clean = ud.manual_filtering(df_sample_building.copy())

            if len(df_sample_building) > 0:

                plt.plot(df_sample_building['timestamp'], df_sample_building['meter_reading'].values, '*',
                         label='actuals source')
                plt.plot(df_sample_building_clean['timestamp'], df_sample_building_clean['meter_reading'].values, '.',
                         label='actuals cleaned')
                plt.title('Consumption for site %d, meter %d, building %d' % (site_id, meter, building), fontsize=12)
                plt.legend()
                plt.show()

print('T')
