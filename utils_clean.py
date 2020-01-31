from collections import defaultdict
import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def filter2(df_train, building_id, meter, min_length, plot=False, verbose=False):

    if verbose:
        print("building_id: {}, meter: {}".format(building_id, meter))

    temp_df = df_train[(df_train['building_id'] == building_id) & (df_train['meter'] == meter)]
    target = temp_df['meter_reading'].values

    splitted_target = np.split(target, np.where(target[1:] != target[:-1])[0] + 1)
    splitted_date = np.split(temp_df['timestamp'].values, np.where(target[1:] != target[:-1])[0] + 1)

    building_idx = []
    for i, x in enumerate(splitted_date):
        if len(x) > min_length:
            start = x[0]
            end = x[-1]
            value = splitted_target[i][0]
            idx = df_train.query(
                '(@start <= timestamp <= @end) and meter_reading == @value and meter == @meter and '
                'building_id == @building_id', engine='python').index.tolist()
            building_idx.extend(idx)
            if verbose:
                print('Length: {},\t{}  -  {},\tvalue: {}'.format(len(x), start, end, value))

    building_idx = pd.Int64Index(building_idx)
    if plot:
        fig, axes = plt.subplots(nrows=2, figsize=(16, 18), dpi=100)

        temp_df.set_index('timestamp')['meter_reading'].plot(ax=axes[0])
        temp_df.drop(building_idx, axis=0).set_index('timestamp')['meter_reading'].plot(ax=axes[1])

        axes[0].set_title(f'Building {building_id} raw meter readings')
        axes[1].set_title(f'Building {building_id} filtered meter readings')

        plt.show()

    return building_idx


def manual_filtering_site_0(df_train):

    df_train['IsFiltered'] = 0

    # meter 0
    print('[Site 0 - Electricity] Filtering zeros')
    df_train.loc[df_train.query('meter == 0 and timestamp < "2016-05-21 00:00:00"').index, 'IsFiltered'] = 1

    print('[Site 0 - Electricity] Filtering outliers')
    df_train.loc[df_train.query(
        'building_id == 0 and meter == 0 and (meter_reading > 400 or meter_reading < -400)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 18 and meter == 0 and meter_reading < 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 22 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 25 and meter == 0 and meter_reading <= 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 38 and meter == 0 and (meter_reading > 2000 or meter_reading < 0)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 41 and meter == 0 and (meter_reading > 2000 or meter_reading < 0)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 53 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 77 and meter == 0 and (meter_reading > 1000 or meter_reading < 0)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 78 and meter == 0 and (meter_reading > 20000 or meter_reading < 0)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 86 and meter == 0 and (meter_reading > 1000 or meter_reading < 0)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 101 and meter == 0 and meter_reading > 400').index, 'IsFiltered'] = 1

    # meter 1
    #
    # print('[Site 0 - Chilled Water] Filtering leading constant values')
    # site0_meter1_thresholds = {
    #     50: [7, 9, 43, 60, 75, 95, 97, 98]
    # }
    #
    # for threshold in site0_meter1_thresholds:
    #     for building_id in site0_meter1_thresholds[threshold]:
    #         filtered_idx = filter2(df_train, building_id, 1, threshold)
    #         df_train.loc[filtered_idx, 'IsFiltered'] = 1
    #
    # print('[Site 0 - Chilled Water] Filtering outliers')
    # df_train.loc[df_train.query('building_id == 60 and meter == 1 and meter_reading > 25000').index, 'IsFiltered'] = 1
    # df_train.loc[df_train.query('building_id == 103 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1

    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)
    df_train.drop(columns=['IsFiltered'], inplace=True)

    return df_train.reset_index(drop=True)
