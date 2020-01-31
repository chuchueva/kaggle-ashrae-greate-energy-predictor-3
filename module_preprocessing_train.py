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
train_file = 'train.csv'
building_file = 'building_metadata.csv'


'''

Main

'''

# Read data

BUILDING_METADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.float32,
                            'year_built': np.float32, 'floor_count': np.float32, 'building_eui': np.float32}
building = pd.read_csv(source_folder + building_file, dtype=BUILDING_METADATA_DTYPES)

TRAIN_DTYPES = {'building_id': np.uint16, 'meter': np.uint8, 'meter_reading': np.float32}
df_train = pd.read_csv(source_folder + train_file, dtype=TRAIN_DTYPES)
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], infer_datetime_format=True)


def filter2(building_id, meter, min_length, plot=False, verbose=False):

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


def manual_filtering(df_train):


    df_train['IsFiltered'] = 0

    #################### SITE 0 ####################

    ##### METER 0 #####

    print('[Site 0 - Electricity] Filtering leading constant values')

    leading_zeros = defaultdict(list)

    for building_id in range(105):
        leading_zeros_last_date = \
        df_train.query('building_id == @building_id and meter_reading == 0 and timestamp < "2016-09-01 01:00:00"',
                       engine='python')['timestamp'].max()
        leading_zeros[leading_zeros_last_date].append(building_id)

    for timestamp in leading_zeros.keys():
        building_ids = leading_zeros[pd.Timestamp(timestamp)]
        filtered_idx = df_train[df_train['building_id'].isin(building_ids)].query(
            'meter == 0 and timestamp <= @timestamp').index
        df_train.loc[filtered_idx, 'IsFiltered'] = 1

    print('[Site 0 - Electricity] Filtering outliers')
    df_train.loc[df_train.query(
        'building_id == 0 and meter == 0 and (meter_reading > 400 or meter_reading < -400)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 18 and meter == 0 and meter_reading < 1300').index, 'IsFiltered'] = 1
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

    ##### METER 1 #####

    print('[Site 0 - Chilled Water] Filtering leading constant values')
    site0_meter1_thresholds = {
        50: [7, 9, 43, 60, 75, 95, 97, 98]
    }

    for threshold in site0_meter1_thresholds:
        for building_id in site0_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    print('[Site 0 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 60 and meter == 1 and meter_reading > 25000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 103 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1


    #################### SITE 1 #####################

    ##### METER 0 #####

    print('[Site 1 - Electricity] Filtering leading constant values')
    site1_meter0_thresholds = {
        20: [106],
        50: [105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 119, 120, 127, 136, 137, 138, 139, 140, 141, 142, 143,
             144, 145, 147, 149, 152, 155]
    }

    for threshold in site1_meter0_thresholds:
        for building_id in site1_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 3 #####
    print('[Site 1 - Hot Water] Filtering leading constant values')
    site1_meter3_thresholds = {
        40: [106, 109, 112, 113, 114, 117, 119, 121, 138, 139, 144, 145]
    }

    for threshold in site1_meter3_thresholds:
        for building_id in site1_meter3_thresholds[threshold]:
            filtered_idx = filter2(building_id, 3, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    print('[Site 1 - Hot Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 119 and meter == 3 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 121 and meter == 3 and meter_reading > 20000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 144 and meter == 3 and meter_reading > 100').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 145 and meter == 3 and meter_reading > 2000').index, 'IsFiltered'] = 1

    #################### SITE 2 #####################

    ##### METER 0 #####
    print('[Site 2 - Electricity] Filtering leading constant values')
    site2_meter0_thresholds = {
        40: [278],
        100: [177, 258, 269],
        1000: [180]
    }

    for threshold in site2_meter0_thresholds:
        for building_id in site2_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 2 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 187 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 254 and meter == 1 and meter_reading > 1600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 257 and meter == 1 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 273 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 281 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

    print('[Site 2 - Chilled Water] Filtering leading constant values')
    site2_meter1_thresholds = {
        100: [207],
        1000: [260]
    }

    for threshold in site2_meter1_thresholds:
        for building_id in site2_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 3 #####################

    ##### METER 0 #####
    print('[Site 3 - Electricity] Filtering leading constant values')
    site3_meter0_thresholds = {
        100: [545]
    }

    for threshold in site3_meter0_thresholds:
        for building_id in site3_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 4 #####################

    ##### METER 0 #####
    print('[Site 4 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 592 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 609 and meter == 0 and meter_reading < 300').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 620 and meter == 0 and meter_reading < 750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 626 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 627 and meter == 0 and meter_reading < 85').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 632 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 645 and meter == 0 and meter_reading < 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 654 and meter == 0 and meter_reading < 40').index, 'IsFiltered'] = 1

    print('[Site 4 - Electricity] Filtering leading constant values')
    site4_meter0_thresholds = {
        100: [577, 604]
    }

    for threshold in site4_meter0_thresholds:
        for building_id in site4_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 5 #####################

    ##### METER 0 #####
    print('[Site 5 - Electricity] Filtering leading constant values')
    site5_meter0_thresholds = {
        100: [681, 723, 733, 739],
        1000: [693]
    }

    for threshold in site5_meter0_thresholds:
        for building_id in site5_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 6 #####################

    ##### METER 0 #####
    print('[Site 6 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 749 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 758 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 769 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 770 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 773 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 778 and meter == 0 and meter_reading < 50').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 781 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 785 and meter == 0 and meter_reading < 200').index, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 6 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 745 and meter == 1 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 750 and meter == 1 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 753 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 755 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 765 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 769 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 770 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 771 and meter == 1 and meter_reading > 20').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 776 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 777 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 780 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 786 and meter == 1 and meter_reading > 7000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 787 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

    print('[Site 6 - Chilled Water] Filtering leading constant values')
    site6_meter1_thresholds = {
        100: [748, 750, 752, 763, 767, 776, 786]
    }

    for threshold in site6_meter1_thresholds:
        for building_id in site6_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 6 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 762 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 764 and meter == 2 and meter_reading < 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 769 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 776 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 784 and meter == 2 and meter_reading < 2000').index, 'IsFiltered'] = 1

    print('[Site 6 - Steam] Filtering leading constant values')
    site6_meter2_thresholds = {
        150: [750, 751, 753, 770],
        500: [774]
    }

    for threshold in site6_meter2_thresholds:
        for building_id in site6_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 7 #####################

    ##### METER 0 #####
    print('[Site 7 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 800 and meter == 0 and meter_reading < 75').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 801 and meter == 0 and meter_reading < 3000').index, 'IsFiltered'] = 1

    print('[Site 7 - Electricity] Filtering leading constant values')
    site7_meter0_thresholds = {
        100: [799, 800, 802]
    }

    for threshold in site7_meter0_thresholds:
        for building_id in site7_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 7 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query(
        'building_id == 799 and meter == 1 and meter_reading > 4000 and timestamp > "2016-11-01"').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 800 and meter == 1 and meter_reading > 400 and timestamp > "2016-11-01"').index, 'IsFiltered'] = 1

    print('[Site 7 - Chilled Water] Filtering leading constant values')
    site7_meter1_thresholds = {
        50: [789, 790, 792]
    }

    for threshold in site7_meter1_thresholds:
        for building_id in site7_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 7 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 797 and meter == 2 and meter_reading > 8000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 800 and meter == 2 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 803 and meter == 2 and meter_reading == 0').index, 'IsFiltered'] = 1

    print('[Site 7 - Steam] Filtering leading constant values')
    site7_meter2_thresholds = {
        100: [800]
    }

    for threshold in site7_meter2_thresholds:
        for building_id in site7_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 8 #####################

    ##### METER 0 #####
    print('[Site 8 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 857 and meter == 0 and meter_reading > 10').index, 'IsFiltered'] = 1

    print('[Site 8 - Electricity] Filtering leading constant values')
    site8_meter0_thresholds = {
        1000: [815, 848]
    }

    for threshold in site8_meter0_thresholds:
        for building_id in site8_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 9 #####################

    ##### METER 0 #####
    print('[Site 9 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 886 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 904 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 921 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 927 and meter == 0 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 927 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 949 and meter == 0 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 954 and meter == 0 and meter_reading > 10000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 955 and meter == 0 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 956 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 961 and meter == 0 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 962 and meter == 0 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 966 and meter == 0 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 977 and meter == 0 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 983 and meter == 0 and meter_reading > 3500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 986 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 990 and meter == 0 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 993 and meter == 0 and meter_reading > 6000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 997 and meter == 0 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)

    print('[Site 9 - Electricity] Filtering leading constant values')
    site9_meter0_thresholds = {
        40: [897],
        50: [874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895,
             896, 898, 899,
             900, 901, 902, 903, 904, 905, 906, 907, 908, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922,
             923, 924, 925,
             926, 927, 928, 929, 930, 931, 932, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949,
             950, 951, 952,
             953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974,
             975, 976, 977,
             978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997],
        100: [909],
    }

    for threshold in site9_meter0_thresholds:
        for building_id in site9_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 9 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 879 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 880 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 885 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 891 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 900 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 903 and meter == 1 and meter_reading > 20000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 906 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 907 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 908 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 910 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 920 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 923 and meter == 1 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 925 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 926 and meter == 1 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 927 and meter == 1 and meter_reading > 20000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 929 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 931 and meter == 1 and meter_reading > 6000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 934 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 946 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 948 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 949 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 954 and meter == 1 and meter_reading > 50000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 955 and meter == 1 and meter_reading > 15000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 956 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 957 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 961 and meter == 1 and meter_reading > 8000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 963 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 964 and meter == 1 and meter_reading > 5000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 965 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 967 and meter == 1 and meter_reading > 1750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 969 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 973 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 976 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 978 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 983 and meter == 1 and meter_reading > 3800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 989 and meter == 1 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 990 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 993 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 994 and meter == 1 and meter_reading > 2500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 996 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)

    print('[Site 9 - Chilled Water] Filtering leading constant values')
    site9_meter1_thresholds = {
        50: [874, 875, 879, 880, 883, 885, 886, 887, 888, 889, 890, 891, 893, 894, 895, 896, 898, 899, 900, 901, 903, 905,
             906, 907, 908,
             910, 911, 912, 913, 914, 915, 916, 917, 918, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 932, 933,
             934, 935, 942,
             945, 946, 948, 949, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968,
             969, 971, 972,
             973, 974, 975, 976, 978, 979, 980, 981, 983, 987, 989, 990, 991, 992, 993, 994, 995, 996, 997]
    }

    for threshold in site9_meter1_thresholds:
        for building_id in site9_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 9 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 875 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 876 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 878 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 879 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 880 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 880 and meter == 2 and meter_reading > 600 and ("2016-06-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 885 and meter == 2 and meter_reading > 250').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 886 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 886 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 887 and meter == 2 and meter_reading > 325').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 888 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 889 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 890 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 894 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 895 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 896 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 896 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 898 and meter == 2 and meter_reading > 150').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 898 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 899 and meter == 2 and meter_reading > 800 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 900 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 901 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 903 and meter == 2 and meter_reading > 2400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 905 and meter == 2 and meter_reading > 140').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 906 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 907 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 907 and meter == 2 and meter_reading > 200 and ("2016-07-01" < timestamp < "2016-08-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 908 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 908 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 910 and meter == 2 and meter_reading > 300 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 911 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 912 and meter == 2 and meter_reading > 120').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 912 and meter == 2 and meter_reading > 90 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 913 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 914 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 915 and meter == 2 and meter_reading > 750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 915 and meter == 2 and meter_reading > 185 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 916 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 917 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 918 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 920 and meter == 2 and meter_reading > 490').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 921 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 922 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 922 and meter == 2 and meter_reading > 200 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 924 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 926 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 926 and meter == 2 and meter_reading > 70 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 927 and meter == 2 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 928 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 929 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 931 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 931 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-11-01")').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 932 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 933 and meter == 2 and meter_reading > 75').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 934 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 942 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 945 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 946 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 948 and meter == 2 and meter_reading > 120').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 949 and meter == 2 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 951 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 952 and meter == 2 and meter_reading > 1600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 953 and meter == 2 and meter_reading > 1250').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 954 and meter == 2 and meter_reading > 10000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 955 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 956 and meter == 2 and meter_reading > 350').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 957 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 957 and meter == 2 and meter_reading > 350 and ("2016-07-01" < timestamp < "2016-11-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 958 and meter == 2 and meter_reading > 1200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 959 and meter == 2 and meter_reading > 2500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 960 and meter == 2 and meter_reading > 350').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 960 and meter == 2 and meter_reading > 100 and ("2016-07-01" < timestamp < "2016-11-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 961 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 963 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 964 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 965 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 966 and meter == 2 and meter_reading > 1750').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 966 and meter == 2 and meter_reading > 500 and ("2016-07-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 967 and meter == 2 and meter_reading > 575').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 968 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 969 and meter == 2 and meter_reading > 700').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 969 and meter == 2 and meter_reading > 400 and ("2016-07-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 971 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 972 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 973 and meter == 2 and meter_reading > 450').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 974 and meter == 2 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 976 and meter == 2 and meter_reading > 60').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 978 and meter == 2 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 978 and meter == 2 and meter_reading > 1250 and ("2016-07-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 980 and meter == 2 and meter_reading > 150').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 981 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 983 and meter == 2 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 983 and meter == 2 and meter_reading > 1750 and ("2016-05-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 987 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 989 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 991 and meter == 2 and meter_reading > 400').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 991 and meter == 2 and meter_reading > 200 and ("2016-06-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 992 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 993 and meter == 2 and meter_reading > 6000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 994 and meter == 2 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 995 and meter == 2 and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 996 and meter == 2 and meter_reading > 260').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 997 and meter == 2 and meter_reading > 300').index, 'IsFiltered'] = 1
    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)

    print('[Site 9 - Steam] Filtering leading constant values')
    site9_meter2_thresholds = {
        40: [889, 910, 934, 955, 962, 974, 976],
        50: [874, 875, 876, 878, 879, 880, 885, 886, 887, 888, 890, 894, 895, 896, 898, 901, 903, 905, 906, 907, 908, 911,
             912, 913, 914, 915, 916, 917, 918, 920, 921,
             922, 924, 925, 926, 927, 928, 929, 931, 932, 933, 942, 945, 946, 948, 949, 951, 952, 953, 954, 956, 957, 958,
             959, 960, 961, 963, 964, 965, 966, 967, 968,
             969, 971, 972, 973, 978, 979, 980, 981, 983, 987, 989, 991, 992, 993, 994, 995, 996, 997]
    }

    for threshold in site9_meter2_thresholds:
        for building_id in site9_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 10 #####################

    ##### METER 0 #####
    print('[Site 10 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 998 and meter == 0 and meter_reading > 300').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1000 and meter == 0 and meter_reading > 300').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1006 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1008 and meter == 0 and (meter_reading > 250 or meter_reading < 5)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1019 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1025 and meter == 0 and meter_reading < 20').index, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 10 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1003 and meter == 1 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1017 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1

    ##### METER 3 #####
    print('[Site 10 - Hot Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1000 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1003 and meter == 3 and meter_reading > 40 and ("2016-06-01" < timestamp < "2016-08-01")').index,
                 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1012 and meter == 3 and meter_reading > 500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1017 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1018 and meter == 3 and meter_reading > 10000').index, 'IsFiltered'] = 1

    #################### SITE 12 #####################

    ##### METER 0 #####
    print('[Site 12 - Electricity] Filtering leading constant values')
    site12_meter0_thresholds = {
        50: [1066]
    }

    for threshold in site12_meter0_thresholds:
        for building_id in site12_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 13 #####################

    ##### METER 0 #####
    print('[Site 13 - Electricity] Filtering outliers')
    df_train.loc[df_train.query(
        'building_id == 1070 and meter == 0 and (meter_reading < 0 or meter_reading > 200)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1073 and meter == 0 and (meter_reading < 60 or meter_reading > 800)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1073 and meter == 0 and timestamp < "2016-08-01" and meter_reading > 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1082 and meter == 0 and (meter_reading < 5 or meter_reading > 200)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1088 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1098 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1100 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1119 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1128 and meter == 0 and (meter_reading < 25 or meter_reading > 175)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1153 and meter == 0 and (meter_reading < 90 or meter_reading > 250)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1157 and meter == 0 and (meter_reading < 110 or meter_reading > 200)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1162 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1163 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1165 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1167 and meter == 0 and (meter_reading == 0 or meter_reading > 250)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1168 and meter == 0 and meter_reading > 6000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1172 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1177 and meter == 0 and meter_reading < 15').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1185 and meter == 0 and (meter_reading > 300 or meter_reading < 10)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1202 and meter == 0 and (meter_reading < 50 or meter_reading > 300)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1203 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1206 and meter == 0 and meter_reading < 40').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1207 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1208 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1209 and meter == 0 and (meter_reading > 350 or meter_reading < 175)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1210 and meter == 0 and (meter_reading > 225 or meter_reading < 75)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1212 and meter == 0 and meter_reading < 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1219 and meter == 0 and (meter_reading < 35 or meter_reading > 300)').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1222 and meter == 0 and meter_reading < 100').index, 'IsFiltered'] = 1

    print('[Site 13 - Electricity] Filtering leading constant values')
    site13_meter0_thresholds = {
        40: [1079, 1096, 1113, 1154, 1160, 1169, 1170, 1189, 1221]
    }

    for threshold in site13_meter0_thresholds:
        for building_id in site13_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 13 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1088 and meter == 1 and meter_reading > 10000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1220 and meter == 1 and meter_reading > 4000').index, 'IsFiltered'] = 1

    print('[Site 13 - Chilled Water] Filtering leading constant values')
    site13_meter1_thresholds = {
        40: [1130, 1160]
    }

    for threshold in site13_meter1_thresholds:
        for building_id in site13_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 13 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1075 and meter == 2 and meter_reading > 3500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1099 and meter == 2 and meter_reading > 30000').index, 'IsFiltered'] = 1

    print('[Site 13 - Steam] Filtering leading constant values')
    site13_meter2_thresholds = {
        40: [1072, 1098, 1158],
        100: [1111],
        500: [1129, 1176, 1189]
    }

    for threshold in site13_meter2_thresholds:
        for building_id in site13_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 14 #####################

    ##### METER 0 #####
    print('[Site 14 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1252 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1258 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1263 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1284 and meter == 0 and meter_reading == 0').index, 'IsFiltered'] = 1

    print('[Site 14 - Electricity] Filtering leading constant values')
    site14_meter0_thresholds = {
        100: [1223, 1225, 1226, 1240, 1241, 1250, 1255, 1264, 1265, 1272, 1275, 1276, 1277, 1278, 1279, 1280, 1283,
              1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1302, 1303, 1317, 1322],
        300: [1319],
        500: [1233, 1234]
    }

    for threshold in site14_meter0_thresholds:
        for building_id in site14_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 14 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1236 and meter == 1 and meter_reading > 2000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1242 and meter == 1 and meter_reading > 800').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1276 and meter == 1 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1280 and meter == 1 and meter_reading > 120').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1311 and meter == 1 and meter_reading > 1100').index, 'IsFiltered'] = 1

    print('[Site 14 - Chilled Water] Filtering leading constant values')
    site14_meter1_thresholds = {
        50: [1239, 1245, 1247, 1248, 1254, 1287, 1295, 1307, 1308],
        100: [1225, 1226, 1227, 1230, 1232, 1233, 1234, 1237, 1240, 1246, 1260, 1263, 1264, 1272, 1276,
              1280, 1288, 1290, 1291, 1292, 1293, 1294, 1296, 1297, 1300, 1302, 1303, 1310, 1311, 1312,
              1317],
        500: [1223]
    }

    for threshold in site14_meter1_thresholds:
        for building_id in site14_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 14 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1249 and meter == 2 and meter_reading > 4000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1254 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1256 and meter == 2 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1263 and meter == 2 and meter_reading > 9000').index, 'IsFiltered'] = 1

    print('[Site 14 - Steam] Filtering leading constant values')
    site14_meter2_thresholds = {
        50: [1225, 1226, 1239, 1254, 1284, 1285, 1286, 1287, 1289, 1290,
             1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1301,
             1303, 1305, 1308, 1309, 1310],
        100: [1238, 1243, 1245, 1247, 1248, 1249, 1250, 1263, 1307]
    }

    for threshold in site14_meter2_thresholds:
        for building_id in site14_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 3 #####
    print('[Site 14 - Hot Water] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1231 and meter == 3 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1232 and meter == 3 and meter_reading > 4500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1236 and meter == 3 and meter_reading > 600').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1251 and meter == 3 and meter_reading > 3000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1268 and meter == 3 and meter_reading > 1500').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1274 and meter == 3 and meter_reading > 1000').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1298 and meter == 3 and meter_reading > 5000').index, 'IsFiltered'] = 1

    print('[Site 14 - Hot Water] Filtering leading constant values')
    site14_meter3_thresholds = {
        40: [1270, 1322, 1323],
        50: [1223, 1224, 1227, 1228, 1229, 1231, 1233, 1234, 1235, 1236, 1240, 1242, 1244, 1246, 1251, 1252, 1260, 1262,
             1265,
             1266, 1267, 1269, 1271, 1272, 1273, 1274, 1275, 1276, 1312, 1317, 1318, 1319, 1321],
        100: [1231, 1232, 1237]
    }

    for threshold in site14_meter3_thresholds:
        for building_id in site14_meter3_thresholds[threshold]:
            filtered_idx = filter2(building_id, 3, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    #################### SITE 15 #####################

    ##### METER 0 #####
    print('[Site 15 - Electricity] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1383 and meter == 0 and meter_reading < 60').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1401 and meter == 0 and meter_reading < 10').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1414 and meter == 0 and meter_reading < 30').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1423 and meter == 0 and meter_reading < 5').index, 'IsFiltered'] = 1

    print('[Site 15 - Electricity] Filtering leading constant values')
    site15_meter0_thresholds = {
        50: [1345, 1359, 1446]
    }

    for threshold in site15_meter0_thresholds:
        for building_id in site15_meter0_thresholds[threshold]:
            filtered_idx = filter2(building_id, 0, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 1 #####
    print('[Site 15 - Chilled Water] Filtering outliers')
    df_train.loc[df_train.query(
        'building_id == 1349 and meter == 1 and meter_reading > 1000 and timestamp < "2016-04-15"').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query(
        'building_id == 1382 and meter == 1 and meter_reading > 300 and timestamp < "2016-02-01"').index, 'IsFiltered'] = 1

    print('[Site 15 - Chilled Water] Filtering leading constant values')
    site15_meter1_thresholds = {
        50: [1363, 1410]
    }

    for threshold in site15_meter1_thresholds:
        for building_id in site15_meter1_thresholds[threshold]:
            filtered_idx = filter2(building_id, 1, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    ##### METER 2 #####
    print('[Site 15 - Steam] Filtering outliers')
    df_train.loc[df_train.query('building_id == 1355 and meter == 2 and meter_reading < 200').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1361 and meter == 2 and meter_reading < 203').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1392 and meter == 2 and meter_reading < 150').index, 'IsFiltered'] = 1
    df_train.loc[df_train.query('building_id == 1426 and meter == 2 and meter_reading > 1000').index, 'IsFiltered'] = 1

    print('[Site 15 - Steam] Filtering leading constant values')
    site15_meter2_thresholds = {
        20: [1425, 1427],
        40: [1329, 1337, 1338, 1341, 1342, 1344, 1347, 1350, 1351, 1354, 1360, 1364,
             1367, 1377, 1379, 1381, 1382, 1383, 1391, 1396, 1402, 1405, 1406, 1409,
             1414, 1418, 1424, 1430, 1433, 1437]
    }

    for threshold in site15_meter2_thresholds:
        for building_id in site15_meter2_thresholds[threshold]:
            filtered_idx = filter2(building_id, 2, threshold)
            df_train.loc[filtered_idx, 'IsFiltered'] = 1

    df_train.drop(df_train.query('IsFiltered == 1').index, inplace=True)
    df_train.drop(columns=['IsFiltered'], inplace=True)

    return df_train.reset_index()


def sigma_filter(df, tolerance=2, rolling_window=168):

    df['meter_reading_ln'] = np.log1p(df['meter_reading'])
    stats = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter'])\
        .rolling(rolling_window, min_periods=2, center=True).meter_reading_ln.agg(['median'])
    std = df.reset_index().set_index('timestamp').groupby(['building_id', 'meter']).meter_reading_ln.std()
    stats['max'] = np.expm1(stats['median'] + tolerance*std)
    stats['min'] = np.expm1(stats['median'] - tolerance*std)
    df = df.merge(stats[['min', 'max']], left_on=['building_id', 'meter', 'timestamp'], right_index=True)
    out_df = df[(df['meter_reading'] <= df['max']) & (df['meter_reading'] >= df['min'])]
    out_df.drop(columns=['min', 'max', 'meter_reading_ln'], inplace=True)

    return out_df.reset_index()


# Main

df_train_clean = manual_filtering(df_train)
# df_train_clean = sigma_filter(df_train)

# Save file
result_file_name = "train_manually_filtered"

print(df_train_clean.head())
df_train_clean.info(verbose=True)

if 'index' in df_train_clean:
    df_train_clean.drop(columns=['index'], inplace=True)



df_train_clean.to_csv(clean_folder + result_file_name + '.csv', date_format='%Y%m%d %H:%M',
                      float_format='%.2f', index=False)
print('File %s is written' % (clean_folder + result_file_name + '.csv'))

df_train_clean.to_feather(clean_folder + result_file_name + '.feather')
print('File %s is written' % (clean_folder + result_file_name + '.feather'))



# further read train_df = pd.read_feather('train.feather')
# https://www.kaggle.com/ganfear/missing-data-and-zeros-visualized