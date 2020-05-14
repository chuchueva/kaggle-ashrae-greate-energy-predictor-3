import numpy as np

storage = 'E:/PythonStorage/04_ashrae_building_modelling/'
CLEAN_FOLDER = storage + 'Cleaned/'
SOURCE_FOLDER = storage + 'Source/'
MODEL_FOLDER = storage + 'Models/'
SPLIT_FOLDER = storage + 'Splited/'

TEST_FILE = 'test.csv'
TRAIN_FILE = 'train.csv'
WEATHER_TRAIN_FILE = 'weather_train.csv'
WEATHER_TEST_FILE = 'weather_test.csv'
BUILDING_FILE = 'building_metadata.csv'
FILTER_SETTINGS_FILE = 'clean-settings-optimal.csv'

FAVOURITE_NUMBER = 145

WEATHER_FILE_TEMPLATE = 'weather_cleaned_site_%d.feather'
TRAIN_FILE_TEMPLATE = 'train_site_%d_meter_%d.feather'
TEST_FILE_TEMPLATE = 'test_site_%d_meter_%d.feather'
MODEL_NAME_TEMPLATE = '%s_site_%s_meter_%s_building_%s_cv_%s'

REGRESS_PROPHET_CV_EDGE = 0.5
SITE_ID_RANGE = 16
METER_RANGE = 4
BUILDING_RANGE = 1450

BUILDING_METADATA_DTYPES = {'site_id': np.uint8, 'building_id': np.uint16, 'square_feet': np.float32,
                            'year_built': np.float32, 'floor_count': np.float32, 'building_eui': np.float32}

TRAIN_DTYPES = {'building_id': np.uint16, 'meter': np.uint8, 'meter_reading': np.float32}

TEST_DTYPES = {'row_id': np.uint32, 'building_id': np.uint16, 'meter': np.uint8}

WEATHER_DTYPES = {'site_id': np.uint8, 'air_temperature': np.float32, 'cloud_coverage': np.float32,
                  'dew_temperature': np.float32, 'precip_depth_1_hr': np.float32, 'sea_level_pressure': np.float32,
                  'wind_direction': np.float32, 'wind_speed': np.float32}

PRIMARY_USE_MAP = {'Education': 1, 'Office': 2, 'Entertainment/public assembly': 3, 'Lodging/residential': 4,
                   'Public services': 5, 'Healthcare': 6, 'Other': 7, 'Parking': 8, 'Manufacturing/industrial': 9,
                   'Food sales and service': 10, 'Retail': 11, 'Warehouse/storage': 12, 'Services': 13,
                   'Technology/science': 14, 'Utility': 15, 'Religious worship': 16}

SITE_METER = {
    0: [0, 1],
    1: [0, 3],
    2: [0, 1, 3],
    3: [0],
    4: [0],
    5: [0],
    6: [0, 1, 2],
    7: [0, 1, 2, 3],
    8: [0],
    9: [0, 1, 2],
    10: [0, 1, 3],
    11: [0, 1, 3],
    12: [0],
    13: [0, 1, 2],
    14: [0, 1, 2, 3],
    15: [0, 1, 2, 3]
}

WEATHER_TIME_OFFSET = {
    0: 5,    1: 0,    2: 9,    3: 6,    4: 8,    5: 0,    6: 6,    7: 6,    8: 5,    9: 7,    10: 8,    11: 6,
    12: 0,    13: 7,    14: 6,    15: 6
}

SITE_COUNTRIES = {
    'England': [1, 5],
    'Ireland': [12],
    'Canada': [7, 11],
    'United_States': [0, 2, 3, 4, 6, 8, 9, 10, 13, 14, 15]
}
