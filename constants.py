
PRIMARY_USE_MAP = {'Education': 1, 'Office': 2, 'Entertainment/public assembly': 3, 'Lodging/residential': 4,
                   'Public services': 5, 'Healthcare': 6, 'Other': 7, 'Parking': 8, 'Manufacturing/industrial': 9,
                   'Food sales and service': 10, 'Retail': 11, 'Warehouse/storage': 12, 'Services': 13,
                   'Technology/science': 14, 'Utility': 15, 'Religious worship': 16}

CLEAN_FOLDER = 'Cleaned/'
SOURCE_FOLDER = 'Source/'
MODEL_FOLDER = 'Models/'
SPLIT_FOLDER = 'Splited/'

FAVOURITE_NUMBER = 145

BUILDING_FILE = 'building_metadata.csv'
WEATHER_FILE_TEMPLATE = 'weather_cleaned_site_%d.feather'
TRAIN_FILE_TEMPLATE = 'train_site_%d_meter_%d.feather'
TEST_FILE_TEMPLATE = 'test_site_%d_meter_%d.feather'
FILTER_FILE = 'ASHRAE-clean-settings.csv'

MODEL_NAME_TEMPLATE = '%s_site_%s_meter_%s_building_%s_cv_%s'

MODEL_TYPE_REGRESS = 'regress'
MODEL_TYPE_PROPHET = 'prophet'

REGRESS_CV_EDGE = 0.5