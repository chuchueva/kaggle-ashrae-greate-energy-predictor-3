import pandas as pd
import numpy as np
import constants as c
import utils_model as um

result_to_blend = ['late_model_64a.csv', 'late_model_140.csv', 'late_model_137.csv']
result_file = 'late_model_159.csv'
mode = 'mean'        # mean, replace, smart

mr = []

for f in result_to_blend:
    df_output = pd.read_csv(c.MODEL_FOLDER + f)
    mr.append(df_output['meter_reading'])

x = np.column_stack(mr)
df_output.index = df_output['row_id'].values

if mode == 'replace':

    mask = np.isnan(x[:, 0])
    df_output.loc[mask, 'meter_reading'] = x[mask, 1]
    df_output.loc[np.invert(mask), 'meter_reading'] = x[np.invert(mask), 0]
    print('%.0f%% nans are replaced from %s' % (sum(mask)/len(df_output)*100, result_to_blend[1]))

elif mode == 'mean':

    df_output['meter_reading'] = np.nanmean(x, axis=1)

elif mode == 'smart':

    mr = []
    result_to_blend_train = [fn.replace('.csv', '_train.csv') for fn in result_to_blend]
    for f in result_to_blend_train:
        df_train = pd.read_csv(c.MODEL_FOLDER + f)
        mr.append(df_train['meter_reading'])
    x_train = np.column_stack(mr)
    df_actual = pd.read_csv(c.MODEL_FOLDER + 'late_model_actuals_train.csv')
    df_output['meter_reading'] = um.get_smart_blend(x, x_train, df_actual['meter_reading'].values)

df_output.drop(columns=['row_id'], inplace=True)
df_output.to_csv(result_file, index=True, index_label='row_id', float_format='%.2f')
print('File %s is written' % result_file)
