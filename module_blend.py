import pandas as pd
import numpy as np

result_to_blend = ['late_model_108.csv', 'late_model_64.csv', 'late_model_71.csv']
result_file = 'late_model_112.csv'
mode = 'mean'        # mean, replace

mr = []

for f in result_to_blend:
    df_output = pd.read_csv(f)
    mr.append(df_output['meter_reading'])

x = np.column_stack(mr)

if mode == 'replace':
    mask = np.isnan(x[:, 0])
    df_output.loc[mask, 'meter_reading'] = x[mask, 1]
    df_output.loc[np.invert(mask), 'meter_reading'] = x[np.invert(mask), 0]
    print('%.0f%% nans are replaced from %s' % (sum(mask)/len(df_output)*100, result_to_blend[1]))
else:
    df_output['meter_reading'] = np.nanmean(x, axis=1)

df_output.drop(columns=['row_id'], inplace=True)
df_output.to_csv(result_file, index=True, index_label='row_id', float_format='%.0f')
print('File %s is written' % result_file)
