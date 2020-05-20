import pandas as pd
import numpy as np

result_to_blend = ['late_model_69_1.csv', 'late_model_69_2.csv']
result_file = 'late_model_69.csv'
mr = []

for f in result_to_blend:
    df_output = pd.read_csv(f)
    mr.append(df_output['meter_reading'])

x = np.column_stack(mr)

df_output['meter_reading'] = np.nanmean(x, axis=1)
df_output.drop(columns=['row_id'], inplace=True)
df_output.to_csv(result_file, index=True, index_label='row_id', float_format='%.0f')
print('File %s is written' % result_file)
