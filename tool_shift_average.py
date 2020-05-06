import pandas as pd

result_to_update_file = 'late_model_29.csv'
result_new_file = 'late_model_39.csv'

df_output = pd.read_csv(result_to_update_file)
df_output.drop(columns=['row_id'], inplace=True)
df_output['meter_reading'] = df_output['meter_reading'] * 1.1
df_output.to_csv(result_new_file, index=True, index_label='row_id', float_format='%.2f')
print('File is written')
