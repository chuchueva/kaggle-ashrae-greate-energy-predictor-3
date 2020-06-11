import pandas as pd
import numpy as np
import constants as c
import gc as gc
import time

start_time = time.time()

leak0 = pd.read_csv(c.SOURCE_FOLDER + 'Leak/site0.csv')
leak1 = pd.read_csv(c.SOURCE_FOLDER + 'Leak/site1.csv')
leak2 = pd.read_csv(c.SOURCE_FOLDER + 'Leak/site2.csv')
leak4 = pd.read_csv(c.SOURCE_FOLDER + 'Leak/site4.csv')
leak15 = pd.read_csv(c.SOURCE_FOLDER + 'Leak/site15.csv')
leak = pd.concat([leak0, leak1, leak2, leak4, leak15])

test = pd.read_csv(c.SOURCE_FOLDER + c.TEST_FILE)
test = test[test.building_id.isin(leak.building_id.unique())]
leak = leak.merge(test, on=["building_id", "meter", "timestamp"], how='left')

leak.rename({'meter_reading_scraped': 'meter_reading'}, axis=1, inplace=True)
leak_out = leak[['row_id', 'meter_reading']]
leak_out.dropna(inplace=True)
print(leak_out)

del test, leak0, leak1, leak2, leak4, leak15, leak
gc.collect()

result_to_update_file = 'late_model_142.csv'
result_new_file = 'late_model_158_leak.csv'

df_output = pd.read_csv(result_to_update_file)
df_output.index = df_output['row_id'].values
df_output.drop(columns=['row_id'], inplace=True)
print('File %s is read' % result_to_update_file)

df_output.loc[leak_out['row_id'].values, 'meter_reading'] = leak_out['meter_reading'].values
df_output.to_csv(result_new_file, index=True, index_label='row_id', float_format='%.2f')
print('File is written, time %.0f sec' % (time.time() - start_time))
