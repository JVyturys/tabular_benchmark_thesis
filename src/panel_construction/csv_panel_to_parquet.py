######################################################################
'''

This file transforms the panel                               
into a compressed 'parquet' format for CPU friendly data handling. 

'''
######################################################################

import config as con
import pandas as pd

data_path = con.DATA_FINAL_PANEL
output_path = con.RESULTS_DIR

panel = pd.read_csv(data_path)

print(f"The data shape is\n {panel.shape}")
print(f"Variable-type distribution:\n {panel.dtypes.value_counts()}")
print(f"Total missing values in the data\n {panel.isnull().sum().sum()}")


panel.to_parquet('panel.parquet', index=False)