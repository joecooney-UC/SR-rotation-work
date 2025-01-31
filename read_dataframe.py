from fastparquet import ParquetFile
# from mi_utils_no_numba import *
import pandas as pd
import numpy as np

pf = ParquetFile('dna_mrna_path_features_and_label.parquet')
df = pf.to_pandas()

target = df.filter(items=['origin']).transpose().to_numpy()

cols_of_interest = df.columns[0:25]
rows = df.filter(items=cols_of_interest).transpose().to_numpy()

data = np.append(rows, target, axis=0)
print([i for i in cols_of_interest])
print(data)

#print(target)
#print(rows)

file = open("pancan_han_stats_0_25.txt", 'w')

file.write("Columns Used:\n")
cols_used = ""

for title in cols_of_interest:
    cols_used += str(title) + ','

file.write(cols_used)
