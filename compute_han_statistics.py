from fastparquet import ParquetFile
from mi_utils_no_numba import *
import pandas as pd
import numpy as np
from sys import argv


pf = ParquetFile('dna_mrna_path_features_and_label.parquet')
df = pf.to_pandas()


target = df.filter(items=['origin']).transpose().to_numpy()

# get the columns of interest and the target - FROM COMAND LINE INPUT

#cols_of_interest = df.columns[argv[1]:argv[2]]

cols_of_interest = df.columns[0:25]
just_the_vars = df.filter(items=cols_of_interest).transpose().to_numpy()

vars_and_target = np.append(just_the_vars, target, axis=0)

# get han stats

h2 = second_order_han(just_the_vars)
h2_cond = second_order_han_cond(vars_and_target)

h3 = third_order_han(just_the_vars)
h3_cond = third_order_han_cond(vars_and_target)

# write to and make the file
file = open("pancan_han_stats_0_25.txt", 'x')

file.write("Columns Used:\n")
cols_used = ""

for title in cols_of_interest:
    cols_used += str(title) + ','

file.write(cols_used + '\n')

file.write(str(h2) + '\n')
file.write(str(h2_cond) + '\n')
file.write(str(h3) + '\n')
file.write(str(h3_cond) + '\n')

file.close()
