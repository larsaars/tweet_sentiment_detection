"""
fuse all datasets from "all" to one big train.csv
"""

import pandas as pd

df = pd.read_csv('datasets/train.csv', sep=';', )



df.to_csv('datasets/train.csv', sep=';', mode='w')