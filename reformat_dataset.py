import pandas as pd
from text_helper import *

# before reformatting the dataset, remove all empty lines
with open('datasets/train.csv', 'rw') as file:
    for line in file:
        if not line.isspace():
            file.write(line)

# now transform every line
df = pd.read_csv('datasets/train.csv', sep=';')
df['text'] = df['text'].apply(transform)
df.to_csv('datasets/train2.csv', sep=';', mode='w')
