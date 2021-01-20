import pandas as pd
from text_helper import *

df = pd.read_csv('datasets/train.csv', sep=';')

print('start')
df['text'] = df['text'].apply(transform)

df.to_csv('datasets/train2.csv', sep=';', mode='w')
