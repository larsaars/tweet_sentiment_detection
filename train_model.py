from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl
import numpy as np

from sklearn.linear_model import LogisticRegression

# read the csv with all data
df = pd.read_csv('datasets/train2.csv', sep=';')
# ensure text is string
df['text'] = df.text.apply(str)
df['target'] = df.target.apply(str)
# init list of stopwords
stop = list(stopwords.words('english'))
# split with hold - out validation - just to calc a score,
# after that the whole model will be recalculated
# train_split, validation_split = train_test_split(df, test_size=0.2, random_state=0, stratify=df.target.values)
# count-vectorize words in the given ngram_range
vectorizer = CountVectorizer(decode_error='replace', stop_words=stop, ngram_range=(1, 2))
# fit the vectorizer (etc. for later whole model training)
final_X_train = vectorizer.fit_transform(df.text.values.astype('U'))
final_y_train = df.target
# # for score calculation
# score_X_train = vectorizer.transform(train_split['text'].values.astype('U'))
# score_X_valid = vectorizer.transform(validation_split['text'].values.astype('U'))
# score_y_train = train_split.target.values
# score_y_valid = validation_split.target.values
# # calc test score
# # 1. create model
# model = LogisticRegression(max_iter=300)
# # 2. fit model
# model.fit(score_X_train, score_y_train)
# # 3. print score
# print('score: %d' % model.score(score_X_valid, score_y_valid))

# after predicting score train everything and save to binaries
model = LogisticRegression(max_iter=300)
model.fit(final_X_train, final_y_train)

# and dump into files
with open('models/vectorizer.pkl', 'wb') as f:
    pkl.dump(vectorizer, f)

with open('models/model.pkl', 'wb') as f:
    pkl.dump(model, f)
