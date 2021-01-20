from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl
import numpy as np

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('datasets/train.csv', sep=';')

stop = list(stopwords.words('english'))

vectorizer = CountVectorizer(decode_error='replace', stop_words=stop, ngram_range=(1, 2))

X_train = vectorizer.fit_transform(df['text'].values.astype('U'))
y_train = df.target.values

print("X_train.shape : ", X_train.shape)
print("y_train.shape : ", y_train.shape)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

with open('models/vectorizer.pkl', 'wb') as f:
    pkl.dump(vectorizer, f)

with open('models/model.pkl', 'wb') as f:
    pkl.dump(model, f)
