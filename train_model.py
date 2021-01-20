from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import pickle as pkl
import numpy as np

from sklearn.linear_model import LogisticRegression

# read the csv with all data
df = pd.read_csv('datasets/train.csv', sep=';')
# init list of stopwords
stop = list(stopwords.words('english'))
# count-vectorize words in the given ngram_range
vectorizer = CountVectorizer(decode_error='replace', stop_words=stop, ngram_range=(1, 2))
# fit the strings to get target and features
X_train = vectorizer.fit_transform(df.iloc[:, 1].values.astype('U'))
y_train = df.iloc[:, 0]
#index of the loop and list of scores to calc mean
scores = []
idx = 0
# init a 5 split k fold for cross validation
kfold = KFold(n_splits=5, shuffle=True)
for train_df_idx, test_df_idx in kfold.split(df):
    # create dataframes from split indices
    train_df = df.iloc[train_df_idx]
    test_df = df.iloc[test_df_idx]

    # gather target and training data from data
    training_data = train_df.iloc[:, 1].values
    training_target = train_df.iloc[:, 0].values

    # same for for testing split set
    testing_data = train_df.iloc[:, 1].values
    testing_target = train_df.iloc[:, 0].values

    # calc for each split a score by:
    # 1. create model
    model = LogisticRegression(max_iter=300)
    # 2. fit model
    model.fit(training_data, training_target)
    # 3. print score
    score = model.score(testing_data, testing_target)
    scores.append(score)
    print('%i. %d' % (idx, score))
    # idx++
    idx += 1

# print mean score
print('kfold mean: %d' % np.mean(scores))

# after predicting score train everything and save to binaries
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

with open('models/vectorizer.pkl', 'wb') as f:
    pkl.dump(vectorizer, f)

with open('models/model.pkl', 'wb') as f:
    pkl.dump(model, f)
