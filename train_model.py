from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import pickle as pkl
import numpy as np
import text_helper as txth

from sklearn.linear_model import LogisticRegression

# read the csv with all data
df = pd.read_csv('datasets/all/tweets_train.csv', sep=';')
# make sure text is string
df['text'] = df.text.apply(str)
# init list of stopwords
stop = list(stopwords.words('english'))
# count-vectorize words in the given ngram_range
vectorizer = CountVectorizer(decode_error='replace', stop_words=stop, ngram_range=(1, 2))
# fit the vectorizer (etc. for later whole model training)
X_train = vectorizer.fit_transform(df.text.values.astype('U'))
y_train = df.target
# index of the loop and list of scores to calc mean
idx = 0
# init a 5 split k fold for cross validation
for train_df_idx, test_df_idx in KFold(n_splits=5, shuffle=True).split(df):
    # create dataframes from split indices
    train_df = pd.DataFrame(df.iloc[train_df_idx])
    test_df = pd.DataFrame(df.iloc[test_df_idx])

    train_df['text'] = train_df.text.apply(txth.transform)
    test_df['text'] = test_df.text.apply(txth.transform)

    # gather target and training data from data
    training_data = vectorizer.transform(train_df.text.values.astype('U'))
    training_target = train_df.target.values

    # same for for testing split set
    testing_data = vectorizer.transform(test_df.text.values.astype('U'))
    testing_target = test_df.target.values

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
