from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('datasets/train2.csv', sep=';')

stop = list(stopwords.words('english'))

train, valid = train_test_split(df, test_size=0.2, random_state=0, stratify=df.target.values)

vectorizer = CountVectorizer(decode_error='replace', stop_words=stop)

X_train = vectorizer.fit_transform(train['text'].values.astype('U'))
X_valid = vectorizer.transform(valid['text'].values.astype('U'))

y_train = train.target.values
y_valid = valid.target.values

print("X_train.shape : ", X_train.shape)
print("X_train.shape : ", X_valid.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_valid.shape)

model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

# todo: test other models, implement text features (extracted), save the model and vectorizer and predict for own
# todo: port to app or desktop usage / tensorflow model etc.


