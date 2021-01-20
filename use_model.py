import pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import text_helper

vectorizer: CountVectorizer
model: LogisticRegression

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pkl.load(f)

with open('models/model.pkl', 'rb') as f:
    model = pkl.load(f)


def predict(text: str):
    return model.predict(vectorizer.transform([text]))
