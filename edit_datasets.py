import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import emoji

df = pd.read_csv('datasets/train.csv', sep=';')

# index the class
sentiments = {
    'positive': '3',
    'neutral': '2',
    'negative': '1'
}


def index_class(sent):
    return sentiments[sent]


def word_count(sent):
    return len(sent.split())


def remove_urls(sent):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sent)


# Work for both emojis and special content.

def emoji_extraction(sent):
    e_sent = emoji.demojize(sent)

    return re.findall(':(.*?):', e_sent)


def emoji_count(sent):
    e_sent = emoji.demojize(sent)
    return len(re.findall(':(.*?):', e_sent))


def emoji_to_text(sent):
    e_sent = emoji.demojize(sent)
    emo = re.findall(':(.*?):', e_sent)
    for e in emo:
        e_sent = e_sent.replace(':{}:'.format(e), '{}'.format(e))
    return e_sent


def find_hashtags(text):
    return re.findall(r'#\w*[a-zA-Z]\w*', text)


def count_hashtags(text):
    get_hashtags = re.findall(r'#\w*[a-zA-Z]\w*', text)
    return len(get_hashtags)


def remove_hashtags(text):
    return re.sub('#\w*[a-zA-Z]\w*', '', text)


def extract_username(sent):
    usernames = re.findall('@[A-Za-z0-9_$]*', sent)
    return usernames


def replace_username(sent):
    usernames = extract_username(sent)
    for un in usernames:
        un = re.sub('@', '', un)
        sent = sent.replace('@{}'.format(un), '{}'.format(un))
    return sent


def find_number(text):
    return re.findall(r'#[0-9]+', text)


def remove_number(text):
    return re.sub('#[0-9]+', '', text)


def find_punctuations(text):
    return re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text)


def count_punctuations(text):
    return len(re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text))


def remove_punctuations(text):
    return re.sub(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', '', text)


def remove_symbols(text):
    return re.sub('[~:*ÛÓ_å¨È$#&%^ª|+-]+?', '', text)


print('start')
df['class'] = df['class'].apply(index_class)
print('index classes')
df['text'] = df['text'].apply(str)
print('ensure strings')
df['char_lens'] = df['text'].apply(len)
print('char_lens')
df['word_count'] = df['text'].apply(word_count)
print('word_count')
df['text'] = df['text'].apply(remove_urls)
print('remove urls')
df['emoji_count'] = df['text'].apply(emoji_count)
print('emoji_count')
df['emojis'] = df['text'].apply(emoji_extraction)
print('emojis')
df['text'] = df['text'].apply(emoji_to_text)
print('emoji to text')
df['hash_count'] = df['text'].apply(count_hashtags)
print('hash_count')
df['hashtags'] = df['text'].apply(find_hashtags)
print('hashtags')
df['text'] = df['text'].apply(remove_hashtags)
print('remove hashtags')
df['text'] = df['text'].apply(replace_username)
print('replace usernames')
df['number'] = df['text'].apply(find_number)
print('number')
df['text'] = df['text'].apply(remove_number)
print('remove numbers')
df['count_punctuation'] = df['text'].apply(count_punctuations)
print('count_punctuation')
df['punctuation'] = df['text'].apply(find_punctuations)
print('punctuation')
df['text'] = df['text'].apply(remove_punctuations)
print('remove punctuations')
df['text'] = df['text'].apply(remove_symbols)
print('remove symbols')

# stop = list(stopwords.words('english'))

# train, valid = train_test_split(df_train, test_size=0.2, random_state=0, stratify=df_train.target.values)

df.to_csv('datasets/train2.csv', sep=';', mode='w')

# vectorizer = CountVectorizer(decode_error='replace', stop_words=stop)
#
# X_train = vectorizer.fit_transform(train.text.values)
# X_valid = vectorizer.transform(valid.text.values)
#
# y_train = train.target.values
# y_valid = valid.target.values
#
# print("X_train.shape : ", X_train.shape)
# print("X_train.shape : ", X_valid.shape)
# print("y_train.shape : ", y_train.shape)
# print("y_valid.shape : ", y_valid.shape)
