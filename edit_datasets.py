import pandas as pd
import re
import emoji

df = pd.read_csv('datasets/train.csv', sep=';')


def index_class(sent):
    return {
        'positive': '3',
        'neutral': '2',
        'negative': '1'
    }[sent]


def word_count(sent):
    return len(sent.split())


def remove_urls(sent):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sent)


def emoji_count(sent):
    e_sent = emoji.demojize(sent)
    return len(re.findall(r':(.*?):', e_sent))


def emoji_to_text(sent):
    e_sent = emoji.demojize(sent)
    emo = re.findall(r':(.*?):', e_sent)
    for e in emo:
        e_sent = e_sent.replace(':{}:'.format(e), '{}'.format(e))
    return e_sent


def count_hashtags(text):
    get_hashtags = re.findall(r'#\w*[a-zA-Z]\w*', text)
    return len(get_hashtags)


def remove_hashtags(text):
    return re.sub(r'#\w*[a-zA-Z]\w*', '', text)


def extract_username(sent):
    usernames = re.findall(r'@[A-Za-z0-9_$]*', sent)
    return usernames


def replace_username(sent):
    usernames = extract_username(sent)
    for un in usernames:
        un = re.sub('@', '', un)
        sent = sent.replace('@{}'.format(un), '{}'.format(un))
    return sent


def remove_number(text):
    return re.sub(r'#[0-9]+', '', text)


def find_punctuations(text):
    return re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text)


def count_punctuations(text):
    return len(re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text))


def remove_punctuations(text):
    # keep points for splitting
    return re.sub(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', '.', text)


def remove_symbols(text):
    # only keep alphabet
    new = ''
    for letter in text:
        for c in 'abcdefghijklmnopqrstuvwxyz ':
            if letter == c:
                new += c
                break

    return re.sub(r' +', ' ', new)


def uppercase_letters(text):
    upper = 0
    for letter in text:
        if letter.isupper():
            upper += 1
    return upper


print('start')
df['class'] = df['class'].apply(index_class)
print('index classes')
df['text'] = df['text'].apply(str)
print('ensure strings')
df['chars'] = df['text'].apply(len)
print('chars')
df['word_count'] = df['text'].apply(word_count)
print('word_count')
df['text'] = df['text'].apply(remove_urls)
print('remove urls')
df['text'] = df['text'].apply(emoji_to_text)
print('emoji to text')
df['hash_count'] = df['text'].apply(count_hashtags)
print('hash_count')
df['text'] = df['text'].apply(remove_hashtags)
print('remove hashtags')
df['text'] = df['text'].apply(replace_username)
print('replace usernames')
df['text'] = df['text'].apply(remove_number)
print('remove numbers')
df['punctuation_count'] = df['text'].apply(count_punctuations)
print('punctuation_count')
df['punctuation'] = df['text'].apply(find_punctuations)
print('punctuation')
df['text'] = df['text'].apply(remove_punctuations)
print('remove punctuations')
df['uppercase'] = df['text'].apply(uppercase_letters)
print('uppercase')
df['text'] = df['text'].apply(str.lower)
print('to lower')
df['text'] = df['text'].apply(remove_symbols)
print('remove symbols')


df.to_csv('datasets/train2.csv', sep=';', mode='w')
