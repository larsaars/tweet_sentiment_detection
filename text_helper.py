import re
import emoji


def word_count(sent):
    return len(sent.split())


def remove_urls(sent):
    return re.sub(r'(?:(?:https?|ftp)://)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', '', sent)


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
    return re.findall(r'@[A-Za-z0-9_$]*', sent)


def replace_username(sent):
    usernames = extract_username(sent)
    for un in usernames:
        un = re.sub('@', '', un)
        sent = sent.replace('@{}'.format(un), '{}'.format(un))
    return sent


def remove_usernames(text):
    return re.sub(r'@[A-Za-z0-9_$]*', '', text)


def remove_numbers(text):
    return re.sub(r'#[0-9]+', '', text)


def find_punctuations(text):
    return re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text)


def count_punctuations(text):
    return len(re.findall(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', text))


def remove_punctuations(text):
    # keep points for splitting
    return re.sub(r'[.?"\'`,\-!:;()\[\]\\/“”]+?', '.', text)


def keep_characters(text):
    alphabet = 'abcdefghijklmnopqrstuvwxyz .,?!_1234567890'
    # only keep alphabet
    new = ''
    # check every letter if can be accepted (is in language)
    for letter in text:
        for c in alphabet:
            if letter == c:
                new += c
                break

    # remove multiple whitespaces
    new = re.sub(r'\s+', ' ', new.replace('\n', ' '))

    return new


def uppercase_letters(text):
    upper = 0
    for letter in text:
        if letter.isupper():
            upper += 1
    return upper


def transform(text) -> str:
    # make sure text is a lowercase string
    text = str(text).lower()
    # emojis to text
    text = emoji_to_text(text)
    # remove all urls
    text = remove_urls(text)
    # remove the username (everything beginning with '@')
    text = remove_usernames(text)
    # keep only the important characters now
    text = keep_characters(text)
    # remove all whitespaces at the start of the string and return
    return text.strip()
