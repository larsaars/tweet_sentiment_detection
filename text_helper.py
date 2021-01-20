import re
import emoji


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
        for c in 'abcdefghijklmnopqrstuvwxyz .?!':
            if letter == c:
                new += c
                break

    new = re.sub(r' +', ' ', new.replace('\n', ' '))

    if new.startswith(' '):
        new = new[1:]

    return new


def uppercase_letters(text):
    upper = 0
    for letter in text:
        if letter.isupper():
            upper += 1
    return upper


def transform(text) -> str:
    return remove_symbols(remove_urls(
        replace_username(
            remove_number(remove_punctuations(str.lower(emoji_to_text(str(text))))))))
