import pandas as pd
import re
import textblob
from string import punctuation

pd.set_option('display.max_columns', 220)
pd.set_option('display.width', 2000)
train = pd.DataFrame(pd.read_csv('../../inputs/train.csv', usecols=['comment_text']))
test = pd.DataFrame(pd.read_csv('../../inputs/test.csv', usecols=['comment_text']))

blob = textblob.TextBlob

pos_dict = {
    'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
    'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj': ['JJ', 'JJR', 'JJS'],
    'adv': ['RB', 'RBR', 'RBS', 'WRB']
}


def pos(text, flag):
    text = blob(text).pos_tags

    tags = 0
    for tag in text:
        if tag[1] in pos_dict[flag]:
            tags += 1
    return tags


def find_word(text, target_word):
    i = 0
    for word in text.split():
        if target_word in word:
            i += 1
    return i


def find_pattern(text, pattern):
    text = re.sub(pattern, ' <pattern> ', text)
    i = 0
    for word in text.split():
        if '<pattern>' in word:
            i += 1
    return i


def polarity(text):
    try:
        return blob(text).sentiment.polarity
    except:
        return 0.0


def subjectivity(text):
    try:
        return blob(text).sentiment.subjectivity
    except:
        return 0.0


def fine_word_length(text, bigger_than, smaller_than=999):
    i = 0
    for word in text.split():
        if smaller_than >= len(word) >= bigger_than:
            i += 1
    return i


def extract_features(data):
    for p in punctuation:
        data[f'{p}'] = data['comment_text'].apply(lambda x: find_word(x, f'{p}'))

    data[f'<sent_count>'] = data['comment_text'].apply(lambda x: len(x.split('.')))
    data[f'<word_count>'] = data['comment_text'].apply(lambda x: len(x.split()))
    data[f'<char_count>'] = data['comment_text'].apply(lambda x: len(x))

    data[f'<char/word>'] = data[f'<char_count>'] / (data[f'<word_count>'] + 1)
    data[f'<word/sentence>'] = data[f'<word_count>'] / (data[f'<sent_count>'] + 1)

    data[f'<isupper>'] = data['comment_text'].apply(lambda x: sum([1 for w in x.split() if w.isupper()]))
    data[f'<istitle>'] = data['comment_text'].apply(lambda x: sum([1 for w in x.split() if w.istitle()]))
    data[f'<isdigit>'] = data['comment_text'].apply(lambda x: sum([1 for w in x.split() if w.isdigit()]))

    data['<http>'] = data['comment_text'].apply(lambda x: find_word(x, 'http'))
    data['<gmail>'] = data['comment_text'].apply(lambda x: find_word(x, '@gmail'))
    data['<www>'] = data['comment_text'].apply(lambda x: find_word(x, 'www'))

    data['<phone_number>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"\d{3,}\s\d{3,}\s\d{3,}\s"))
    data['<email>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"\S+\@\S+"))
    data['<username>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"\s\@\S+"))
    data['<hashtag>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"\s\#\S+"))
    data['<21st century>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"20\d{2}"))
    data['<20st century>'] = data['comment_text'].apply(lambda x: find_pattern(x, r"19\d{2}"))

    data['<word_length_0-3>'] = data['comment_text'].apply(lambda x: fine_word_length(x, 0, 3))
    data['<word_length_3-7>'] = data['comment_text'].apply(lambda x: fine_word_length(x, 4, 7))
    data['<word_length_7-11>'] = data['comment_text'].apply(lambda x: fine_word_length(x, 8, 10))
    data['<word_length_11-20>'] = data['comment_text'].apply(lambda x: fine_word_length(x, 11, 20))
    data['<word_length_20-enf>'] = data['comment_text'].apply(lambda x: fine_word_length(x, 20))

    # tags
    for flag in pos_dict.keys():
        data[flag] = data['comment_text'].apply(lambda x: pos(x, flag))

    data['<polarity>'] = data['comment_text'].apply(lambda x: polarity(x))
    data['<subjectivity>'] = data['comment_text'].apply(lambda x: subjectivity(x))
    return data.drop(['comment_text'], axis=1)

train_feaures = extract_features(train)
test_feaures = extract_features(test)
train = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\pro_train.csv')
test = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\pro_test.csv')

train = pd.concat([train, train_feaures], axis=1)
test = pd.concat([test, test_feaures], axis=1)

train.to_csv('pro_train_&_features.csv', index=False)
test.to_csv('pro_test_&_features.csv', index=False)
