import pandas as pd
import re
# data = pd.read_csv('../inputs/pro_train.csv')
data = pd.read_csv('../../inputs/train.csv')
test = pd.read_csv('../../inputs/test.csv')







'''data['lens'] = data['comment_text'].apply(lambda x: len(x.split()))
import matplotlib.pyplot as plt
import seaborn as sb
print(np.max(data['lens'])) # 2321 ; 1250
print(np.mean(data['lens'])) # 61 ; 31
print(np.std(data['lens'])) # 98 ; 55
data.drop(index=data[data['lens'] > np.max(data['lens']) - 100].index, axis=0, inplace=True)
print(data[data['lens'] > np.max(data['lens']) - 100].values)
'''

def drop_duplicate_words(text:str):
    import collections
    clean = []
    for word in text.split():
        if collections.Counter(clean)[word] <= 3:
            clean.append(word)


    return ' '.join(clean)



# correct FE words drop duplicate words
def replace(text, words, correct_word):
    for word in words:
        text = re.sub(word, correct_word, text)
    return text

for df in [data, test]:
    df['comment_text'] = df['comment_text'].apply(lambda x: rplace(x, ['bich'], 'bitch'))
    df['comment_text'] = df['comment_text'].apply(lambda x: rplace(x, ['motherfukkin', 'motherfucker', 'muthafucka'], 'motherfucker'))
    df['comment_text'] = df['comment_text'].apply(lambda x: rplace(x, ['niger', 'nigger', 'nig'], 'nigga'))
    df['comment_text'] = df['comment_text'].apply(lambda x: rplace(x, ['fuckin', 'fukin'], 'fucking'))
    df['comment_text'] = df['comment_text'].apply(lambda x: drop_duplicate(rplace(x, ['thank', 'thanks'], 'thanks')))

'''
data.to_csv('../inputs/pro_train.csv', index=False)
test.to_csv('../inputs/pro_test.csv', index=False)
print('done')
'''
def clean_tweet(tweet):

    import re, nltk, contractions
    from nltk.corpus import stopwords

    #from nltk.stem import wordnet

    # Character entity references
    tweet = re.sub(r"&", " and ", tweet)

    # Typos, slang and informal abbreviations
    tweet = re.sub(r"[http]?\S+\.com", " <url> ", tweet)
    tweet = re.sub(r"[www]?\S+\.com", " <url> ", tweet)
    tweet = re.sub(r"\S*\@\S+", " <user> ", tweet)
    tweet = re.sub(r"\#\S+", " <hashtag> ", tweet)
    tweet = re.sub("\s{2,}", " ", tweet)
    tweet = re.sub(r"\d*/\d*/?\d*?", " date ", tweet)
    tweet = re.sub(r"\d{3,}\s\d{3,}\s\d{3,}\s", ' <number> ', tweet)
    tweet = re.sub("\d+:\d+", ' hour', tweet)
    tweet = re.sub(r"20\d{2}", " year ", tweet)
    tweet = re.sub(r"19\d{2}", " year ", tweet)
    tweet = re.sub(r"18\d{2}", " year ", tweet)
    tweet = re.sub(r"\d+(yr| year)", " old year ", tweet)

    # drop drop repeated words
    tweet = tweet.lower()
    import collections
    clean = []
    for word in tweet.split():
        if collections.Counter(clean)[word] < 5:
            clean.append(word)
    # Hashtags and usernames
    '''
    tweet = re.sub(r"january|february|march|april|may|june|july|august|september|october|november|december",
                   " month ", tweet)
    tweet = re.sub(r"\S*music\S*", " music ", tweet)
    tweet = re.sub(r"\S*love\S*", " love ", tweet)
    tweet = re.sub(r"\S*summer\S*", " summer ", tweet)
    tweet = re.sub(r"\S*NASA\S*", "  nasa ", tweet)
    tweet = re.sub(r"\S*book\S*", " book ", tweet)
    tweet = re.sub(r"\S*island\S*", " Island ", tweet)
    tweet = re.sub(r"\S*cit(ies|y)\S*", " city ", tweet)
    tweet = re.sub(r"\sRT\S*", " news ", tweet)
    tweet = re.sub(r"\S*health\S*", " health ", tweet)
    tweet = re.sub(r"\S*save\S*", " save ", tweet)
    tweet = re.sub(r"\S*traffic\S*", " traffic ", tweet)
    tweet = re.sub(r"\S*conflict\S*", " conflict ", tweet)
    tweet = re.sub(r"\S*storm\S*", " storm ", tweet)
    tweet = re.sub(r"\S*oil\S*", " oil ", tweet)
    tweet = re.sub(r"\S*video\S*", " video ", tweet)
    tweet = re.sub(r"\S*fire\S*", " fire ", tweet)
    tweet = re.sub(r"\S*weather\S*", " weather ", tweet)
    tweet = re.sub(r"\S*sun\S+", " sun ", tweet)
    tweet = re.sub(r"\S*bbc\S*", " news", tweet)
    tweet = re.sub(r"\S*day\S*", " day ", tweet)
    tweet = re.sub(r"\S*effect\S*", " effect ", tweet)
    tweet = re.sub(r"\S*terror\S*", " terrorism ", tweet)
    tweet = re.sub(r"\S*social\S*", " social ", tweet)
    tweet = re.sub(r"\S*word\S*", " word ", tweet)
    tweet = re.sub(r"\S*accident\S*", " accident ", tweet)
    tweet = re.sub(r"\S*sport\S*", " sport ", tweet)
    tweet = re.sub(r"\S*news", " news ", tweet)
    tweet = re.sub(r"\S*dam", " dam ", tweet)
    tweet = re.sub(r"\S*video\S*", " video ", tweet)
    tweet = re.sub(r"\S*games?", " game ", tweet)
    tweet = re.sub(r"\S*youtube\S*", " youtube ", tweet)
    tweet = re.sub(r"\shrs ", " hour ", tweet)
    tweet = re.sub(r"\stxt ", " text ", tweet)
    tweet = re.sub(r"\s+k\s+", " ok ", tweet)
    tweet = re.sub(r"\s+b\s+", " be ", tweet)
    tweet = re.sub(r"\s+u[rs]?\s+", " you ", tweet)
    tweet = re.sub(r"\shom ", " home ", tweet)
    tweet = re.sub(r" yous ", " you ", tweet)
    tweet = re.sub(r"\s+n\s+", " in ", tweet)
    tweet = re.sub(r"\sfr\s", " for ", tweet)'''
    tweet = re.sub(r"\d+.?\d+?", " number ", tweet)
    # solve short cuts (e.g you're = you are ...)
    tweet = contractions.fix(tweet)

    tweet = re.sub(r'[^a-z]', ' ', tweet)

    pos_dict = {
        'N': 'n',
        'V': 'v',
        'J': 'a',
        'R': 'r'}
    non_stop = ['no', 'how', 'why', 'ok', 'while', 'but', 'for', 'though', 'although']
    tweet = ' '.join([word for word in tweet.split() if word not in stopwords.words('english') and word.__len__() > 2 and word not in non_stop])
    tuples = nltk.pos_tag(tweet.split())
    tweet = ' '.join([nltk.wordnet.WordNetLemmatizer().lemmatize(tup[0], pos=pos_dict.get(tup[1][0], 'n')) for tup in tuples])
    if tweet == '':
        tweet = 'none'
    return tweet
data['comment_text'] = data['comment_text'].apply(lambda x: clean_tweet(x))
test['comment_text'] = test['comment_text'].apply(lambda x: clean_tweet(x))

data.to_csv('../inputs/pro_train.csv', index=False)
test.to_csv('../inputs/pro_test.csv', index=False)
print('done')


