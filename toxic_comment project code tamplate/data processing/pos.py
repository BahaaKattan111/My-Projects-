import time
import numpy as np
import pandas as pd
import textblob, gensim
blob = textblob.TextBlob

'''
in this file it's supposed to train a vector embedding but for pos tag and use this embedding file to a multi-channel
2D-CNN layer. 2 channels, A channel for word embedding and a channel for pos embedding, and every channel will be the input
for lstm

and concatenate their hidden_states.

But i didn't do it ...
'''

def pos(text):
    text = blob(text).pos_tags

    tags = []
    for tag in text:
        tags.append(tag[-1])
    return tags

data = pd.read_csv('../../inputs/pro_train.csv')['comment_text']
test = pd.read_csv('../../inputs/pro_test.csv')['comment_text']
data = pd.concat([data, test], axis=0)
start = time.time()
pos_tags = data.apply(lambda x: pos(x))
end = time.time()
print(f'time hours: {np.round((end - start) / 3600, 2)}')
all_words = set([w for row in pos_tags for w in row])
pos2index = {k:v for v, k in enumerate(all_words)}
print(all_words)
pd.to_pickle(pos2index, '../pos2index.pkl')