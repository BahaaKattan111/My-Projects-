import random

import pandas as pd
import numpy as np
import tqdm
from nltk.probability import FreqDist

data = pd.read_csv('../inputs/aug_pro_train_1.csv')['comment_text']
vaild = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\aug_pro_vaild_1.csv')['comment_text']
test = pd.read_csv('../inputs/pro_test.csv')['comment_text']
data = pd.concat([data, vaild, test], axis=0)
# Term Frequency and Inverse Document Frequency

def prepare_text_dict(text, min_freq=1, max_freq=5):
    from nltk.probability import FreqDist
    # split text
    all_words = [w for row in text for w in row.split()]

    # the frequencies of words in text
    freq = FreqDist(all_words)
    max_freq = (max(freq.values()) - max_freq)
    clean_words = []

    # drop words occurred less min_freq
    for row in text:
        for word in row.split():
            if (freq[word] > min_freq) & (freq[word] < max_freq):
                clean_words.append(word)

    # vocabs set
    clean_words = set(clean_words)
    # convert words to indexes
    word2index = {w: i for i, w in enumerate(clean_words,2)}  # start at 2
    word2index['<pad>'] = 1
    word2index['<unk>'] = 0

    del freq  # just to save some RAM memory

    return word2index

def create_embedding_dict(file, vocab):
    embeddings = {}
    print('\nembeddings...')

    with open(file, 'r', encoding='UTF-8') as f:
        for line in tqdm.tqdm(f.readlines(), total=len(f.readlines())):
            vector = line.split()  # split by 'space'

            word = vector[0]  # word
            dims = vector[1:]  # embedding vector
            if word in vocab.keys():  # if word is in our vocabs:
                embeddings[word] = np.array(dims)  # assign 'word' to its vector embedding in 'tweeter_embeddings' dict

    missing = len(vocab) - len(embeddings)
    print(
        f'Done! embeddings found for our vocabs: {len(embeddings)} | missing: {missing} ; {missing / len(word2index) * 100:.1f}%')
    try:
        return pd.DataFrame(embeddings)
    except:
        print(len(embeddings))


# create Look-Up table
def create_embedding_matrix(word_index, embedding_dict, vocabs, dim):
    print('create_embedding_matrix...')
    embedding_dict['<pad>'] = np.ones(dim)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocabs), dim))
    # embedding_matrix = np.zeros((len(vocabs), dim))

    for index, word in enumerate(tqdm.tqdm(word_index, total=len(embedding_matrix))):
        if word in embedding_dict:
            try:
                embedding_matrix[index, :] = np.array(embedding_dict[word], dtype='float')
            except:
                continue
        else:
            embedding_matrix[index] = np.random.uniform(-0.25, 0.25, dim)
    return embedding_matrix


# process text dict
word2index = prepare_text_dict(data)  # notice that we are won't use the 'test_set' vocabs
pd.to_pickle(word2index, 'embedding utils/big_word2index.pkl')
glove = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\glove.6B.50d.txt ',vocab=word2index)
glove_matrix = create_embedding_matrix(word2index, glove, word2index, dim=len(glove))
pd.to_pickle(glove_matrix, 'embedding utils/glove_matrix.pkl')

twitter = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\glove.twitter.27B.50d.txt', vocab=word2index)
twitter_matrix = create_embedding_matrix(word2index, twitter, word2index, dim=len(twitter))
pd.to_pickle(twitter_matrix, 'embedding utils/twitter_matrix.pkl')

wiki = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\wiki-news-300d-1M-subword.vec ', vocab=word2index)
wiki_matrix = create_embedding_matrix(word2index, wiki, word2index, dim=len(wiki))
pd.to_pickle(wiki_matrix, 'embedding utils/wiki_matrix.pkl')

big_matrix = np.random.uniform(-0.25, 0.25, (len(word2index), 400))
twitter['<pad>'] = np.ones(50)
glove['<pad>'] = np.ones(50)
wiki['<pad>'] = np.ones(300)

for i, word in tqdm.tqdm(enumerate(word2index.keys()), total=len(word2index)):

    if word in twitter.keys():
        big_matrix[i, :50] = np.array(twitter[word], dtype='float')
    else:
        continue
    if word in glove.keys():
        big_matrix[i, 50:100] = np.array(glove[word], dtype='float')
    else:
        continue
    if word in wiki.keys():
        big_matrix[i, 100:] = np.array(wiki[word], dtype='float')
    else:
        continue
glove_matrix = create_embedding_matrix(word2index, glove, word2index, dim=len(glove))
pd.to_pickle(big_matrix, 'embedding utils/big_matrix.pkl')
print('Total word2index ', len(word2index))

# --fasttext--
# fasttext, vocabs_fasttext = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\wiki-news-300d-1M-subword.vec', vocab=word2index)
# pd.to_pickle(matrix, 'fasttext_matrix.pkl')
# print('fast_text_matrix: ', len(matrix))

# --tweeter--
# tweeter, vocabs_tweeter = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\glove.twitter.27B.50d.txt', vocab=word2index)
# twitter = create_embedding_matrix(word2index, twitter, vocabs_twitter, dim=50)
# pd.to_pickle(twitter, 'embedding utils/twitter_matrix.pkl')
# print('twitter_matrix: ', len(twitter))

# --glove--
# glove, vocabs_glove = create_embedding_dict(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\glove.6B.50d.txt', vocab=word2index)
# glove = create_embedding_matrix(word2index, glove, vocabs_glove, dim=50)
# pd.to_pickle(glove, 'embedding utils/glove_matrix.pkl')
# print('glove_matrix: ', len(glove))

# print('diff matrix: ', twitter - glove)
