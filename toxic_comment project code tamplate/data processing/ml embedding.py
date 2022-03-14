import pandas as pd
import numpy as np
import tqdm
from nltk.probability import FreqDist

data = pd.read_csv('../inputs/pro_train.csv')['comment_text']
test = pd.read_csv('../inputs/pro_test.csv')['comment_text']
data = pd.concat([data, test], axis=0)



def prepare_text_dict(text, min_freq=3):
    from nltk.probability import FreqDist
    # split text
    all_words = [w for row in text for w in row.split()]

    # the frequencies of words in text
    freq = FreqDist(all_words)
    clean_words = []

    # drop words occurred less min_freq
    for row in text:
        for word in row.split():
            if freq[word] > min_freq:
                clean_words.append(word)

    # vocabs set
    clean_words = set(clean_words)
    # convert words to indexes
    word2index = {w: i for i, w in enumerate(clean_words, 2)}  # start at 2
    word2index['<pad>'] = 1
    word2index['<unk>'] = 0

    # convert indexes to words
    index2word = {i: w for i, w in enumerate(clean_words, 2)}  # start at 2
    index2word[1] = '<pad>'
    index2word[0] = '<unk>'

    del freq  # just to save some RAM memory

    return word2index, index2word


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
    return embeddings, vocab


# create Look-Up table
def create_embedding_matrix_for_sentence(data, word_index, embedding_dict, dim):
    print('create_embedding_matrix...')

    embedding_dict['<pad>'] = np.ones(dim)
    embedding_matrix = []
    for row in tqdm.tqdm(data, total=len(data)):
        row_matrix = []
        for word in row:
            if word in word_index:
                row_matrix.append(np.array(embedding_dict[word], dtype='float'))
            else:
                row_matrix.append(np.random.uniform(-0.25, 0.25, dim))

        embedding_matrix.append(np.mean(row_matrix, axis=0))
    return embedding_matrix


# process text dict
word2index, index2word = prepare_text_dict(data)  # notice that we are won't use the 'test_set' vocabs
# word2index = pd.read_pickle('embedding utils/word2index.pkl')
twitter, vocabs = create_embedding_dict(r'/inputs/glove.twitter.27B.50d.txt', vocab=word2index)

matrix = create_embedding_matrix(test, word2index, twitter, dim=50)
pd.to_pickle(matrix, 'test_twitter_ml_matrix.pkl')
pd.to_pickle(word2index, 'word2index.pkl')
print('Total word2index ', len(word2index))
print('twitter_tfidf_matrix: ', len(matrix))







