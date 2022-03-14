import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw

data = pd.read_csv('../../inputs/pro_train.csv')
valid = data.sample(frac=0.3, random_state=99)
print(len(data))
data.drop(index=valid.index, inplace=True)
print(len(data))
classes = data.iloc[:, 2:].sum()
print(classes)
print()
classes = valid.iloc[:, 2:].sum()
print(classes)
aug = naw.WordEmbsAug(model_type='glove', model_path='../../inputs/glove.twitter.27B.50d.txt')

# apply the function above to your text data and create a new column

from googletrans import Translator
import numpy as np


def back_translate(sequence, PROB=1):
    languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
                 'sw', 'vi', 'es', 'el']

    # instantiate translator
    translator = Translator()

    # store original language so we can convert back
    # org_lang = translator.detect(sequence).lang

    # randomly choose language to translate sequence to
    # random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])

    # translate to new language and back to original
    translated = translator.translate(sequence, dest='ar', src='en').text
    # translate back to original language
    translated_back = translator.translate(translated, dest='en', src='ar').text

    # apply with certain probability
    if np.random.uniform(0, 1) <= PROB:
        output_sequence = translated_back
    else:
        output_sequence = sequence

    return output_sequence


def google_augmenting(col_text, col_classes):
    print('len text before aug: ', len(col_text))

    all_text = []
    all_class = []

    for text, classes in zip(col_text, col_classes):
        label_n = []
        text = back_translate(text)
        label_n.append(classes)
        all_text.append(text)
    print('len text after aug: ', len(all_text))
    print('len aug: ', len(all_text) - len(col_text))

    return np.array(all_text), np.array(all_class)


def augmenting(col_text, col_classes, n=2):
    print('len text before aug: ', len(col_text))

    all_text = []
    all_class = []

    for text, classes in zip(col_text, col_classes):
        label_n = []
        text = aug.augment(text, n=n)
        label_n.append(classes)
        for i in range(n):
            all_text.append(text[i])
            all_class.append(classes)
    print('len text after aug: ', len(all_text))
    print('len aug: ', len(all_text) - len(col_text))

    return np.array(all_text), np.array(all_class)


def augment_col(data, col, google=False):
    col_text = data.loc[data[col] == 1, 'comment_text'].values
    col_classes = data.loc[data[col] == 1, 'toxic':].values
    if google:
        all_text, all_class = google_augmenting(col_text, col_classes)
    else:
        all_text, all_class = augmenting(col_text, col_classes)
    augment_df = pd.DataFrame(all_class,
                              columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    augment_df['comment_text'] = all_text

    data = pd.concat([data, augment_df], axis=0)
    return data


threat_aug = augment_col(data, 'threat', google=False)
severe_toxic_aug = augment_col(data, 'severe_toxic', google=False)
obscene_hate_aug = augment_col(data, 'obscene', google=False)
insult_hate_aug = augment_col(data, 'insult', google=False)
identity_hate_aug = augment_col(data, 'identity_hate', google=False)
data = pd.concat([threat_aug, severe_toxic_aug, severe_toxic_aug, obscene_hate_aug, insult_hate_aug], axis=0)
classes = data.iloc[:, 2:].sum()
print(classes)
data.to_csv('aug_pro_train.csv', index=False)
valid.to_csv('aug_pro_vaild.csv', index=False)
