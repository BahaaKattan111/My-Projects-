import time

from sklearn import metrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import warnings
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

pd.set_option('display.width', 2222)

warnings.filterwarnings('ignore')

df = pd.read_csv(r'pro_train.csv')
test = pd.read_csv(r'pro_test.csv')
sub = pd.read_csv('..\inputs\sample_submission.csv')

'''f_df = pd.read_csv(r'../inputs/pro_train_FE_drop.csv').loc[:, '1':]
f_df_ = pd.read_csv(r'../inputs/pro_train_&_features.csv').loc[:, '!':]
df = pd.concat([df, f_df_, f_df], axis=1)
df.to_csv('pro_train_all_features.csv',index=False)

f_test = pd.read_csv(r'../inputs/pro_test_&_features.csv').loc[:, '!':]
f_test_ = pd.read_csv(r'../inputs/pro_test_FE_drop.csv').loc[:, '1':]
test = pd.concat([test, f_test_, f_test], axis=1)
df.to_csv('pro_test_all_features.csv',index=False)
test_feats = pd.concat([f_test, f_test_], axis=1)
df_feats = pd.concat([f_df, f_df_], axis=1)
'''

from sklearn.preprocessing import StandardScaler

feats = pd.concat([df.loc[:, '!':], test.loc[:, '!':]], axis=0)

scaler = StandardScaler()
scaler.fit(feats)

df['kfold'] = -1
kf = KFold()
for fold, (tr, val) in enumerate(kf.split(df)):
    df.loc[val, 'kfold'] = fold


def run(fold):
    df['comment_text'].fillna('something', axis=0, inplace=True)
    train = df[df.kfold != 1].reset_index(drop=True).drop('kfold', 1)
    valid = df[df.kfold == 1].reset_index(drop=True).drop('kfold', 1)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    tfidf.fit(df['comment_text'])
    x_train = tfidf.transform(train['comment_text'].values)
    #x_train = scaler.transform(train.loc[:, '!':])
    # x_train = np.concatenate([matrix, feats],axis=1)
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    x_test = tfidf.transform(valid['comment_text'].values)
    # x_test = scaler.transform(valid.loc[:, '!':])
    # x_test = np.concatenate([matrix, feats],axis=1)
    y_test = valid[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    print(f'train shape {x_train.shape} ; valid shape {x_test.shape} ')
    print(f'train_y shape {y_train.shape} ; valid_y shape {y_test.shape} ')
    #clf = LogisticRegression(C=10, penalty='l2', solver='liblinear', random_state=45)
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier()
    clf = OneVsRestClassifier(clf)

    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()

    score_train = metrics.roc_auc_score(y_train, clf.predict(x_train), average='weighted')
    score_test = metrics.roc_auc_score(y_test, clf.predict(x_test), average='weighted')
    print(f'train_score: {score_train:5f} | valid_score: {score_test:5f}, fold: {fold}, time: {np.round((end - start) / 60, 2)}:min')

    r'''test = pd.read_csv(r'pro_test_all_features.csv')
    matrix = tfidf.transform(test['comment_text'].values).todense()
    # test = scaler.transform(test.loc[:, '!':])
    # test = np.concatenate([matrix, feats], axis=1)
    preds = clf.predict(tfidf.transform(test))

    submission = pd.concat([sub['id'], pd.DataFrame(preds,
                                                    columns=["toxic", "severe_toxic", "obscene", "threat", "insult",
                                                             "identity_hate"])], axis=1)
    print(submission.head())
    submission.to_csv(rf'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\src\sub_.csv', index=False)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    run(fold=1)
