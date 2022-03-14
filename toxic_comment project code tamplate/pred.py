import os

import pandas as pd

average_sub = pd.DataFrame(columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
import torch
import deep_urils
from torch.utils import data
from sklearn.model_selection import KFold
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#df = pd.read_csv('aug_pro_train_.csv')
#valid = pd.read_csv('aug_pro_vaild_.csv')
# test = pd.read_csv('../inputs/pro_test_FE_drop.csv')

pos2index = pd.read_pickle('embedding utils/big_word2index.pkl')


sample_submission = pd.read_csv('../inputs/sample_submission.csv')
#for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
#    test[col] = np.random.uniform(.0, 1, len(test))

word2index = pd.read_pickle('embedding utils/big_word2index.pkl')
embed_num = len(word2index)
embed_dim = 50

kernel_num = 100
kernel_sizes = [1, 2, 3, 4, 7]

if __name__ == '__main__':
    for i, file in enumerate(os.listdir('run files')[:1]):
        if 'model' in file:
            pretrained = torch.load(f'run files/{file}')
            print('pred...')
            model = deep_urils.C_LSTM(embed_num, embed_dim, kernel_num, kernel_sizes)
            # model = deep_urils.CNN_Text(embed_num, embed_dim, kernel_num, kernel_sizes)
            model.load_state_dict(pretrained)
            model.to('cuda')

            train = pd.read_csv('../inputs/aug_pro_train.csv')
            valid = pd.read_csv('../inputs/aug_pro_valid.csv')
            from sklearn.metrics import roc_auc_score
            '''
                        df['kfold'] = -1
            kf = KFold(5)
            for fold, (tr, val) in enumerate(kf.split(df)):
                df.loc[val, 'kfold'] = fold
            train = df[df.kfold != 4].reset_index(drop=True)
            y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
            valid = df[df.kfold == 4].reset_index(drop=True)
            y_val = valid[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
            train.drop('kfold', axis=1, inplace=True)
            valid.drop('kfold', axis=1, inplace=True)'''
            from sklearn.model_selection import train_test_split

            train, valid = train_test_split(df, test_size=0.3, random_state=999)

            print(f'shape= train: {train.shape} ~ val: {valid.shape}')

            train_set = deep_urils.TEST_Data(train, word2index, pos2index)
            val_set = deep_urils.TEST_Data(valid, word2index, pos2index)
            train_loader = data.DataLoader(train_set, batch_size=50, collate_fn=deep_urils.collate_fn_padded_test)
            val_loader = data.DataLoader(val_set, batch_size=50, collate_fn=deep_urils.collate_fn_padded_test)
            prediction = deep_urils.pred(model, val_loader)
            
            scores = []
            cols = 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
            for i, col in  enumerate(cols):
                print('RUC_AUC:')
                score = roc_auc_score(valid.loc[:, col], np.vstack(prediction)[:, i])
                print(f'{col}=', score)
                scores.append(score)
            print(f'mean score=', np.mean(scores))
            '''
            test_set = deep_urils.TEST_Data(test, word2index, pos2index)
            test_loader = data.DataLoader(test_set, batch_size=50, collate_fn=deep_urils.collate_fn_padded_test)
            prediction = deep_urils.pred(model, test_loader)
            labels = pd.DataFrame(np.vstack(prediction), columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

            sub = pd.concat([sample_submission['id'], labels], axis=1)
            sub.to_csv(f'submission_LSTM_50d.csv',index=False)
            print(f'submission_LSTM_fedrop.csv DONE')
            '''