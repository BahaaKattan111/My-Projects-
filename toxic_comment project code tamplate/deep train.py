import pandas as pd
from torch import nn
from torch.utils import data
import deep_urils, os
import numpy as np
import warnings, time, torch
from sklearn.model_selection import KFold


def seed_everything(seed=10):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
pd.set_option('display.max_columns',20)
pd.set_option('display.width', 2000)

seed_everything()
warnings.filterwarnings('ignore')

# fitting feaures scaler
'''
f_df = pd.read_csv(r'../inputs/pro_train_FE_drop.csv').loc[:, '1':]
f_df_ = pd.read_csv(r'../inputs/pro_train_&_features.csv').loc[:, '!':]
df = pd.concat([df, f_df_, f_df], axis=1)
print(len(df.columns))

f_test = pd.read_csv(r'../inputs/pro_test_&_features.csv').loc[:, '!':]
f_test_ = pd.read_csv(r'../inputs/pro_test_FE_drop.csv').loc[:, '1':]

test_feats = pd.concat([f_test, f_test_], axis=1)
df_feats = pd.concat([f_df, f_df_], axis=1)

from sklearn.preprocessing import StandardScaler
feats = pd.concat([test_feats, df_feats], axis=0)

scaler = StandardScaler()
scaler.fit(feats)'''

word2index = pd.read_pickle('embedding utils/big_word2index.pkl')
# pos2index = pd.read_pickle('pos2index.pkl')
matrix = pd.read_pickle(r'embedding utils/twitter_matrix.pkl')
print('vocabs: ', len(word2index))
print('matrix: ', matrix.shape)
# print('pos2index: ', len(pos2index))

embed_num = len(word2index)
embed_dim = matrix.shape[1]

kernel_num = 100
kernel_sizes = [1, 2, 3, 4, 7]


def weights_init_uniform_rule(m):

    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)

    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        n = m.out_channels

        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

epochs = 10
if __name__ == '__main__':
    # using kflod
    '''df['kfold'] = -1
    kf = KFold(5)
    for fold, (tr, val) in enumerate(kf.split(df)):
        df.loc[val, 'kfold'] = fold
    from sklearn.model_selection import train_test_split
    for i in range(5):
    print(f'...')
    print(f'FOLD: {i}')
    # train = df[df.kfold != i].reset_index(drop=True).drop('kfold', axis=1)
    # valid = df[df.kfold == i].reset_index(drop=True).drop('kfold', axis=1)'''

    # model = deep_urils.CNN_Text(embed_num, embed_dim, kernel_num, kernel_sizes)
    model = deep_urils.GRU(embed_num, embed_dim)
    model.apply(weights_init_uniform_rule)
    model.embedding.weight.data = torch.nn.Parameter(torch.tensor(matrix, dtype=torch.float), requires_grad=True)
    model.to(deep_urils.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    train = pd.read_csv('../inputs/aug_pro_train_1.csv')
    valid = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\toxic_nlp\inputs\aug_pro_vaild_1.csv')
    print(f'shape= train: {train.shape} ~ val: {valid.shape}')
    train_set = deep_urils.Data(train, word2index, {})
    val_set = deep_urils.Data(valid, word2index, {})

    train_loader = data.DataLoader(train_set, batch_size=50, collate_fn=deep_urils.collate_fn_padded)
    val_loader = data.DataLoader(val_set, batch_size=50, collate_fn=deep_urils.collate_fn_padded)
    deep_urils.run(model, train_loader, optimizer, val_loader, scheduler, 1, epochs=epochs)