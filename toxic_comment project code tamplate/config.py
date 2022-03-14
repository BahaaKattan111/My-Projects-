import pandas as pd
import numpy as np

average_sub = pd.DataFrame()
sub_0 = pd.read_csv('submission_LSTM_0.csv')
average_sub['id'] = sub_0['id']
sub_0.drop('id', axis=1, inplace=True)
sub_1 = pd.read_csv('submission_LSTM_1.csv').drop('id', axis=1)
sub_2 = pd.read_csv('submission_LSTM_2.csv').drop('id', axis=1)
sub_3 = pd.read_csv('submission_LSTM_3.csv').drop('id', axis=1)
sub_4 = pd.read_csv('submission_LSTM_4.csv').drop('id', axis=1)
for col in sub_0.columns:
    average_sub[col] = (sub_0[col] +  sub_1[col] + sub_2[col] + sub_3[col] + sub_4[col]) / 5
print(average_sub)

# pseudo labeling
'''
average_sub = pd.read_csv('../subs/test_pred.csv')
average_sub['comment_text'] = pd.read_csv('../inputs/test.csv')['comment_text']

pl_train = pd.DataFrame()

for col in average_sub.drop(['id', 'comment_text'], axis=1):
    rounded_cold = []
    for v in average_sub[col].values:
        if v >= 0.80:
            rounded_cold.append((1))
        elif v <= 0.3:
            rounded_cold.append((0))
        else:
            rounded_cold.append(v)

    average_sub[col] = rounded_cold

# transpose data
T = pd.DataFrame(average_sub.drop(['id', 'comment_text'], axis=1).T)
# take only data with no floot values (only [0.,1.]
ps_test_cols = []
ps_test_text = []
for col, text in zip(T, average_sub['comment_text']):
    # convert values to str so we can count how many digits they are
    s = T[col].values.astype(str)
    z = []
    for v in s:
        z.append(len(v))
    if max(z) == 3:
        # append values that are only 33 ch (e.g 0.0, 1.0)
        ps_test_cols.append(T[col])
        ps_test_text.append(text)
ps_test_text = pd.DataFrame(ps_test_text, columns=['comment_text']).reset_index(drop=True)
ps_test_cols = pd.DataFrame(ps_test_cols, columns=average_sub.drop(['id', 'comment_text'], axis=1).columns).reset_index(drop=True)
ps_test = pd.concat([ps_test_text, ps_test_cols],1)
print(ps_test)

ps_test.to_csv('pseudo_labeling_test.csv',index=False)
'''
# model_1
"""class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 200, 2, batch_first=True, bidirectional=True, dropout=.4)
        self.Linear_1 = nn.Linear(2*200, 300)
        self.norm = nn.BatchNorm1d(300)
        # self.Linear_2 = nn.Linear(300, 100)
        self.out = nn.Linear(300, 6)
        self.n_layers = 4
        self.n_hidden = 200
        self.bs = 32
    def init_hidden(self):
        device = "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, self.bs, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, self.bs, self.n_hidden).zero_().to(device))

        return h

    def forward(self, x):
        x = self.embedding(x)
        h = self.init_hidden()
        out, (hid, c) = self.lstm(x, h)
        # avg_pool = torch.mean(out, 1)
        # max_pool, _ = torch.max(out, 1)
        # x = nn.Dropout(.4)(torch.cat((avg_pool, max_pool), 1).resize(bs, -1))
        cat = torch.cat([hid[-1, :, :], hid[-2, :, :]], 1)
        x = F.leaky_relu(nn.Dropout(.4)(self.Linear_1(cat)), 0.001)
        x = self.norm(x)
        x = torch.sigmoid(self.out(x))
        return x
"""

# PCNN model_2 0.92
'''class PCNN(nn.Module):

    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super(PCNN, self).__init__()
        # self.embed_glove = nn.Embedding(embed_num, embed_dim)
        # self.embed_tweeter = nn.Embedding(embed_num, embed_dim)
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(.5)
        # self.fc1 = nn.Linear(320, 256)
        self.out = nn.Linear(512, 6)

    def Piecewise_MaxPool1d(self, x):
        # if x.size(2) > 6:
        index = 5
        part_1 = x[:, :, :index]
        part_2 = x[:, :, -index:]
        part_1 = F.max_pool1d(part_1, part_1.size(2)).squeeze(2)
        part_2 = F.max_pool1d(part_2, part_2.size(2)).squeeze(2)
        x = torch.cat([part_1, part_2], 1)
        return x

    # else:
    #    return F.max_pool1d(x, x.size(2)).squeeze(2)
    def forward(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x_ = [F.tanh(self.Piecewise_MaxPool1d(i)) for i in x]  # [(N, Co), ...]*len(Ks)
        x_ = self.dropout(torch.cat(x_, 1))
        return torch.sigmoid(self.out(x_))

embed_num = len(word2index) # 90442
embed_dim = 50
kernel_num = 64
kernel_sizes = [1, 2, 4, 6]
model = CNN_Text(embed_num, embed_dim, kernel_num, kernel_sizes)
model.load_state_dict(pretrained)
model.to('cuda')'''

# MAX&AVG_POOL_LSTM model_3 0.95
'''class LSTM_2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, n_hidden=100, n_layers=3):
        super(LSTM_2, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = 32
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.n_hidden, self.n_layers, batch_first=True, bidirectional=True, dropout=.6)
        for i, layer in enumerate(self.lstm._all_weights):
            for j, w in enumerate(layer):
                if 'bias' in (w):
                    ones = len(self.lstm.all_weights[i][j])
                    self.lstm.all_weights[i][j] = torch.ones(ones)
        self.out = nn.Linear(int(self.n_hidden*2/2), 6)


    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers*2, batch_size, self.n_hidden).zero_().to(device))
        return h

    def forward(self, x, hiddens):
        x = self.embedding(x)
        out, (hid, c) = self.lstm(x, hiddens)
        cat = torch.cat([hid[-1, :, :], hid[-2, :, :]], 1)
        # avg_pool = torch.avg_pool1d(cat, 2)

        max_pool = torch.tanh(torch.max_pool1d(cat, 2))
        # x = nn.Dropout(.4)(1torch.cat((avg_pool, max_pool), dim=2)).squeeze(1)
        #x = F.relu(nn.Dropout(.6)(self.Linear_1(max_pool)))
        # x = self.norm(x)
        x = nn.Dropout(.6)(max_pool)
        x = torch.sigmoid(self.out(x))
        return x
'''

# gru + ps_test 0.965
'''class GRU(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, n_hidden=50, n_layers=2):
        super(LSTM_2, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = 32
        self.epoch = 1

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.GRU(embedding_dim, self.n_hidden, 2, batch_first=True, bidirectional=True, dropout=0.5)

        self.out = nn.Linear(n_hidden*2, 6)

        for i, layer in enumerate(self.lstm._all_weights):
            for j, w in enumerate(layer):
                if 'bias' in (w):
                    ones = len(self.lstm.all_weights[i][j])
                    self.lstm.all_weights[i][j] = torch.ones(ones)

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        o = np.random.uniform(-0.25, 0.25, (4, batch_size, self.n_hidden))
        h = weights.new(torch.tensor(o, requires_grad=False, device=device, dtype=torch.float)).to(device)
        return h

    def forward(self, x, pos, h):
        x = self.embedding(x)
        out, hid = self.lstm1(x, h)
        if self.epoch < 3:
            hid = F.dropout(hid, 0.25)
        elif 5 > self.epoch > 3:
            hid = F.dropout(hid, 0.50)
        elif 5 >= self.epoch:
            hid = F.dropout(hid, float(np.random.uniform(0.50, 0.8, 1).round(3)))

        cat = torch.cat([hid[-1, :, :], hid[-2, :, :]], 1)
        x = torch.sigmoid(self.out(cat))
        return x'''

# LSTM_FEATURES 0.971
'''
class LSTM_2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, n_hidden=32, n_layers=1, n_features=121):
        super(LSTM_2, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = 32
        self.epoch = 1

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.GRU(embedding_dim, n_hidden, n_layers, batch_first=True, bidirectional=True, dropout=0.4)

        self.linear_1 = nn.Linear(376, 500)
        self.norm = nn.BatchNorm1d(500)
        self.linear_2 = nn.Linear(500, 100)

        self.out_fe = nn.Linear(100, 6)  # n_hidden*2 +
        self.out = nn.Linear(n_hidden * 2, 6)

        for i, layer in enumerate(self.lstm._all_weights):
            for j, w in enumerate(layer):
                if 'bias' in (w):
                    ones = len(self.lstm.all_weights[i][j])
                    self.lstm.all_weights[i][j] = torch.ones(ones)

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        o = np.random.uniform(-0.1, 0.1, (2, batch_size, self.n_hidden))
        h = weights.new(torch.tensor(o, requires_grad=False, device=device, dtype=torch.float)).to(device)
        return h

    def forward(self, x, features, h):
        embed = F.dropout(self.embedding(x), 0.2)
        out, hid = self.lstm1(embed, h)

        x_fe = torch.tanh(F.dropout(self.linear_1(features), 0.4))
        x_fe = self.norm(x_fe)
        x_fe = torch.tanh(F.dropout(self.linear_2(x_fe), 0.4))

        # out, hid = self.lstm2(out, hid)
        cat = torch.cat([hid[-1, :, :], hid[-2, :, :]], 1)

        x_fe = torch.sigmoid(self.out_fe(x_fe)) * 0.5
        x = torch.sigmoid(self.out(cat)) * 0.5
        x = x + x_fe
        return x

'''