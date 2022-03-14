import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import time
import torch.nn.functional as F
from sklearn import metrics
import os
import numpy as np
import textblob

device = 'cuda'
blob = textblob.TextBlob


def pos_tag(text):
    text = blob(text).pos_tags

    tags = []
    for tag in text:
        tags.append(tag[-1])
    return tags


class Data(Dataset):
    def __init__(self, df, word2index, pos2index):
        self.pos2index = pos2index  # to index text
        self.word2index = word2index  # to index text

        # sort text by length of total words
        df['length'] = df['comment_text'].apply(lambda x: len(x.split()))
        df.sort_values(by='length', inplace=True)
        df.drop(['length'], axis=1, inplace=True)
        '''
        self.features = df.loc[:, '!':]
        self.features = scaler.transform(self.features)
        '''

        self.text = df['comment_text'].apply(lambda x: x.split()).values
        self.target = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # pos = [self.pos2index[w] for w in pos_tag(self.raw_text[item]) if w in self.pos2index]
        # features = self.features[item]
        sequence = [self.word2index.get(w, 0) for w in self.text[item]]
        target = self.target[item]
        return {'sequence': sequence, 'target': target, 'pos': target}


class TEST_Data(Dataset):
    def __init__(self, df, word2index, pos2index):
        self.pos2index = pos2index  # to index text

        self.word2index = word2index  # to index text
        # prepare data
        self.text = df['comment_text'].apply(lambda x: x.split()).values

        # self.raw_text = df['raw_text']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # pos = [self.pos2index[w] for w in pos_tag(self.raw_text[item]) if w in self.pos2index]
        # features = self.features[item]
        sequence = [self.word2index.get(w, 0) for w in self.text[item]]

        return {'sequence': sequence, 'pos': [2, 3, 4, 4, 1]}


def pad_text(list_text, seq_length, pad_index=1, truncate=300, pos=False):
    paded_text = []

    for text in list_text:
        text = text + (seq_length - len(text)) * [pad_index]  # 'pad_index' is the index for 'PAD' in vocabs dict

        if len(text) > truncate:
            if pos:
                text = text[:truncate * 2]
            else:
                text = text[:truncate]
        if seq_length < 3:
            text += (3 - seq_length) * [pad_index]
        paded_text.append(text)
    return paded_text


def collate_fn_padded(batch):
    target = [b['target'] for b in batch]
    sequence = [b['sequence'] for b in batch]
    pos = [b['pos'] for b in batch]

    max_length = max([len(b) for b in sequence])
    sequence = pad_text(sequence, max_length)

    # max_length = max([len(b) for b in pos])
    # pos = pad_text(pos, max_length, pos=True)

    # convert list to torch.tensor
    return {'target': torch.tensor(target, dtype=torch.float, device=device),
            'pos': torch.tensor(pos, dtype=torch.float, device=device),
            'sequence': torch.tensor(sequence, dtype=torch.long, device=device)}


class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, n_hidden=32, n_layers=1, n_features=121):
        super(BiLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = 32
        self.epoch = 1

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.GRU(embedding_dim, n_hidden, n_layers, batch_first=True, bidirectional=True, dropout=0.4)

        self.linear_1 = nn.Linear(375, 500)
        self.norm = nn.BatchNorm1d(500)
        self.linear_2 = nn.Linear(500, 100)

        self.out_fe = nn.Linear(100, 6)  # n_hidden*2 +
        self.out = nn.Linear(n_hidden * 2, 6)

        '''for i, layer in enumerate(self.lstm._all_weights):
            for j, w in enumerate(layer):
                if 'bias' in (w):
                    ones = len(self.lstm.all_weights[i][j])
                    self.lstm.all_weights[i][j] = torch.ones(ones)'''

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

        x_fe = self.out_fe(x_fe) * 0.5
        x = self.out(cat) * 0.5
        x = x + x_fe
        return x


class GRU(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, n_hidden=64, n_layers=1, n_features=436):
        super(GRU, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bs = 32
        self.epoch = 1

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.GRU(embedding_dim, self.n_hidden, self.n_layers, batch_first=True, bidirectional=True,)
        self.out = nn.Linear(n_hidden * 2, 6)

        # feaures

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        o = np.random.uniform(-.1, .1, (self.n_layers * 2, batch_size, self.n_hidden))
        h = weights.new(torch.tensor(o, requires_grad=False, device=device, dtype=torch.float)).to(device)
        return h

    def forward(self, x, features, h):
        embed = self.embedding(x)
        out, hid = self.lstm1(embed, h)
        cat = torch.cat([hid[-1, :, :], hid[-2, :, :]], 1)
        x = self.out(cat)
        return x


class CNN_Text(nn.Module):

    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super(CNN_Text, self).__init__()
        # self.embed_glove = nn.Embedding(embed_num, embed_dim)
        # self.embed_tweeter = nn.Embedding(embed_num, embed_dim)
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(.3)
        self.hidden_1 = nn.Linear(1280, 1000)
        self.norm_1 = nn.BatchNorm1d(1000)

        self.hidden_2 = nn.Linear(1000, 512)
        self.norm_2 = nn.BatchNorm1d(512)

        self.out = nn.Linear(512, 6)

        '''self.fe_linear_1 = nn.Linear(376, 1000)
        self.fe_norm_1 = nn.BatchNorm1d(1000)
        self.fe_linear_2 = nn.Linear(1000, 512)
        self.fe_norm_2 = nn.BatchNorm1d(512)
        self.out_fe = nn.Linear(512, 6)
        '''

    def forward(self, x, x_fe):
        # features
        # x_fe = torch.relu(self.fe_linear_1(x_fe))
        # x_fe = self.fe_norm_1(x_fe)
        # x_fe = torch.relu(self.fe_linear_2(x_fe))
        # x_fe = self.fe_norm_2(x_fe)

        # embedd -> conv
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(self.dropout(conv(x))).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        # pooling
        x = [torch.tanh(F.max_pool1d(i, i.size(2))).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        # dense layer
        x = F.tanh(self.dropout(self.hidden_1(x)))
        x = self.norm_1(x)

        x = F.tanh(self.hidden_2(x))
        x = self.norm_2(x)

        x = torch.sigmoid(self.out(x))
        # x_ = torch.sigmoid(self.out_(x_)) * 0.5

        # x_fe = torch.sigmoid(self.out_fe(x_fe)) * 0.5
        # x = x + x_fe
        return x


class C_LSTM(nn.Module):

    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super().__init__()
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.conv2d = nn.Conv2d(1, kernel_num, (3, embed_dim))
        self.dropout = nn.Dropout(.2)
        self.lstm1 = nn.LSTM(kernel_num, 100, batch_first=True, bidirectional=True, num_layers=1)
        self.lstm2 = nn.LSTM(250, 100, batch_first=True, bidirectional=True, num_layers=1)

        self.hidden_1 = nn.Linear(200, 200)

        self.out = nn.Linear(200, 6)

    def init_hidden(self, x):
        return 3

    def forward(self, x, x_fe, h):
        # embedd -> conv
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = self.conv2d(x).squeeze(3)  # [(N, Co, W), ...]*len(Ks)
        x = nn.Tanh()(x)
        x = torch.permute(x, [0, 2, 1])

        out, (hid, c) = self.lstm1(x)
        out_, (hid_, c_) = self.lstm2(torch.cat([out, x], 2))
        # hid = F.dropout(torch.cat([hid, hid_], dim=2), 0.6)
        hid = torch.cat([hid[-1, :, :], hid[-2, :, :]], dim=1)

        # out = F.dropout(torch.cat([o for o in hid], dim=1), 0.6)
        # x =F.max_pool1d(hid, 3)

        x = F.elu(self.hidden_1(hid))

        x = self.out(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=200, output_size=6, n_layers=2, p=.8, use_features=False, features_size=None):
        # Note: I will speak in singular form just because the paper is explaining using example of one sentence; but in reality we have more then one sentence
        super(SelfAttention, self).__init__()
        self.use_features = use_features
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, dropout=p, bidirectional=True,
                            batch_first=True)

        # '2 * hidden_size' is same as 'da-by-2u' (e.g. 'any hyperparameter' * '2(num_direction)')
        self.W_s1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # no bias as in paper

        # 'hidden_size' is same as 'da' (e.g. 'any hyperparameter')
        self.W_s2 = nn.Linear(hidden_size, hidden_size, bias=False)  # no bias as in paper
        self.fc_layer = nn.Linear(hidden_size * 2 * hidden_size, hidden_size)  # optinal layer
        self.output = nn.Linear(hidden_size, output_size)

        if self.use_features:
            self.f_Linear = nn.Linear(features_size, hidden_size)
            self.f_output = nn.Linear(hidden_size, output_size)

    def attention_weights(self, H):
        # H shape(batch, seq_len, num_directions * hidden_size)
        W_s1 = torch.tanh(self.W_s1(H))  # same shape as 'H'

        W_s2 = self.W_s2(W_s1)  # same shape as 'H'
        weight_matrix = W_s2.permute(0, 2, 1)  # (batch, num_directions * hidden_size, seq_len)

        A = F.softmax(weight_matrix, dim=2)  # apply softmax for second dim as in the paper (seq_len)
        # the softmax() ensures all the computed weights sum up to 1.

        # now 'A' have the same size as 'n' as in ('n-by-2', which is the shape of H), which means we created the attention_weights
        # as expected from the paper (Since H is sized n-by-2u, the annotation vector 'A' will have a size 'n') step.
        return A
    def init_hidden(self, x):
        return 3
    def forward(self, x, features=None, h=None):
        inputs = self.embedding(x)
        output, (hidden_state, cell_state) = self.LSTM(inputs)

        H = output  # output is the (concatenate each −→ht with ←−ht to obtain a hidden state 'H') step;
        # because output contains all the hidden states.

        attention_weights = self.attention_weights(H)  # apply self-attention mechanism step

        M = torch.bmm(attention_weights,
                      H)  # multiply H by the attention weights; (the resulting matrix is the sentence embedding) step

        # flatten M; the (encode a variable length sentence into a fixed size embedding) step.
        fixed_size_embedding = M.view(-1, M.size()[1] * M.size()[2])  # (batch, seq_len * hidden_size)

        x = self.fc_layer(fixed_size_embedding)

        if self.use_features:
            x = self.output(x) * 0.5
            x_2 = nn.functional.leaky_relu(self.f_Linear(features), .001)
            x_2 = self.f_output(x_2) * 0.5
            final_x = x + x_2
            return final_x
        else:
            return self.output(x)

def l1_penalty(parm, l1_lambda=0.0001):
   l1_norm = sum(p.abs().sum() for p in parm)
   return l1_norm * l1_lambda


def train(model, iterator, optimizer):
    epoch_loss = 0
    model.train()
    all_pred = []
    all_y = []
    l1_reg = torch.tensor(0.).to(device)
    l2_reg = torch.tensor(0.).to(device)
    for batch in iterator:
        optimizer.zero_grad()
        X = batch['sequence']
        y = batch['target']
        pos = batch['pos']
        batch_size = len(y)
        h = model.init_hidden(batch_size)

        pred = model(X, pos, h).squeeze()

        for param in model.parameters():
        #    l1_reg += l1_penalty(param)
            l2_reg += torch.norm(param, 1) ** 2

        loss_fun = nn.BCEWithLogitsLoss()(pred, y) + l2_reg.detach().cpu() #+ l1_reg.detach().cpu()
        loss_fun.backward()
        torch.cuda.empty_cache()
        all_pred.append(pred.squeeze().detach().cpu())
        all_y.append(y.detach().cpu())

        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        epoch_loss += loss_fun.item()
    total_epoch_loss = epoch_loss / len(iterator)
    total_epoch_score = metrics.roc_auc_score(np.vstack(all_y), np.vstack(all_pred))
    return total_epoch_loss, total_epoch_score

def val(model, iterator, return_pred=False):
    epoch_loss = 0
    all_pred = []
    all_y = []

    model.eval()

    for batch in iterator:
        X = batch['sequence']
        y = batch['target']
        pos = batch['pos']

        batch_size = len(y)
        h = model.init_hidden(batch_size)
        with torch.no_grad():
            pred = model(X, pos, h).squeeze()
            loss_fun = nn.BCEWithLogitsLoss()(pred, y)
            torch.cuda.empty_cache()

        epoch_loss += loss_fun.item()
        all_pred.append(pred.squeeze().detach().cpu())
        all_y.append(y.detach().cpu())

    total_epoch_loss = epoch_loss / len(iterator)
    total_epoch_score = metrics.roc_auc_score(np.vstack(all_y), np.vstack(all_pred))

    if return_pred:
        return total_epoch_loss, total_epoch_score, np.vstack(all_pred)
    else:
        return total_epoch_loss, total_epoch_score


def collate_fn_padded_test(batch):
    sequence = [b['sequence'] for b in batch]
    pos = [b['pos'] for b in batch]

    # extract max_length
    max_length = max([len(b) for b in sequence])
    sequence = pad_text(sequence, max_length)

    '''max_length = max([len(b) for b in pos])
    pos = pad_text(pos, max_length, pos=True)'''
    # convert list to torch.tensor
    return {'pos': torch.tensor(pos, dtype=torch.float, device=device),
            'sequence': torch.tensor(sequence, dtype=torch.long, device=device)}


def pred(model, iterator):
    model.eval()
    all_pred = []

    for batch in iterator:
        with torch.no_grad():
            X = batch['sequence']
            pos = batch['pos']
            batch_size = len(X)

            h = model.init_hidden(batch_size)
            pred = model(X, pos, h).squeeze()
            all_pred.append(pred.squeeze().cpu().detach().numpy())
    return all_pred


def run(model, train_loader, optimizer, val_loader, scheduler, fold, epochs=100):
    print('TRAINING STARTED...')
    train_loss_log = []
    val_loss_log = []

    train_score_log = []
    val_score_log = []
    epoch_log = []
    epoch_type_log = []

    best_val_score = 0
    total_start = time.time()
    for epoch in range(epochs):
        # --------------------------------------#
        start = time.time()
        early_stop = 0
        model.epoch = epoch + 1

        train_loss, train_score = train(model, train_loader, optimizer)
        val_loss, val_score = val(model, val_loader)
        scheduler.step(val_loss)

        end = time.time()
        # --------------------------------------#
        score = f'train_score={np.round(train_score, 3)} | val_score={np.round(val_score, 3)}'
        loss = f'train_loss={np.round(train_loss, 4)} | val_loss={np.round(val_loss, 4)}'

        train_loss_log.append(np.round(train_loss, 4))
        val_loss_log.append(np.round(val_loss, 4))

        train_score_log.append(np.round(train_score, 4))
        val_score_log.append(np.round(val_score, 4))

        epoch_log.append(epoch + 1)
        epoch_type = 'epoch'
        if val_score > best_val_score:
            epoch_type = 'BEST_EPOCH'
        epoch_type_log.append(epoch_type)

        log = pd.DataFrame({'train_loss_log': train_loss_log, 'val_loss_log': val_loss_log,
                            'train_score_log': train_score_log, 'val_score_log': val_score_log, 'epoch_log': epoch_log,
                            'epoch_type_log': epoch_type_log})
        log.to_csv(f'run files/log_fold_{fold}.csv', index=False)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        for f in os.listdir('run files'):
            if 'current' in f:
                if str(epoch + 1) not in f:
                    os.remove(f'run files/{f}')
        torch.save(state, f'run files/current_model_{np.round(val_score, 2)}_epoch_{epoch + 1}_fold_{fold}.pt')

        if val_score > best_val_score:
            best_val_score = val_score
            for f in os.listdir('run files'):
                if 'best' in f:
                    os.remove(f'run files/{f}')
            torch.save(model.state_dict(), f'run files/best_model_{np.round(val_score, 2)}_fold_{fold}.pt')

            print(
                f'BEST EPOCH({epoch + 1}) time=({np.round(int(end - start) / 60 / 60, 2)})hours: loss=({loss}) | score=({score})')

        else:
            print(
                f'epoch({epoch + 1}) time=({np.round(int(end - start) / 60 / 60, 2)})hours: loss=({loss}) | score=({score})')
            early_stop += 1
        if early_stop == 10:
            break
    total_end = time.time()
    print(f'TRAINING time={np.round(int(total_end - total_start) / 60 / 60, 2)}')
