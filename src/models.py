from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import time
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from torch.nn import functional as F
import config
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(config.processed_TRAINING_FILE).sample(frac=1, random_state=0).reset_index(drop=True)
data[f'pad_col_1'] = 0
data[f'pad_col_2'] = 0
scaler = StandardScaler()

print(len(data.columns))
'''
for i in range(12, 16):
    data[f'pad_col_{i}'] = 0'''

data['kfold'] = -1
kfold = KFold()
for i, (tr_, val_) in enumerate(kfold.split(data)):
    data.loc[val_, 'kfold'] = i

fold = 1
def data_loader(data, fold, batch=32):
    train = data.loc[data['kfold'] != fold]
    val = data.loc[data['kfold'] == fold]

    X_train = train.drop(['kfold', 'trip_duration'], axis=1).values
    X_train = torch.tensor(scaler.fit_transform(X_train).reshape((X_train.shape[0], 1, 5, 5)), dtype=torch.float)
    y_train = torch.tensor(train['trip_duration'].values, dtype=torch.float)

    # -----#
    X_val = val.drop(['kfold', 'trip_duration'], axis=1).values
    X_val = torch.tensor(scaler.transform(X_val).reshape((X_val.shape[0], 1, 5, 5)), dtype=torch.float)

    y_val = torch.tensor(val['trip_duration'].values, dtype=torch.float)

    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)

    return DataLoader(train_set, batch_size=batch), DataLoader(val_set, batch_size=batch)


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=252, kernel_size=(4, 4)):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=500, kernel_size=(2, 2))
        x = torch.zeros((2, 1, 5, 5))
        shape = self.convs(x)

        self.shape = shape.size()[1] * shape.size()[2] * shape.size()[3]
        self.dense = nn.Linear(self.shape, 250)

        self.BatchNorm1d = nn.BatchNorm1d(250)
        self.Dropout = nn.Dropout(.3)
        self.out = nn.Linear(250, 1)

    def convs(self, x):
        x = F.relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x), 0.00001)
        return x
    def forward(self, x):
        x = self.convs(x).view((-1, self.shape))
        x = self.Dropout(x)
        x = F.leaky_relu(self.dense(x), 0.00001)
        x = self.BatchNorm1d(x)
        x = self.out(x)
        return x
import config

model = CNN()

optimizer = optim.SGD(model.parameters(), 0.00001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01)

loss = nn.MSELoss()
from sklearn.metrics import mean_squared_error as mse
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    return np.sqrt(mse(y_true, y_pred))


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        X = batch[0]
        y = batch[1]

        pred = model(X).squeeze(1)

        loss_fun = criterion(pred, y)
        loss_fun.backward()

        acc = rmsle(pred.detach().numpy(), y.detach().numpy())

        optimizer.step()
        epoch_loss += loss_fun.item()
        epoch_acc += acc
    scheduler.step()
    total_epoch_loss = epoch_loss / len(iterator)
    total_epoch_acc = epoch_acc / len(iterator)

    return total_epoch_loss, total_epoch_acc


def val(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    for batch in iterator:
        X = batch[0]
        y = batch[1]
        with torch.no_grad():
            pred = model(X).squeeze(1)


        loss_fun = criterion(pred, y)

        acc = rmsle(pred.detach().numpy(), y.detach().numpy())

        epoch_loss += loss_fun.item()
        epoch_acc += acc

    total_epoch_loss = epoch_loss / len(iterator)
    total_epoch_acc = epoch_acc / len(iterator)

    return total_epoch_loss, total_epoch_acc


def run(model, train_loader, optimizer, loss, val_loader, epochs=100):
    print('TRAINING STARTED...')
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # --------------------------------------#
        start = time.time()
        train_loss, train_loss_ = train(model, train_loader, optimizer, loss)
        val_loss, val_loss_ = val(model, val_loader, loss)
        end = time.time()
        # --------------------------------------#
        mse = f'train_loss={np.round(train_loss, 4)} | val_loss={np.round(val_loss, 4)}'
        rmsle = f'train_loss={np.round(train_loss_, 4)} | val_loss={np.round(val_loss_, 4)}'

        if val_loss_ < best_val_loss:
            best_val_loss = val_loss_
            torch.save(model.state_dict(), 'cnn_model_max_pool.pt')
            print(
                f'BEST EPOCH({epoch + 1}) time=({np.round(int(end-start) / 60 / 60, 2)})hours: MSE=({mse}) | RMSLE=({rmsle})')
        else:
            print(f'epoch({epoch + 1}) time=({np.round(int(end-start) / 60 / 60, 2)})hours: MSE=({mse}) | RMSLE=({rmsle})')

def make_pred(model, data, file_name='cnn_model.pt'):
    model = model.load_state_dict(torch.load(file_name))
    model.eval()
    with torch.no_grad():
        pred = torch.exp(model(data)).item()
    return pred


train_loader, val_loader = data_loader(data, fold, batch=100)
run(model, train_loader, optimizer, loss, val_loader)










