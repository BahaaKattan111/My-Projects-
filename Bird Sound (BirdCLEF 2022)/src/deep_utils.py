import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import warnings, os, torchaudio
from sklearn import metrics
import cv2

device = 'cuda'
warnings.filterwarnings('ignore')
from librosa import display

import numpy as np

'''def comp_metric(y_true, y_pred, epsilon=1e-9):
    """ Function to calculate competition metric in an sklearn like fashion

    Args:
        y_true{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Ground truth (correct) target values.
        y_pred{array-like, sparse matrix} of shape (n_samples, n_outputs)
            - Estimated targets as returned by a classifier.
    Returns:
        The single calculated score representative of this competitions evaluation
    """

    # Get representative confusion matrices for each label
    mlbl_cms = sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred)

    # Get two scores (TP and TN SCORES)
    tp_scores = np.array([
        mlbl_cm[threshold=0.5, threshold=0.5]/(epsilon+mlbl_cm[:, threshold=0.5].sum()) \
        for mlbl_cm in mlbl_cms
        ])
    tn_scores = np.array([
        mlbl_cm[0, 0]/(epsilon+mlbl_cm[:, 0].sum()) \
        for mlbl_cm in mlbl_cms
        ])

    # Get average
    tp_mean = tp_scores.mean()
    tn_mean = tn_scores.mean()

    return round((tp_mean+tn_mean)/2, 8)'''
from sklearn.metrics import jaccard_score


def macro_jaccard_index(pred, target, threshold=0.05):
    pred = pred >= threshold
    scores = []
    scores += [jaccard_score(target[:, i], pred[:, i], pos_label=1) for i in range(pred.shape[1])]
    scores += [jaccard_score(target[:, i], pred[:, i], pos_label=0) for i in range(pred.shape[1])]
    return np.mean(scores)


class SoundDataset_TRAIN(Dataset):
    def __init__(self, data, labels={}, class_dict=None):
        self.data = data
        if class_dict is not None:
            self.labels = [class_dict[labels] for label in labels]  # in case class_dict exist
        else:
            self.labels = labels # it's one-hot-encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label, signal = self.labels.values[item], self.data[item]
        signal = F.dropout2d(torch.tensor(signal, dtype=torch.float), 0.1).cpu().numpy()
        sig = np.zeros((3, 128, 80))

        sig[0, :, :] = signal[0, :]
        sig[1, :, :] = signal[0, :]
        sig[2, :, :] = signal[0, :]

        return {'signals': torch.as_tensor(sig, dtype=torch.float).to(device),
                'target': torch.as_tensor(label, dtype=torch.long).to(device)}
class SoundDataset_VAL(Dataset):
    def __init__(self, data, labels={}, class_dict=None):

        self.data = data
        # mean -74.30503278616727
        # std 23.726224427065763
        self.labels = labels

        # self.labels = [class_dict[label] for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label, signal = self.labels.values[item], self.data[item]
        sig = np.zeros((3, 128, 80))

        sig[0, :, :] = signal[0, :]
        sig[1, :, :] = signal[0, :]
        sig[2, :, :] = signal[0, :]

        return {'signals': torch.as_tensor(sig, dtype=torch.float).to(device),
                'target': torch.as_tensor(label, dtype=torch.long).to(device)}



class CNNNetwork(nn.Module):
    def __init__(self, class_dict, hid_size=64, kernel_size=(2, 2)):
        super().__init__()
        # self.model = torchvision.models.alexnet().features[:9]
        # self.model.requires_grad_(False)
        self.epoch = 0
        #self.resnet = torchvision.models.vgg11(True).features[:10]
        self.resnet = torchvision.models.vgg11(True).features[:10]
        self.resnet.requires_grad_(False)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=hid_size, kernel_size=kernel_size, stride=(1, 1), padding=1,
                      bias=True),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(hid_size),
            nn.Conv2d(in_channels=hid_size, out_channels=hid_size, kernel_size=kernel_size, stride=(1, 1), padding=1,
                      bias=True),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(hid_size),
            nn.Conv2d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=kernel_size, stride=(1, 1),
                      padding=1, bias=True),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.flatten = nn.Flatten()
        x = torch.zeros((1, 3, 128, 80))
        self.shape = self.convs(x, True)
        self.linear = nn.Linear(self.shape, 500)
        self.linear_2 = nn.Linear(500, 500)
        self.out = nn.Linear(500, class_dict)

    def convs(self, x, shape=False):
        x = self.resnet(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if shape:
            x = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x):
        x = self.convs(x)
        '''x = x.view((-threshold=0.5, 128, 10*10))
        x = torch.permute(x, [0, 2, threshold=0.5])
        out, (hid, c) = self.lstm1(x)
        hid = torch.cat([hid[-threshold=0.5, :, :], hid[-2, :, :]], dim=threshold=0.5)'''
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear_2(x))
        predictions = self.out(x)

        return predictions

class CNNN_LSTM_network(nn.Module):
    def __init__(self, class_dict, hid_size=64, kernel_size=(2, 2)):
        super().__init__()
        # self.model = torchvision.models.alexnet().features[:9]
        # self.model.requires_grad_(False)
        # self.resnet = torchvision.models.vgg11(True).features[:10]
        # self.resnet.requires_grad_(False)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=hid_size, kernel_size=kernel_size, stride=(1, 1), padding=1,
                      bias=True),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(hid_size),
            nn.Conv2d(in_channels=hid_size, out_channels=hid_size, kernel_size=kernel_size, stride=(1, 1), padding=1,
                      bias=True),
            nn.LeakyReLU(0.001),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(hid_size),
            nn.Conv2d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=kernel_size, stride=(1, 1),
                      padding=1, bias=True),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(hid_size * 2),
            nn.Conv2d(in_channels=hid_size * 2, out_channels=hid_size * 2, kernel_size=kernel_size, stride=(1, 1),
                      padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(hid_size * 2),
            nn.Conv2d(in_channels=hid_size * 2, out_channels=hid_size * 3, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv6 = nn.Sequential(
            nn.BatchNorm2d(hid_size * 3),
            nn.Conv2d(in_channels=hid_size * 3, out_channels=hid_size * 3, kernel_size=(2, 2), stride=(1, 1),
                      padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.conv7 = nn.Sequential(
            nn.BatchNorm2d(hid_size * 3),
            nn.Conv2d(in_channels=hid_size * 3, out_channels=hid_size * 3, kernel_size=(2, 2), stride=(1, 1),
                      padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear_1 = nn.Linear(200, 500)
        self.linear_2 = nn.Linear(500, 500)
        self.out = nn.Linear(500, class_dict)

    def convs(self, x, shape=False):
        x = self.resnet(x)
        x = self.conv1(x)
        x = self.conv2(x)
        '''max_1 = F.max_pool2d(x, (2, 2))
        max_2 = F.max_pool2d(x, (3, 3))
        max_3 = F.max_pool2d(x, (4, 4))
        max_4 = F.max_pool2d(x, (5, 5))

        max_1 = max_1.view(max_1.shape[0],max_1.shape[threshold=0.5] * max_1.shape[2] * max_1.shape[3])
        max_2 = max_2.view(max_2.shape[0],max_2.shape[threshold=0.5] * max_2.shape[2] * max_2.shape[3])
        max_3 = max_3.view(max_3.shape[0],max_3.shape[threshold=0.5] * max_3.shape[2] * max_3.shape[3])
        max_4 = max_4.view(max_4.shape[0],max_4.shape[threshold=0.5] * max_4.shape[2] * max_4.shape[3])
        x = F.dropout(torch.cat([max_1, max_2, max_3, max_4], threshold=0.5), .4)'''
        x = self.conv3(x)
        if shape:
            x = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x):
        x = self.convs(x)
        '''x = x.view((-threshold=0.5, 128, 10*10))
        x = torch.permute(x, [0, 2, threshold=0.5])
        out, (hid, c) = self.lstm1(x)
        hid = torch.cat([hid[-threshold=0.5, :, :], hid[-2, :, :]], dim=threshold=0.5)'''
        s1, s2, s3, s4 = x.shape
        x = x.view(s1, s2, s3 * s4)

        out, (hid, c) = self.lstm1(x)
        hid = torch.cat([hid[-1, :, :], hid[-2, :, :]], dim=1)
        x = F.tanh(self.linear_1(hid))
        predictions = self.out(x)

        return predictions


class C_LSTM(nn.Module):

    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super().__init__()
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.conv2d = nn.Conv2d(1, kernel_num, (3, embed_dim))
        self.dropout = nn.Dropout(.2)
        self.lstm1 = nn.LSTM(kernel_num, 100, batch_first=True, bidirectional=True, num_layers=1)

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
        # out_, (hid_, c_) = self.lstm2(torch.cat([out, x], 2))
        # hid = F.dropout(torch.cat([hid, hid_], dim=2), 0.6)
        # hid = torch.cat([hid[-threshold=0.5, :, :], hid[-2, :, :]], dim=threshold=0.5)

        # out = F.dropout(torch.cat([o for o in hid], dim=threshold=0.5), 0.6)
        # x =F.max_pool1d(hid, 3)

        x = F.elu(self.hidden_1(out))

        x = self.out(x)
        print(x)
        return x


def train(model, iterator, optimizer):
    model.train()
    epoch_loss = 0
    all_pred = []
    all_y = []
    grad_scaler = torch.cuda.amp.GradScaler()

    # l1_reg = torch.tensor(0.).to(device)
    # l2_reg = torch.tensor(0.).to(device)
    w = 0
    for batch in iterator:

        w += 1
        optimizer.zero_grad()
        X = batch['signals']
        y = batch['target']
        with torch.cuda.amp.autocast():
            pred = model(X)

        # for param in model.parameters():
        #    l1_reg += l1_penalty(param)
        #    l2_reg += torch.norm(param, threshold=0.5) ** 2
        loss_fun = nn.CrossEntropyLoss()(pred, y.argmax(1))  # + l2_reg.detach().cpu() #+ l1_reg.detach().cpu()
        grad_scaler.scale(loss_fun).backward()
        # loss_fun.backward()
        torch.cuda.empty_cache()
        # _, pred = pred.argmax(threshold=0.5)

        all_pred.append(F.softmax(pred, 1).cpu().detach().numpy())  # .view(-threshold=0.5)
        all_y.append(y.cpu().detach().numpy())  # .view(-threshold=0.5)

        nn.utils.clip_grad_norm(model.parameters(), 5)
        if w / 5 == round(w / 5):
            grad_scaler.step(optimizer)
            # optimizer.step()
            grad_scaler.update()
        epoch_loss += loss_fun.item()
    torch.cuda.empty_cache()

    total_epoch_loss = epoch_loss / len(iterator)

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    pred = pred.argmax(1)
    y = y.argmax(1)

    total_epoch_accuracy_score = metrics.accuracy_score(y, pred),  # average='macro')
    total_epoch_f1_score = metrics.f1_score(y, pred, average='macro')
    return total_epoch_loss, total_epoch_f1_score, total_epoch_accuracy_score


def val(model, iterator, return_pred=False):
    epoch_loss = 0
    all_pred = []
    all_y = []

    model.eval()

    for batch in iterator:
        X = batch['signals']
        y = batch['target']

        with torch.no_grad():
            pred = model(X)
            loss_fun = nn.CrossEntropyLoss()(pred, y.argmax(1))
            torch.cuda.empty_cache()

        epoch_loss += loss_fun.item()
        all_pred.append(F.softmax(pred, 1).detach().cpu())
        all_y.append(y.detach().cpu())
    torch.cuda.empty_cache()

    total_epoch_loss = epoch_loss / len(iterator)

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)

    y = y.argmax(1)
    pred = pred.argmax(1)

    # total_epoch_accuracy_score = metrics.accuracy_score(y, pred),  # average='macro')
    total_epoch_accuracy_score = metrics.accuracy_score(y, pred),  # average='macro')
    total_epoch_f1_score = metrics.f1_score(y, pred, average='macro')

    if return_pred:
        return total_epoch_loss, total_epoch_f1_score, total_epoch_accuracy_score, pred

    else:
        return total_epoch_loss, total_epoch_f1_score, total_epoch_accuracy_score


def run(model, train_loader, optimizer, val_loader, scheduler, fold, epochs=250, checkpoint=None, best_f1_score=None):
    import time, os
    print('TRAINING STARTED...')
    if 'model_1' not in os.listdir():
        os.mkdir('run files')
    train_loss_log = []
    val_loss_log = []

    val_accuracy_score_log = []
    val_f1_score_log = []
    train_accuracy_score_log = []
    train_f1_score_log = []

    epoch_log = []
    epoch_type_log = []
    if best_f1_score:
        best_f1_score = best_f1_score
    else:
        best_f1_score = 0.
    total_start = time.time()
    if checkpoint:
        epochs_to_run = range(checkpoint, epochs)
    else:
        epochs_to_run = range(1, epochs)

    for epoch in epochs_to_run:
        # --------------------------------------#
        start = time.time()
        early_stop = 0
        model.epoch = epoch

        train_loss, train_f1_score, train_accuracy_score = train(model, train_loader, optimizer)
        val_loss, val_f1_score, val_accuracy_score = val(model, val_loader)
        scheduler.step(val_loss)

        end = time.time()
        # --------------------------------------#
        accuracy = f'train={np.round(train_accuracy_score, 3)} | val={np.round(val_accuracy_score, 3)}'
        f1_score = f'train={np.round(train_f1_score, 3)} | val={np.round(val_f1_score, 3)}'
        loss = f'train={np.round(train_loss, 4)} | val={np.round(val_loss, 4)}'

        train_loss_log.append(np.round(train_loss, 4))
        val_loss_log.append(np.round(val_loss, 4))

        val_accuracy_score_log.append(np.round(val_accuracy_score, 3))
        val_f1_score_log.append(np.round(val_f1_score, 3))

        train_accuracy_score_log.append(np.round(train_accuracy_score, 3))
        train_f1_score_log.append(np.round(train_f1_score, 3))

        epoch_log.append(epoch)
        epoch_type = 'epoch'
        if val_f1_score > best_f1_score:
            epoch_type = 'BEST_EPOCH'
        epoch_type_log.append(epoch_type)
        log = pd.DataFrame({'train_loss_log': train_loss_log, 'val_loss_log': val_loss_log,
                            'train_accuracy_log': train_accuracy_score_log, 'val_accuracy_log': val_accuracy_score_log,
                            'train_f1_score_log': train_f1_score_log, 'val_f1_score_log': val_f1_score_log,
                            'epoch_log': epoch_log, 'epoch_type_log': epoch_type_log})
        log.to_csv(f'model_1/log_fold_{fold}.csv', index=False)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        for f in os.listdir('run files'):
            if 'current' in f:
                if str(epoch) not in f:
                    os.remove(f'run files/{f}')
        torch.save(state, f'run files/current_model_{np.round(val_f1_score, 2)}_epoch_{epoch}_fold_{fold}.pt')

        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            for f in os.listdir('run files'):
                if 'best' in f:
                    os.remove(f'run files/{f}')
            torch.save(model.state_dict(), f'run files/best_model_{np.round(best_f1_score, 2)}_fold_{fold}.pt')

            print(
                f'BEST EPOCH({epoch}) time=({np.round(int(end - start) / 60 / 60, 2)})hours: loss=({loss}) | F1-score=({f1_score}) | Accuracy=({accuracy})')

        else:
            print(
                f'epoch({epoch}) time=({np.round(int(end - start) / 60 / 60, 2)})hours: loss=({loss}) | F1-score=({f1_score}) | Accuracy=({accuracy})')
            early_stop += 1
        if early_stop == 10:
            break
    total_end = time.time()
    print(f'TRAINING time={np.round(int(total_end - total_start) / 60 / 60, 2)}')
