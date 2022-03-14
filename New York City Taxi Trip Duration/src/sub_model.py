import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, config
print
# sub_0 = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_1_pred_0.393_last.csv')
# sub_1 = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_1_pred_0.2076.csv')
# sub_2 = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_2_pred_0.2071.csv')
# sub_3 = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_3_pred_0.2067.csv')
# sub_4 = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\pythonProject\taxi_trip\inputs\sub_4_pred_0.2062.csv')
# average_sub = np.concatenate([sub_0, sub_1, sub_2, sub_3, sub_4], axis=1).mean(axis=1)

test = pd.read_csv(config.processed_TEST_FILE).drop(['passenger_count', 'pickup_longitude', 'pickup_latitude'], axis=1)
test = test.values.reshape((test.shape[0], 1, 4, 4))


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=252, kernel_size=(2, 2)):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=128, kernel_size=kernel_size)
        x = torch.zeros((2, 1, 4, 4))
        shape = self.convs(x)

        self.shape = shape.size()[1] * shape.size()[2] * shape.size()[3]
        self.dense = nn.Linear(self.shape, 128)
        self.BatchNorm1d = nn.BatchNorm1d(128)
        self.Dropout = nn.Dropout(.5)
        self.out = nn.Linear(128, 1)

    def convs(self, x):
        x = F.leaky_relu(self.conv_1(x), 0.0001)
        # x = F.leaky_relu(self.conv_2(x), 0.0001)
        return x
    def forward(self, x):
        x = self.convs(x).view((-1, self.shape))
        x = self.Dropout(x)
        x = F.leaky_relu(self.dense(x), 0.001)
        x = self.BatchNorm1d(x)
        x = self.out(x)
        return x
sub = pd.read_csv('../inputs/sample_submission.csv')

print(test.shape)
def make_pred(model, data, file_name='cnn_model.pt'):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test), 100):
            pred = torch.exp(model(torch.tensor(data[i: i+100], dtype=torch.float)))
            predictions.append(pred)
    return np.vstack(predictions)

sub['trip_duration'] = make_pred(CNN(), test, file_name='cnn_model.pt')
print(sub)

sub.to_csv('submission_cnn.csv', index=False)
