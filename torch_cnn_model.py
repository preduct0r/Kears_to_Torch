import torch
from torch.nn import Sequential
from torch.nn import functional as F
from torch import nn


class Config:
    def __init__(self, shape, lr, n_classes, num_epochs=500, dr = 0.1):
        self.shape = shape
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes



class torch_model(nn.Module):
    def __init__(self, config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8), gpu_lstm=True):
        super(torch_model, self).__init__()
        self.config = config
        self.fc1 = nn.Conv1d(in_channels= config.shape[1], out_channels=8, kernel_size=k_size[0], stride=1)
        self.bn_1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(kernel_size=p_size[0])

        self.fc2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=k_size[1], stride=1)
        self.bn_2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool1d(kernel_size=p_size[1])

        self.fc3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=k_size[2], stride=1)
        self.bn_3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool1d(kernel_size=p_size[2])

        self.fc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k_size[3], stride=1)
        self.bn_4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool1d(kernel_size=p_size[3])

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2)
        self.linear = nn.Linear(64, )


    def forward(self, x):
        out = self.fc1(x)
        out = self.bn_1(out)
        out = self.relu1(out)
        out = self.mp1(out)

        out = self.fc2(x)
        out = self.bn_2(out)
        out = self.relu2(out)
        out = self.mp2(out)

        out = self.fc3(x)
        out = self.bn_3(out)
        out = self.relu3(out)
        out = self.mp3(out)

        out = self.fc4(x)
        out = self.bn_4(out)
        out = self.relu4(out)
        out = self.mp4(out)

        out = self.lstm(out)
        out = F.adaptive_max_pool1d(out.numpy().unsqueeze(0), output_size=1)
        out = F.dropout(out, p=self.config.dr)
        out = nn.Linear()
        return out