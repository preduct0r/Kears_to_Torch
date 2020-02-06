import torch
from torch.nn import Sequential
from torch.nn import functional as F
from torch import nn
import numpy as np

class Config:
    def __init__(self, shape, lr, n_classes, num_epochs=500, dr = 0.1):
        self.shape = shape
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.dr = dr



class torch_model(nn.Module):
    def __init__(self, config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8)):
        super(torch_model, self).__init__()
        self.config = config                                                                                        # 64*1*16000
        self.fc1 = nn.Conv1d(in_channels= config.shape[1], out_channels=8, kernel_size=k_size[0], stride=1)         # 64*8*15937
        self.bn_1 = nn.BatchNorm1d(8)                                                                               # 64*8*15937
        self.relu1 = nn.ReLU()                                                                                      # 64*8*15937
        self.mp1 = nn.MaxPool1d(kernel_size=p_size[0])                                                              # 64*8*5312

        self.fc2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=k_size[1], stride=1)                       # 64*16*5281
        self.bn_2 = nn.BatchNorm1d(16)                                                                              # 64*16*5281
        self.relu2 = nn.ReLU()                                                                                      # 64*16*5281
        self.mp2 = nn.MaxPool1d(kernel_size=p_size[1])                                                              # 64*16*1760

        self.fc3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=k_size[2], stride=1)                      # 64*32*1745
        self.bn_3 = nn.BatchNorm1d(32)                                                                              # 64*32*1745
        self.relu3 = nn.ReLU()                                                                                      # 64*32*1745
        self.mp3 = nn.MaxPool1d(kernel_size=p_size[2])                                                              # 64*32*581

        self.fc4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k_size[3], stride=1)                      # 64*64*574
        self.bn_4 = nn.BatchNorm1d(64)                                                                              # 64*64*574
        self.relu4 = nn.ReLU()                                                                                      # 64*64*574
        self.mp4 = nn.MaxPool1d(kernel_size=p_size[3])                                                              # 64*64*191


        self.lstm = nn.LSTM(191, 128, bias=True)                                                                    # 64*64*128
        self.gap = nn.AdaptiveAvgPool1d(output_size = 1)                                                            # 64*64*1
        self.linear = nn.Linear(64, self.config.n_classes)                                                         #


    def forward(self, x):
        out = self.fc1(x)
        out = self.bn_1(out)
        out = self.relu1(out)
        out = self.mp1(out)

        out = self.fc2(out)
        out = self.bn_2(out)
        out = self.relu2(out)
        out = self.mp2(out)

        out = self.fc3(out)
        out = self.bn_3(out)
        out = self.relu3(out)
        out = self.mp3(out)

        out = self.fc4(out)
        out = self.bn_4(out)
        out = self.relu4(out)
        out = self.mp4(out)


        out = self.lstm(out)
        out = self.gap(out[0])
        out = F.dropout(out, p=self.config.dr)
        out = out.view(-1, 64)

        out = self.linear(out)

        return out