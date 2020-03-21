import warnings
warnings.simplefilter(action='ignore')
import os
import pickle
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
import torch
from torch.autograd import Variable
import torch.cuda as cuda
import matplotlib
from matplotlib import pyplot as plt
import time

from prepare_datasets import rectify_data
from torch_cnn_model import  Config, torch_model, EarlyStopping, My_Dataset, Batcher
from prepare_datasets import get_raw_data
from sampler import BalancedBatchSampler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path



def train_net(base_path, size=32000, n_classes=4):
    experiments_path = os.path.join(r'C:\Users\kotov-d\Documents\TASKS\cross_inference', os.path.basename(base_path))
    Path(experiments_path).mkdir(parents=True, exist_ok=True)

    [x_train, x_val, x_test, y_train, y_val, y_test] = get_raw_data(base_path, experiments_path, n_classes, size=16000)
    # x_train, y_train = balance_classes(x_train, y_train)

    config = Config(lr=0.001, batch_size=512, num_epochs=1000, n_classes=n_classes)
    net = torch_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8))

    if cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    batcher_train = DataLoader(My_Dataset(x_val, y_val), batch_size=config.batch_size)
    # batcher_train = Batcher(x_train, y_train, batch_size=config.batch_size)
    batcher_val = DataLoader(My_Dataset(x_train, y_train), batch_size=config.batch_size,
                               )
    start_time = time.time()

    train_loss = []
    valid_loss = []
    train_fscore = []
    valid_fscore = []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    net.to(device)
    early_stopping = EarlyStopping()
    min_loss = 1000

    for epoch in range(config.num_epochs):
        iter_loss = 0.0
        correct = 0
        f_scores= 0
        iterations = 0

        net.train()

        for i, (items, classes) in enumerate(batcher_train):
        # for i in range(x_train.shape[0]//config.batch_size):
            # items,classes = batcher_train.get()

            items = items.to(device)
            classes = classes.to(device)
            optimizer.zero_grad()
            outputs = net(items)
            loss = criterion(outputs, classes.long())
            iter_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum()

            f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

            iterations += 1

            torch.cuda.empty_cache()

        train_loss.append(iter_loss / iterations)
        train_fscore.append(f_scores / iterations)

        early_stopping.update_loss(train_loss[-1])
        if early_stopping.stop_training():
            break

        ############################
        # Validate
        ############################
        iter_loss = 0.0
        correct = 0
        f_scores = 0
        iterations = 0

        net.eval()  # Put the network into evaluate mode

        for i, (items, classes) in enumerate(batcher_val):

            items = items.to(device)
            classes = classes.to(device)

            outputs = net(items)
            loss = criterion(outputs, classes.long())
            iter_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum()

            f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

            iterations += 1

        valid_loss.append(iter_loss / iterations)
        valid_fscore.append(f_scores / iterations)

        if valid_loss[-1] < min_loss:
            torch.save(net, os.path.join(experiments_path, "net_unbalanced_{}cls.pb".format(n_classes)))
            min_loss = valid_loss[-1]

        print('Epoch %d/%d, Tr Loss: %.4f, Tr Fscore: %.4f, Val Loss: %.4f, Val Fscore: %.4f'
              % (epoch + 1, config.num_epochs, train_loss[-1], train_fscore[-1],
                 valid_loss[-1], valid_fscore[-1]))

    with open(os.path.join(experiments_path,"loss_track_{}cls.pkl".format(n_classes)), 'wb') as f:
        pickle.dump([train_loss, train_fscore, valid_loss, valid_fscore], f)

    print(time.time()-start_time)


# train_net(r'C:\Users\kotov-d\Documents\BASES\friends')
# train_net(r'C:\Users\kotov-d\Documents\BASES\iemocap_last')
train_net(r'C:\Users\kotov-d\Documents\BASES\RAMAS\ramas')
# train_net(r'C:\Users\kotov-d\Documents\BASES\telecom_vad')


