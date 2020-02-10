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
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.cuda as cuda

from torch_batcher import My_Dataset
from prepare_datasets import rectify_data
from torch_cnn_model import  Config, torch_model



if __name__ == "__main__":
# загрузка посчитанных данных
    base_path = r'C:\Users\kotov-d\Documents\BASES\IEMOCAP\iemocap'
    train_meta_path = os.path.join(base_path, 'meta_train.csv')
    test_meta_path = os.path.join(base_path, 'meta_test.csv')

    # # подсчитать test и train выборки
    # test = rectify_data(base_path, test_meta_path)
    # train = rectify_data(base_path, train_meta_path)

    # загрузить уже подсчитанные test и train выборки
    hf_train= h5py.File(r'C:\Users\kotov-d\Documents\TASKS\ulma\x_train.h5', 'r')
    train = hf_train.get('x_train').value
    hf_train.close()

    hf_test = h5py.File(r'C:\Users\kotov-d\Documents\TASKS\ulma\x_test.h5', 'r')
    test = hf_test.get('x_test').value
    hf_test.close()




    batch_size = 64
    cur_feature_dimension = 1
    feat_t = False
    show_train_info = 1



    config = Config(shape=(16000, 1), lr=0.001, num_epochs=100, n_classes=3)
    net = torch_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8))
    # print(summary(net, torch.cuda.FloatTensor(1,16000)))
    # input()


    (N,W) = train.shape

    np.random.shuffle(train)

    x_train = train[:int(0.8*N), :16000]
    y_train = train[:int(0.8*N), -1]
    x_val = train[int(0.8*N):, :16000]
    y_val = train[int(0.8*N):, -1]
    x_test = test[:, :16000]
    y_test = test[:, -1]
    print('np.unique(y_train) ', np.unique(y_train, return_counts=True))
    print('np.unique(y_test) ', np.unique(y_test, return_counts=True))

    print('================================================\n\nStart learning')

    batcher_train = DataLoader(My_Dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    batcher_val = DataLoader(My_Dataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    # history = model.fit_generator(batcher_train, validation_data=batcher_val, epochs=config.num_epochs,
    #                               use_multiprocessing=False, verbose=show_train_info)

    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    train_fscore = []
    valid_fscore = []

    if cuda.is_available():
        net = net.cuda()

    # Our loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)

    for epoch in range(config.num_epochs):

        ############################
        # Train
        ############################

        iter_loss = 0.0
        correct = 0
        iterations = 0
        f_scores= 0

        net.train()  # Put the network into training mode

        for i, (items, classes) in enumerate(batcher_train):

            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)

            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()

            optimizer.zero_grad()  # Clear off the gradients from any past operation
            outputs = net(items)  # Do the forward pass
            loss = criterion(outputs, classes.long())                                               # Calculate the loss
            iter_loss += loss.item()  # Accumulate the loss
            loss.backward()  # Calculate the gradients with help of back propagation
            optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients

            # Record the correct predictions for training data
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum()

            f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

            iterations += 1

        # Record the training loss
        train_loss.append(iter_loss / iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / len(batcher_train.dataset)))
        train_fscore.append(f_scores/iterations)
        # print(100.0 * correct / len(batcher_train.dataset))                                                   рень с перестановкой множителей местами!

        ############################
        # Validate - How did we do on the unseen dataset?
        ############################


        loss = 0.0
        correct = 0
        iterations = 0
        f_scores = 0

        net.eval()  # Put the network into evaluate mode

        for i, (items, classes) in enumerate(batcher_val):

            # Convert torch tensor to Variable
            items = Variable(items)
            classes = Variable(classes)

            # If we have GPU, shift the data to GPU
            if cuda.is_available():
                items = items.cuda()
                classes = classes.cuda()

            outputs = net(items)  # Do the forward pass
            loss += criterion(outputs, classes.long()).item()  # Calculate the loss
                                                                            # loss = criterion(outputs, classes.long()).sum()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes.data.long()).sum()

            f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

            iterations += 1

        # Record the validation loss
        valid_loss.append(loss / iterations)
        # Record the validation accuracy
        valid_accuracy.append(100.0 * correct / float(len(batcher_val.dataset)))
        valid_fscore.append(f_scores/iterations)

        # print(correct, float(len(batcher_val.dataset)))
        # print(100 * correct / float(len(batcher_val.dataset)))


        print('Epoch %d/%d, Tr Loss: %.4f, Tr Fscore: %.4f, Val Loss: %.4f, Val Fscore: %.4f'
              % (epoch + 1, config.num_epochs, train_loss[-1], train_fscore[-1],
                 valid_loss[-1], valid_fscore[-1]))