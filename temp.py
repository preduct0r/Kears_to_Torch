import pickle
import pandas as pd
import os
from matplotlib import pyplot as plt
from glob import glob
import torch
import os
from prepare_datasets import get_raw_data
from torch_cnn_model import  Config, torch_model, EarlyStopping, My_Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader



def plot_loss(experiments_path, n_classes):
    data_path = os.path.join(experiments_path, "loss_track_{}cls.pkl".format(n_classes))

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    [train_loss, train_fscore, valid_loss, valid_fscore] = data

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



def evaluate_test(experiments_path, x, y, n_classes=4):
    net = torch.load(os.path.join(experiments_path, 'net_unbalanced_{}cls.pb'.format(n_classes)))
    batcher_test = DataLoader(My_Dataset(x, y), batch_size=512)

    ############################
    # Validate
    ############################
    loss = 0.0
    correct = 0
    iterations = 0
    f_scores = 0

    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    torch.manual_seed(7)

    for i, (items, classes) in enumerate(batcher_test):
        items = items.to('cuda')
        classes = classes.to('cuda')

        outputs = net(items)
        loss += criterion(outputs, classes.long()).item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data.long()).sum()

        f_scores += f1_score(predicted.cpu().numpy(), classes.data.cpu().numpy(), average='macro')

        iterations += 1

    print(loss / iterations)
    print(f_scores / iterations)


base_path = r'C:\Users\kotov-d\Documents\BASES\ramas'

experiments_path = os.path.join(r'C:\Users\kotov-d\Documents\TASKS\cross_inference', os.path.basename(base_path))
n_classes = 4
[x_train, x_val, x_test, y_train, y_val, y_test] = get_raw_data(base_path, experiments_path, n_classes)


# plot_loss(experiments_path, n_classes)
evaluate_test(experiments_path, x_test, y_test, n_classes)