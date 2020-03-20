import pickle
import pandas as pd
import os
from matplotlib import pyplot as plt
from glob import glob


data_path = r"C:\Users\kotov-d\Documents\TASKS\keras_to_torch\loss_track.pkl"

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

