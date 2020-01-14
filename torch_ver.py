import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
import h5py

from cnn_model import  Config
from prepare_datasets import rectify_data
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":
# загрузка посчитанных данных
    base_path = r'C:\Users\preductor\Documents\dummy_files'
    train_meta_path = os.path.join(base_path, 'meta_train.csv')
    test_meta_path = os.path.join(base_path, 'meta_test.csv')

    # подсчитать test и train выборки
    test = rectify_data(base_path, test_meta_path)
    train = rectify_data(base_path, train_meta_path)



    batch_size = 64
    cur_feature_dimension = 1
    feat_t = False
    show_train_info = 1



    config = Config(shape=(16000, 1), lr=0.001, num_epochs=20, n_classes=10)
    # заменить модель на торчовый аналог
    model = get_oleg_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8), gpu_lstm=False)


    (N,W) = train.shape

    np.random.shuffle(train)

    x_train = train[:int(0.8*N), :16000]
    y_train = train[:int(0.8*N), -1]
    x_val = train[int(0.8*N):, :16000]
    y_val = train[int(0.8*N):, -1]
    x_test = test[:, :16000]
    y_test = test[:, -1]

    print('=================================================================\n=================================================================')




    dataloader = DataLoader(data, batch_size=4,
                            shuffle=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())

