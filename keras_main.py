import warnings

from opt_einsum.backends import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler)
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
import h5py

from cnn_model import  Config, linear_decay_lr, get_oleg_model, get_oleg_model_2d, get_seresnet18_model
from batcher import DummyBatcher
from prepare_datasets import rectify_data
import tensorflow as tf



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




    batcher_train = DummyBatcher(config, batch_size, x_train, y_train)
    batcher_val = DummyBatcher(config, batch_size, x_val, y_val)

    my_callback = EarlyStopping(patience=5)
    history = model.fit_generator(batcher_train, validation_data=batcher_val, epochs=config.num_epochs,
                                  use_multiprocessing=False, verbose=show_train_info, callbacks=[my_callback])

    # start prediction
    batcher_test = DummyBatcher(config, batch_size, x_test)

    preds = model.predict_generator(batcher_test)

    print('***************')
    print('Unweighted accuracy:', accuracy_score(y_test, preds))
    print('Weighted accuracy  :', recall_score(y_test, preds, average='macro'))
    print('f1-score           :', f1_score(y_test, preds, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test, preds))
