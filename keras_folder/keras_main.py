import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle

import numpy as np
from keras.callbacks import (EarlyStopping)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score
import h5py

from keras_folder.cnn_model import  Config, get_oleg_model
from keras_folder.batcher import DummyBatcher
import tensorflow as tf



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

    # = stackoverflow snippet for launching GPU ==================
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # ============================================================


    batch_size = 64
    cur_feature_dimension = 1
    feat_t = False
    show_train_info = 1



    config = Config(shape=(16000, 1), lr=0.001, num_epochs=20, n_classes=10)
    model = get_oleg_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8), gpu_lstm=True)
    print(model.summary())
    input()


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

    with open (r"C:\Users\kotov-d\Documents\TASKS\task#7\model.pkl","wb") as f:
        pickle.dump(model, f)

    # start prediction
    with open (r"C:\Users\kotov-d\Documents\TASKS\task#7\model.pkl","rb") as f:
        model = pickle.load(f)

    batcher_test = DummyBatcher(config, batch_size, x_test)

    preds = model.predict_generator(batcher_test)

    print('***************')
    print('Unweighted accuracy:', accuracy_score(y_test, preds))
    print('Weighted accuracy  :', recall_score(y_test, preds, average='macro'))
    print('f1-score           :', f1_score(y_test, preds, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test, preds))