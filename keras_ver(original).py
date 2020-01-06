import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler)
from keras.utils import Sequence, to_categorical
from librosa.core import load

from cnn_model import  Config, linear_decay_lr, get_oleg_model, get_oleg_model_2d, get_seresnet18_model
from trainutils import f1
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score

# Batcher ==============================================================================================================
# ======================================================================================================================
class DummyBatcher(Sequence):
    def __init__(self, config, batch_size, x, y=None, offset=True,
                 repeat_test=10, feature_dim=2, transpose_feature=True):
        self.config = config
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.indexes = np.arange(len(x))
        np.random.shuffle(self.indexes)
        self.offset = offset
        self.repeat_test = repeat_test
        self.feature_dim = feature_dim
        self.transpose_feature = transpose_feature

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if self.y is not None:
            indexes_tmp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            res_x = []
            for i in indexes_tmp:
                res_x.append(self.prepare_feature(self.x[i]))
            res_x = np.asarray(res_x)
            res_y = to_categorical(self.y, self.config.n_class)[indexes_tmp]
            return res_x, res_y
        else:
            indexes_tmp = np.arange(len(self.x))[index * self.batch_size:(index + 1) * self.batch_size]
            res_x = []
            for i in indexes_tmp:
                for j in range(self.repeat_test):
                    res_x.append(self.prepare_feature(self.x[i]))
            res_x = np.asarray(res_x)
            return res_x

# Feature loader (raw and prepared the same way) =======================================================================
# ======================================================================================================================
# TODO: доставать имена папок, extra-labels и мета-файлов из base_description.yml

def load_feature(base_path, feature_type, train_meta, test_meta, extra_labels, sr=8000):
    feature_dir = os.path.join(base_path, 'feature', feature_type)

    if feature_type == 'raw':

        def collect_raw_x(wav_dir, extra_labels, meta, sr):
            df = pd.read_csv(meta, sep=';')
            x = []
            y = []
            f = []
            f_and_labels = [df.cur_name, df.cur_label]
            for e_l in extra_labels:
                f_and_labels.append(df[e_l])
            for i in range(len(f_and_labels[0])):
                file = f_and_labels[0][i]
                label = [f_and_labels[1][i]]
                for j in range(2, len(f_and_labels)):
                    label.append(f_and_labels[j][i])
                wav_data, _ = load(os.path.join(wav_dir, file), sr=sr)
                x.append(wav_data)
                y.append(label)
                f.append(file)
            return x, y, f

        x_train, y_train, f_train = collect_raw_x(os.path.join(base_path, 'data'),
                                                  extra_labels,
                                                  os.path.join(base_path, train_meta), sr)

        x_test, y_test, f_test = collect_raw_x(os.path.join(base_path, 'data'),
                                               extra_labels,
                                               os.path.join(base_path, test_meta), sr)

    else:
        with open(os.path.join(feature_dir, 'x_train.pkl'), 'rb') as f:
            x_train = pickle.load(f)

        with open(os.path.join(feature_dir, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)

        with open(os.path.join(feature_dir, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)

        with open(os.path.join(feature_dir, 'f_train.pkl'), 'rb') as f:
            f_train = pickle.load(f)

        with open(os.path.join(feature_dir, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)

        with open(os.path.join(feature_dir, 'f_test.pkl'), 'rb') as f:
            f_test = pickle.load(f)

    return x_train, y_train, f_train, x_test, y_test, f_test

# ======================================================================================================================
# ======================================================================================================================
base_path = r'D:\emo_bases\cmu_mosei'
feature_type = 'raw'
sr = 8000
train_meta = 'meta_train.csv'
test_meta = 'meta_test.csv'

# IEMOCAP current extra labels
# extra_labels = ['valence', 'activation', 'dominance', 'sess_id']
extra_labels = []


x_train_m, y_train_m, f_train_m, x_test_m, y_test_m, f_test_m = load_feature(base_path,
                                                                 feature_type,
                                                                 train_meta,
                                                                 test_meta,
                                                                 extra_labels,
                                                                 sr)

# ======================================================================================================================
# ======================================================================================================================
base_path = r'D:\emo_bases\iemocap'
feature_type = 'raw'
sr = 8000
train_meta = 'meta_train.csv'
test_meta = 'meta_test.csv'

# IEMOCAP current extra labels
extra_labels = ['valence', 'activation', 'dominance', 'sess_id']
# extra_labels = []

x_train, y_train, f_train, x_test, y_test, f_test = load_feature(base_path,
                                                                 feature_type,
                                                                 train_meta,
                                                                 test_meta,
                                                                 extra_labels,
                                                                 sr)

# ======================================================================================================================
# ======================================================================================================================
# filter MOSEI and add to IEMOCAP
mosei_emos = ['anger', 'happiness', 'sadness']
to_iemo = {'anger': 'ang', 'happiness': 'hap', 'sadness': 'sad'}

for x, y, f in zip(x_train_m, y_train_m, f_train_m):
    if y[0] not in mosei_emos:
        continue
    x_train.append(x)
    y_train.append([to_iemo[y[0]]])
    f_train.append(f)

for x, y, f in zip(x_test_m, y_test_m, f_test_m):
    if y[0] not in mosei_emos:
        continue
    x_test.append(x)
    y_test.append([to_iemo[y[0]]])
    f_test.append(f)

# ======================================================================================================================
# ======================================================================================================================
top_freq = 100

for i, x in enumerate(x_train):
    x_train[i] = x[:top_freq, :].T

for i, x in enumerate(x_test):
    x_test[i] = x[:top_freq, :].T

# ======================================================================================================================
# ======================================================================================================================
for x, f in zip(x_train, f_train):
    if f == 'Ses02F_impro08_F012.wav':
        print(f, x.shape)
        print(x.shape[0])
#     if x.shape[0] == 0:
#         print(f, x.shape)

# ======================================================================================================================
# ======================================================================================================================
print('Extra labels:   ', extra_labels)
print('Labels example: ', y_train[1])
print('Feature shape:  ', x_train[1].shape)

# To balance classes ===================================================================================================
# ======================================================================================================================
def cut_x(x, mask):
    res = []
    for r, m in zip(x, mask):
        if m:
            res.append(r)
    return res


def expand_x_y(x, y, max_balance=True):
    event_nums, event_nums_count = np.unique(y, return_counts=True)

    max_n_labels = np.amax(event_nums_count)
    min_n_labels = np.amin(event_nums_count)
    if max_balance:
        # сколько раз нужно будет продублировать каждый класс, чтобы выровнять классы
        n_repeats = [int(max_n_labels / dr) for dr in event_nums_count]

        y_exp = np.copy(y)
        x_exp = x

        for i in range(len(n_repeats)):
            rep = n_repeats[i]
            if rep != 1:
                name = event_nums[i]
                mask = y == name

                # здесь берем все время из изначальных
                temp_labels = y[mask]
                temp_features = cut_x(x, mask)

                mixup_size = len(temp_labels)
                for j in range(rep):
                    y_exp = np.hstack([y_exp, temp_labels])
                    x_exp = x_exp + temp_features
    else:
        ind = np.arange(0, len(y))
        np.random.shuffle(ind)
        y_exp = []
        x_exp = []
        for i in range(len(event_nums)):
            name = event_nums[i]
            mask = y == name
            mask = mask[ind]

            # здесь берем все время из изначальных
            temp_labels = y[mask]
            temp_features = cut_x(x, mask)
            for j in range(min_n_labels):
                y_exp.append(temp_labels[j])
                x_exp.append(temp_features[j])
        y_exp = np.array(y_exp)
    return x_exp, y_exp

# Здесь надо фильтрануть фичи, оставить только нужные метки ============================================================
# ======================================================================================================================
desired_emos = ['neu', 'hap', 'ang', 'sad']

# ======================================================================================================================
# ======================================================================================================================
y_train_lab = np.array([y[0] for y in y_train])
sess_train = np.array([y[-1] for y in y_train])

y_test_lab = np.array([y[0] for y in y_test])
sess_test = np.array([y[-1] for y in y_test])

# ======================================================================================================================
# ======================================================================================================================
event_names, event_counts = np.unique(np.hstack((y_train_lab, y_test_lab)), return_counts=True)

for n, c in zip(event_names, event_counts):
    if c < 100 or n == 'xxx' or n == 'fru' or n == 'exc':
        x_train = cut_x(x_train, y_train_lab != n)
        f_train = cut_x(f_train, y_train_lab != n)
        sess_train = sess_train[y_train_lab != n]
        y_train_lab = y_train_lab[y_train_lab != n]

        x_test = cut_x(x_test, y_test_lab != n)
        f_test = cut_x(f_test, y_test_lab != n)
        sess_test = sess_test[y_test_lab != n]
        y_test_lab = y_test_lab[y_test_lab != n]

x_test = cut_x(x_test, y_test_lab != 'oth')
y_test_lab = y_test_lab[y_test_lab != 'oth']

event_names, event_counts = np.unique(np.hstack((y_train_lab, y_test_lab)), return_counts=True)
event_to_id = {v: i for i, v in enumerate(event_names)}

for n, c in zip(event_names, event_counts):
    print(n, c)

# ======================================================================================================================
# ======================================================================================================================
y_train_num = np.array([event_to_id[event] for event in y_train_lab])
y_test_num = np.array([event_to_id[event] for event in y_test_lab])



# директория для сохранения обученных моделей ==========================================================================
# ======================================================================================================================
models_dir = r'C:\Projects\EventDetectionSDK\python\experiments\emo\raw_wav_IEMOCAP_cpuclstm'

if not os.path.exists(models_dir):
    os.mkdir(models_dir)

# ======================================================================================================================
# ======================================================================================================================
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_num, test_size=0.33, random_state=42)

# ======================================================================================================================
# ======================================================================================================================
# обучение на IEMOCAP + MOTYA
batch_size = 30
cur_feature_dimension = 1
feat_t = False
show_train_info = 0

max_cut = False


def gen_giper_params():
    shapes = [(16000, 1)]
    expand = [True, False]
    for s in shapes:
        for e in expand:
            yield (s, e)


final_probas = []

for feature_shape, to_expand in gen_giper_params():
    config = Config(shape=feature_shape, lr=0.001, n_class=len(event_names), max_epochs=50)

    x_train_s, y_train_s = x_train, y_train
    if to_expand:
        x_train_s, y_train_s = expand_x_y(x_train, y_train, max_cut)

    batcher_train = DummyBatcher(config, batch_size, x_train_s, y_train_s,
                                 feature_dim=cur_feature_dimension, offset=True, transpose_feature=feat_t)
    batcher_val = DummyBatcher(config, batch_size, x_val, y_val,
                               offset=True, feature_dim=cur_feature_dimension, transpose_feature=feat_t)

    # to encode i-n look into get_train_params
    #         model_file = os.path.join(models_dir, 'sff_full10_ulm1d_{}_{}_{}.h5'.format(feature_shape[0], to_expand, sess_id))
    model_file = os.path.join(models_dir, 'rawwav1d_{}_{}_{}.h5'.format(feature_shape[0], to_expand, 1))
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=show_train_info, save_best_only=True)
    reduce_lr = LearningRateScheduler(linear_decay_lr(5, config.max_epochs))
    stopper = EarlyStopping(patience=20)
    callbacks_list = [checkpoint, reduce_lr, stopper]

    p_size = [3, 3, 3, 3]
    k_size = [64, 32, 16, 8]

    model = get_oleg_model(config, p_size, k_size)
    #     print(model.summary())
    history = model.fit_generator(batcher_train, callbacks=callbacks_list,
                                  validation_data=batcher_val,
                                  epochs=config.max_epochs, use_multiprocessing=False, verbose=show_train_info)

    # start prediction
    batcher_test = DummyBatcher(config, batch_size, x_test, offset=True,
                                feature_dim=cur_feature_dimension, repeat_test=10, transpose_feature=feat_t)
    n_preds_on_sample = batcher_test.repeat_test

    model.load_weights(model_file)

    preds = model.predict_generator(batcher_test)
    tmp = []
    pred_num = []
    for i in range(0, len(preds), n_preds_on_sample):
        tmp.append(preds[i: i + n_preds_on_sample, :].mean(axis=0))
        pred_num.append(np.argmax(tmp[-1], axis=0))
    final_probas.append(np.copy(np.asarray(tmp)))
    pred_label = [event_names[i] for i in pred_num]

    print('***************')
    print(model_file)
    print('Unweighted accuracy:', accuracy_score(y_test_lab, pred_label))
    print('Weighted accuracy  :', recall_score(y_test_lab, pred_label, average='macro'))
    print('f1-score           :', f1_score(y_test_lab, pred_label, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test_lab, pred_label))

# ======================================================================================================================
# ======================================================================================================================
os.chdir(r'C:\Projects\EventDetectionSDK\python\experiments\fsin')
from trainutils import plot_confusion_matrix, f1, plot_and_calc_roc_auc

y_test_one_hot = to_categorical(y_test_num)

for y_pred in final_probas:
    fpr, tpr, thrs, best_thrs = plot_and_calc_roc_auc(y_test_one_hot, y_pred, event_names)
    for event, thr in zip(event_names, best_thrs):
        print('Best threshold for event {} is {}'.format(event, thr))
    print('**')

# ======================================================================================================================
# ======================================================================================================================
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_num, test_size=0.33, random_state=42)

def gen_giper_params():
    shapes = [(16000, 1)]
    expand = [True, False]
    for s in shapes:
        for e in expand:
            yield (s, e)


cur_feature_dimension = 1
show_folds_metric = True
batch_size = 30
show_train_info = 1
feat_t = False
gpu_lstm = False

for feature_shape, to_expand in gen_giper_params():
    final_preds = []
    for sess_id in np.unique(sess_train):
        # train start

        config = Config(shape=feature_shape, lr=0.001, n_class=len(event_names), max_epochs=200)

        val_mask = sess_train == sess_id
        tr_mask = sess_train != sess_id

        x_train_s = cut_x(x_train, tr_mask)
        y_train_s = y_train_num[tr_mask]

        if to_expand:
            x_train_s, y_train_s = expand_x_y(x_train_s, y_train_s)

        x_val_s = cut_x(x_train, val_mask)
        y_val_s = y_train_num[val_mask]

        batcher_train = DummyBatcher(config, batch_size, x_train_s, y_train_s,
                                     feature_dim=cur_feature_dimension, offset=True, transpose_feature=feat_t)
        batcher_val = DummyBatcher(config, batch_size, x_val_s, y_val_s,
                                   offset=True, feature_dim=cur_feature_dimension, transpose_feature=feat_t)

        # to encode i-n look into get_train_params
        #         model_file = os.path.join(models_dir, 'sff_full10_ulm1d_{}_{}_{}.h5'.format(feature_shape[0], to_expand, sess_id))
        model_file = os.path.join(models_dir, 'rawwav1d_{}_{}_{}.h5'.format(feature_shape[0], to_expand, sess_id))
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=show_train_info, save_best_only=True)
        reduce_lr = LearningRateScheduler(linear_decay_lr(5, config.max_epochs))
        stopper = EarlyStopping(patience=10)
        callbacks_list = [checkpoint, reduce_lr, stopper]

        #         p_size = [2, 2, 2, 2]
        #         k_size = [16, 16, 8, 4]

        model = get_oleg_model(config, gpu_lstm=gpu_lstm)
        #         model = get_oleg_model_2d(config)
        print(model.summary())
        history = model.fit_generator(batcher_train, callbacks=callbacks_list,
                                      validation_data=batcher_val,
                                      epochs=config.max_epochs, use_multiprocessing=False, verbose=show_train_info)
        # end train

        # start prediction
        batcher_test = DummyBatcher(config, batch_size, x_test, offset=True,
                                    feature_dim=cur_feature_dimension, repeat_test=10, transpose_feature=feat_t)
        n_preds_on_sample = batcher_test.repeat_test

        model.load_weights(model_file)

        preds = model.predict_generator(batcher_test)
        tmp = []
        pred_num = []
        for i in range(0, len(preds), n_preds_on_sample):
            tmp.append(preds[i: i + n_preds_on_sample, :].mean(axis=0))
            pred_num.append(np.argmax(tmp[-1], axis=0))
        final_preds.append(np.asarray(tmp))
        pred_label = [event_names[i] for i in pred_num]

        if show_folds_metric:
            print('***************')
            print('Fold #{}'.format(sess_id))
            print(model_file)
            print('Unweighted accuracy:', accuracy_score(y_test_lab, pred_label))
            print('Weighted accuracy  :', recall_score(y_test_lab, pred_label, average='macro'))
            print('f1-score           :', f1_score(y_test_lab, pred_label, average='macro'))
            print('Confusion matrix:\n', confusion_matrix(y_test_lab, pred_label))

    preds = np.zeros_like(final_preds[0])
    for pred in final_preds:
        preds += pred
    pred_num = []
    for pred in preds:
        pred_num.append(np.argmax(pred, axis=0))
    pred_label = [event_names[i] for i in pred_num]

    print('***************')
    print('feature shape: ', feature_shape)
    print('To expand', to_expand)
    print('Unweighted accuracy:', accuracy_score(y_test_lab, pred_label))
    print('Weighted accuracy  :', recall_score(y_test_lab, pred_label, average='macro'))
    print('f1-score           :', f1_score(y_test_lab, pred_label, average='macro'))
    print('Confusion matrix:\n', confusion_matrix(y_test_lab, pred_label))

# ======================================================================================================================
# ======================================================================================================================
p_size = [2, 2, 2, 2]
k_size = [16, 16, 8, 4]
config = Config(shape=feature_shape, lr=0.001, n_class=len(event_names), max_epochs=200)

# model = get_oleg_model(config, p_size, k_size)
model = get_oleg_model_2d(config)
print(model.metrics)

# Only eval ============================================================================================================
# ======================================================================================================================
from keras.models import load_model

models_dir = r'C:\Projects\EventDetectionSDK\python\experiments\emo\cnn1d_lstm_ulm'

final_preds = []

for sess_id in [1, 2, 3, 5]:
    config = Config(shape=(16000, 1), lr=0.001, n_class=len(event_names), max_epochs=200)

    model_file = os.path.join(models_dir, 'rawwav1d_16000_True_{}.h5'.format(sess_id))
    # start prediction
    batcher_test = DummyBatcher(config, 30, x_test, offset=True,
                                feature_dim=1, repeat_test=10, transpose_feature=False)
    n_preds_on_sample = batcher_test.repeat_test

    model = load_model(model_file)

    preds = model.predict_generator(batcher_test)
    tmp = []
    for i in range(0, len(preds), n_preds_on_sample):
        tmp.append(preds[i: i + n_preds_on_sample, :].mean(axis=0))
    final_preds.append(np.asarray(tmp))

    print('***************')
    print('Fold #{}'.format(sess_id))

preds = np.zeros_like(final_preds[0])
for pred in final_preds:
    preds += pred
preds = preds / len(final_preds)

# ======================================================================================================================
# ======================================================================================================================
ang_proba = preds[:, 0]
hap_proba = preds[:, 1]
neu_proba = preds[:, 2]
sad_proba = preds[:, 3]

# ======================================================================================================================
# ======================================================================================================================
top10_ang = np.argsort(ang_proba)[::-1][:10]
top10_hap = np.argsort(hap_proba)[::-1][:10]
top10_neu = np.argsort(neu_proba)[::-1][:10]
top10_sad = np.argsort(sad_proba)[::-1][:10]

# ======================================================================================================================
# ======================================================================================================================
print(np.array(f_test)[top10_neu])