import os, pickle
import pandas as pd
import numpy as np
import lightgbm
import librosa
import ntpath
import h5py



def get_data():
    base_dir = r'C:\Users\kotov-d\Documents\bases'

    omg_features_path = os.path.join(base_dir, 'omg', 'feature', 'opensmile')
    iemocap_features_path = os.path.join(base_dir, 'iemocap', 'feature', 'opensmile')
    mosei_features_path = os.path.join(base_dir, 'cmu_mosei', 'feature', 'opensmile')

    def make_x_and_y_iemocap(x, y):
        temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
        temp = temp[temp['cur_label'] != 'xxx'][temp['cur_label'] != 'oth'][temp['cur_label'] != 'dis'][
            temp['cur_label'] != 'fru'][temp['cur_label'] != 'exc'] \
            [temp['cur_label'] != 'sur'][temp['cur_label'] != 'fea'][temp['cur_label'] != 'neu']
        new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
        return [new_x, new_y]

    with open(os.path.join(iemocap_features_path, 'x_train.pkl'), 'rb') as f:
        iemocap_x_train = pickle.load(f)
    with open(os.path.join(iemocap_features_path, 'x_test.pkl'), 'rb') as f:
        iemocap_x_test = pickle.load(f)
    with open(os.path.join(iemocap_features_path, 'y_train.pkl'), 'rb') as f:
        iemocap_y_train = pickle.load(f).loc[:, 'cur_label']
    with open(os.path.join(iemocap_features_path, 'y_test.pkl'), 'rb') as f:
        iemocap_y_test = pickle.load(f).loc[:, 'cur_label']

    [iemocap_x_train, iemocap_y_train] = make_x_and_y_iemocap(iemocap_x_train, iemocap_y_train)
    [iemocap_x_test, iemocap_y_test] = make_x_and_y_iemocap(iemocap_x_test, iemocap_y_test)

    def make_x_and_y_omg(x, y):
        dict_emo = {'anger': 'ang', 'happy': 'hap', 'neutral': 'neu', 'surprise': 'sur', 'disgust': 'dis', 'sad': 'sad',
                    'fear': 'fea'}
        y = y.map(lambda x: dict_emo[x])
        temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
        temp = temp[temp['cur_label'] != 'dis'][temp['cur_label'] != 'sur'] \
            [temp['cur_label'] != 'fea'][temp['cur_label'] != 'neu'].reset_index(drop=True)
        new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
        return [new_x, new_y]

    with open(os.path.join(omg_features_path, 'x_train.pkl'), 'rb') as f:
        omg_x_train = pickle.load(f)
    with open(os.path.join(omg_features_path, 'x_test.pkl'), 'rb') as f:
        omg_x_test = pickle.load(f)
    with open(os.path.join(omg_features_path, 'y_train.pkl'), 'rb') as f:
        omg_y_train = pickle.load(f).loc[:, 'cur_label']
    with open(os.path.join(omg_features_path, 'y_test.pkl'), 'rb') as f:
        omg_y_test = pickle.load(f).loc[:, 'cur_label']

    [omg_x_train, omg_y_train] = make_x_and_y_omg(omg_x_train, omg_y_train)
    [omg_x_test, omg_y_test] = make_x_and_y_omg(omg_x_test, omg_y_test)

    def make_x_and_y_mosei(x, y):
        dict_emo = {'anger': 'ang', 'happiness': 'hap', 'surprise': 'sur', 'disgust': 'dis', 'sadness': 'sad',
                    'fear': 'fea'}
        y = y.map(lambda x: dict_emo[x])
        temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
        temp = temp[temp['cur_label'] != 'dis'][temp['cur_label'] != 'sur'] \
            [temp['cur_label'] != 'fea'].reset_index(drop=True)
        new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
        return [new_x, new_y]

    with open(os.path.join(mosei_features_path, 'x_train.pkl'), 'rb') as f:
        mosei_x_train = pickle.load(f)
    with open(os.path.join(mosei_features_path, 'x_test.pkl'), 'rb') as f:
        mosei_x_test = pickle.load(f)
    with open(os.path.join(mosei_features_path, 'y_train.pkl'), 'rb') as f:
        mosei_y_train = pickle.load(f).loc[:, 'cur_label']
    with open(os.path.join(mosei_features_path, 'y_test.pkl'), 'rb') as f:
        mosei_y_test = pickle.load(f).loc[:, 'cur_label']

    [mosei_x_train, mosei_y_train] = make_x_and_y_mosei(mosei_x_train, mosei_y_train)
    [mosei_x_test, mosei_y_test] = make_x_and_y_mosei(mosei_x_test, mosei_y_test)

    # ==========================================================================
    # take only top 100 features
    # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
    #                      objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
    #                      subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
    #
    # clf.fit(iemocap_x_train, iemocap_y_train)
    #
    # with open(os.path.join(r'C:\Users\preductor\PycharmProjects\STC', 'clf' + '.pkl'), 'wb') as f:
    #     clf = pickle.dump(clf, f, protocol=3)

    with open(os.path.join(r'C:\Users\kotov-d\Documents\task#5', 'clf' + '.pickle'), 'rb') as f:
        clf = pickle.load(f)

    dict_importance = {}
    for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
        dict_importance[feature] = importance

    best_features = []

    for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
        if idx == 100:
            break
        best_features.append(w)

    iemocap_x_train = iemocap_x_train.loc[:, best_features]
    iemocap_x_test = iemocap_x_test.loc[:, best_features]
    omg_x_train = omg_x_train.loc[:, best_features]
    omg_x_test = omg_x_test.loc[:, best_features]
    mosei_x_train = mosei_x_train.loc[:, best_features]
    mosei_x_test = mosei_x_test.loc[:, best_features]

    return [[iemocap_x_train, iemocap_x_test, omg_x_train, omg_x_test, mosei_x_train, mosei_x_test],
        [iemocap_y_train, iemocap_y_test, omg_y_train, omg_y_test, mosei_y_train, mosei_y_test]]

def rectify_data(base_dir, meta_path):
    meta_data = pd.read_csv(meta_path, delimiter=';')
    # meta_data = meta_data.iloc[:20]
    # print(meta_data.shape[0])

    rect_x, rect_y, ground_truth_sr = [], [], 8000
    for idx ,row in meta_data.loc[:,['cur_name','cur_label']].iterrows():
        x, sr = librosa.core.load(os.path.join(base_dir, 'data', row[0]), sr=None, mono=False, res_type='kaiser_best', dtype=np.float32)
        # if sr != ground_truth_sr:
        #     print('What the hell is going on with sample rate of your files?')
        #     raise Exception()
        x.astype(np.float16)
        x = [k for k in x]
        y = row[1]

        for k in range(0, len(x)-16000, 8000):
            rect_x.append(tuple(x[k:k+16000]))
            rect_y.append(y)
        for element in rect_x[::-1]:
            if len(element) < 16000:
                element = np.pad(element, (0, 16000 - len(element)), mode='constant')
                rect_x.append(tuple(element))
                rect_y.append(y)
            else:
                break

        if idx%50==0 and idx!=0:
            print(idx)

    def make_array(x):
        new_x = [np.array(c) for c in x]

        array_x = np.vstack(new_x)
        return array_x

    # final_x = np.empty(shape=(len(rect_x), len(rect_x[0])))
    # for idx, i in enumerate(rect_x):
    #     final_x[idx] = i

    final_x = make_array(rect_x)

    dict_emo, reverse_dict_emo, mark = {}, {}, True
    if mark==True:
        for idx,i in enumerate(np.unique(rect_y)):
            dict_emo[i] = idx
            reverse_dict_emo[idx] = i
            mark=False
        with open(r'C:\Users\kotov-d\Documents\ulma\dictionaries.pkl', 'wb') as f:
            pickle.dump([dict_emo, reverse_dict_emo], f, protocol=2)
    else:
        with open(r'C:\Users\kotov-d\Documents\ulma\dictionaries.pkl', 'rb') as f:
            [dict_emo, reverse_dict_emo] = pickle.load(f)

    rect_y = [dict_emo[i] for i in rect_y]
    rect_y = np.array(rect_y).reshape(-1,1)

    rect_data = np.hstack((final_x, rect_y))
    print(rect_data.shape, np.unique(rect_data[:,-1]))




    if ntpath.basename(str(meta_path))=='meta_train.csv':
        h5f = h5py.File(r'C:\Users\kotov-d\Documents\ulma\x_train.h5', 'w')
        h5f.create_dataset('x_train', data=rect_data)
        h5f.close()
    else:
        h5f = h5py.File(r'C:\Users\kotov-d\Documents\ulma\x_test.h5', 'w')
        h5f.create_dataset('x_test', data=rect_data, dtype=np.float16)
        h5f.close()



