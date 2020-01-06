import numpy as np
from keras.utils import Sequence, to_categorical


class DummyBatcher(Sequence):
    def __init__(self, config, batch_size, x, y=None, feature_dim=2):
        self.config = config
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.indexes = np.arange(len(x))
        np.random.shuffle(self.indexes)
        self.feature_dim = feature_dim

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if self.y is not None:
            indexes_tmp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            res_x = []
            for i in indexes_tmp:
                res_x.append(self.x[i])
            res_x = np.asarray(res_x)[:, :, np.newaxis]
            res_y = to_categorical(self.y)[indexes_tmp]
            return res_x, res_y
        else:
            indexes_tmp = np.arange(len(self.x))[index * self.batch_size:(index + 1) * self.batch_size]
            res_x = []
            for i in indexes_tmp:
                for j in range(self.repeat_test):
                    res_x.append(self.prepare_feature(self.x[i]))
            res_x = np.asarray(res_x)
            return res_x