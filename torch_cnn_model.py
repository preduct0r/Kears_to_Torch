import torch
from torch.nn import Sequential
from torch.nn import functional as F
from torch import nn



def get_oleg_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8), gpu_lstm=True):
    dr = 0.1

    # model
    acoustic_model = Sequential()

    # conv
    # p_size = [3, 3, 3, 3]
    # p_size = [2, 2, 2, 2]

    # k_size = [64, 32, 16, 8]
    # k_size = [16, 16, 8, 4]
    model = nn.Sequential(
        nn.Conv1D(config.shape, (config.shape[0], 8, kernel_size), stride=1, padding=0, bias=True, padding_mode='zeros'),
        nn.BatchNorm1d(),
        nn.ReLU(),
        nn.nn.MaxPool1d(kernel_size = p_size[0])
    )

    # acoustic_model.add(Conv1D(input_shape=config.shape, filters=8, kernel_size=k_size[0], activation=None, strides=1))
    # acoustic_model.add(BatchNormalization())
    # acoustic_model.add(Activation('relu', name='act1'))
    # acoustic_model.add(MaxPooling1D(pool_size=p_size[0]))
    #
    # acoustic_model.add(Conv1D(filters=16, kernel_size=k_size[1], activation=None, strides=1))
    # acoustic_model.add(BatchNormalization())
    # acoustic_model.add(Activation('relu'))
    # acoustic_model.add(MaxPooling1D(pool_size=p_size[1]))
    #
    # acoustic_model.add(Conv1D(filters=32, kernel_size=k_size[2], activation=None, strides=1))
    # acoustic_model.add(BatchNormalization())
    # acoustic_model.add(Activation('relu'))
    # acoustic_model.add(MaxPooling1D(pool_size=p_size[2]))
    #
    # acoustic_model.add(Conv1D(filters=64, kernel_size=k_size[3], activation=None, strides=1))
    # acoustic_model.add(BatchNormalization())
    # acoustic_model.add(Activation('relu'))
    # acoustic_model.add(MaxPooling1D(pool_size=p_size[3]))


    # if gpu_lstm:
    #     acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    #     # acoustic_model.add(Dropout(dr))
    #     acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    # else:
    #     acoustic_model.add(LSTM(units=128, return_sequences=True))
    #     # acoustic_model.add(Dropout(dr))
    #     acoustic_model.add(LSTM(units=128, return_sequences=True))

    # acoustic_model.add(GlobalMaxPooling1D())
    # acoustic_model.add(Dropout(dr))

    # fc
    acoustic_model.add(Dense(units=config.n_classes, activation=None))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('softmax', name='last_activation'))

    # launch model
    acoustic_model.compile(optimizer=optimizers.Adam(lr=config.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return acoustic_model
