import tensorflow
from keras.layers import (Convolution2D, Input, BatchNormalization,
                          MaxPool2D, Dropout, GlobalAveragePooling2D, GlobalMaxPooling1D,
                          ZeroPadding2D, Activation, Dense, concatenate, CuDNNLSTM,
                          Reshape, Permute, multiply, Average, GlobalMaxPooling2D, GlobalAveragePooling2D,
                          Conv2D, add, MaxPooling2D, Flatten, LSTM, Conv1D, MaxPooling1D)

from keras import models
from keras.models import Model, Sequential
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs


# from experiments.fsin.trainutils import f1

import tensorflow.keras.backend as K


class Config:
    def __init__(self, shape, lr, n_classes, num_epochs=500):
        self.shape = shape
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_classes = n_classes


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    res = true_positives / (possible_positives + K.epsilon())
    return res


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    res = true_positives / (predicted_positives + K.epsilon())
    return res


def f1(y_true, y_pred):
    def _recall(_y_true, _y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(_y_true * _y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(_y_true, 0, 1)))
        res = true_positives / (possible_positives + K.epsilon())
        return res

    def _precision(_y_true, _y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(_y_true * _y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(_y_pred, 0, 1)))
        res = true_positives / (predicted_positives + K.epsilon())
        return res

    pr = _precision(y_true, y_pred)
    rec = _recall(y_true, y_pred)
    return 2*((pr*rec)/(pr+rec+K.epsilon()))


def get_oleg_model(config, p_size=(3, 3, 3, 3), k_size=(64, 32, 16, 8), gpu_lstm=True):
    dr = 0.1

    # model
    acoustic_model = Sequential()

    # conv
    # p_size = [3, 3, 3, 3]
    # p_size = [2, 2, 2, 2]

    # k_size = [64, 32, 16, 8]
    # k_size = [16, 16, 8, 4]

    acoustic_model.add(Conv1D(input_shape=config.shape, filters=8, kernel_size=k_size[0], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu', name='act1'))
    acoustic_model.add(MaxPooling1D(pool_size=p_size[0]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv1D(filters=16, kernel_size=k_size[1], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling1D(pool_size=p_size[1]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv1D(filters=32, kernel_size=k_size[2], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling1D(pool_size=p_size[2]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv1D(filters=64, kernel_size=k_size[3], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling1D(pool_size=p_size[3]))
    # acoustic_model.add(GlobalMaxPooling1D())
    # acoustic_model.add(Dropout(dr))

    if gpu_lstm:
        acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
        # acoustic_model.add(Dropout(dr))
        acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    else:
        acoustic_model.add(LSTM(units=128, return_sequences=True))
        # acoustic_model.add(Dropout(dr))
        acoustic_model.add(LSTM(units=128, return_sequences=True))

    acoustic_model.add(GlobalMaxPooling1D())
    acoustic_model.add(Dropout(dr))

    # fc
    acoustic_model.add(Dense(units=config.n_classes, activation=None))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('softmax', name='last_activation'))

    # launch model
    acoustic_model.compile(optimizer=optimizers.Adam(lr=config.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return acoustic_model


def get_oleg_model_2d(config):
    l_r = 0.0001
    dr = 0.1

    # model
    acoustic_model = Sequential()

    # conv
    p_size = [(2, 2), (2, 2), (2, 2), (2, 2)]
    k_size = [(4, 5), (3, 4), (3, 3), (3, 3)]
    filter_numbers = [8, 16, 16, 24]
    acoustic_model.add(Conv2D(input_shape=config.shape, filters=filter_numbers[0],
                              kernel_size=k_size[0], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling2D(pool_size=p_size[0]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv2D(filters=filter_numbers[1], kernel_size=k_size[1], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling2D(pool_size=p_size[1]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv2D(filters=filter_numbers[2], kernel_size=k_size[2], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling2D(pool_size=p_size[2]))
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Conv2D(filters=filter_numbers[3], kernel_size=k_size[3], activation=None, strides=1))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('relu'))
    acoustic_model.add(MaxPooling2D(pool_size=p_size[3], name='last_pool'))

    layer_shape = acoustic_model.get_layer('last_pool').output_shape
    target_shape = (int(layer_shape[1]), int(layer_shape[2] * layer_shape[3]))
    # acoustic_model.add(GlobalMaxPooling1D())
    # acoustic_model.add(Dropout(dr))

    acoustic_model.add(Reshape(target_shape=target_shape))

    # class my_pool(GlobalAveragePooling2D):
    #     def call(self, inputs):
    #         return K.mean(inputs, axis=[-1])
    #
    # acoustic_model.add(my_pool())


    # rec
    # acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    # acoustic_model.add(Dropout(dr))
    acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    # acoustic_model.add(Dropout(dr))
    acoustic_model.add(CuDNNLSTM(units=128, return_sequences=True))
    acoustic_model.add(GlobalMaxPooling1D())
    acoustic_model.add(Dropout(dr))

    # fc
    acoustic_model.add(Dense(units=config.n_class, activation=None))
    acoustic_model.add(BatchNormalization())
    acoustic_model.add(Activation('softmax'))

    # launch model
    acoustic_model.compile(optimizer=optimizers.Adam(lr=l_r), loss='categorical_crossentropy', metrics=['accuracy', f1])

    return acoustic_model


def get_altered_indian_model(config, metric, sff=True, use_lstm=True):
    inp = Input(shape=config.shape)

    def cnn_block(x, filters, conv_kernel_size, conv_stride, pool_size):
        x = Conv2D(filters=filters, kernel_size=conv_kernel_size, strides=conv_stride)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size, strides=(1, 1))(x)

        return x

    # # FxT
    # x = cnn_block(inp, 16, (12, 16), (1, 4), (100, 150))
    # x = cnn_block(x, 24, (8, 12), (1, 1), (50, 75))
    # x = cnn_block(x, 32, (5, 7), (1, 1), (25, 37))

    # TxF
    if sff:
        x = cnn_block(inp, 16, (16, 12), (4, 1), (150, 100))
    else:
        x = cnn_block(inp, 16, (4, 3), (2, 2), (5, 5))
    x = cnn_block(x, 24, (3, 3), (1, 1), (4, 4))
    x = cnn_block(x, 32, (3, 3), (1, 1), (3, 3))
    x = cnn_block(x, 64, (3, 3), (2, 2), (3, 3))
    if use_lstm:
        x = Reshape(target_shape=(int(x.shape[1]), int(x.shape[2] * x.shape[3])), name='reshape')(x)
        x = LSTM(128, go_backwards=True, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(units=64)(x)
    x = Activation('relu')(x)
    out = Dense(config.n_class, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.lr)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=[metric])
    return model


def get_indian_model(config, sff=False, use_lstm=True):
    inp = Input(shape=config.shape)

    def cnn_block(x, filters, conv_kernel_size, conv_stride, pool_size):
        x = Conv2D(filters=filters, kernel_size=conv_kernel_size, strides=conv_stride)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=pool_size, strides=(1, 1))(x)

        return x

    # # FxT
    # x = cnn_block(inp, 16, (12, 16), (1, 4), (100, 150))
    # x = cnn_block(x, 24, (8, 12), (1, 1), (50, 75))
    # x = cnn_block(x, 32, (5, 7), (1, 1), (25, 37))

    # TxF
    if sff:
        x = cnn_block(inp, 16, (16, 12), (4, 1), (150, 100))
    else:
        x = cnn_block(inp, 16, (16, 12), (1, 1), (150, 100))
    x = cnn_block(x, 24, (12, 8), (1, 1), (75, 50))
    x = cnn_block(x, 32, (7, 5), (1, 1), (37, 25))
    if use_lstm:
        x = Reshape(target_shape=(int(x.shape[1]), int(x.shape[2] * x.shape[3])), name='reshape')(x)
        x = LSTM(128, go_backwards=True, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(units=64)(x)
    x = Activation('relu')(x)
    out = Dense(config.n_class, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.lr)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc', f1])
    return model


def get_2nd_place_model(config, metric, batch_norm=True):

    inp = Input(shape=config.shape)
    x = BatchNormalization()(inp)
    nf = 48
    x = ZeroPadding2D(((2, 2), (2, 2)))(x)
    x = Convolution2D(kernel_size=(5, 5), filters=nf, strides=2, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)

    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=2*nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    # x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    # x = Convolution2D(kernel_size=(3, 3), filters=2*nf, strides=1, activation='relu', data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(0.3)(x)

    # x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    # x = Convolution2D(kernel_size=(3, 3), filters=4*nf, strides=1, activation='relu', data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=4*nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=6*nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    # x = Convolution2D(kernel_size=(3, 3), filters=6*nf, strides=1, activation='relu', data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = MaxPool2D(pool_size=(2, 2))(x)
    # x = Dropout(0.3)(x)

    # x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    # x = Convolution2D(kernel_size=(3, 3), filters=8*nf, strides=1, activation='relu', data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=8*nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)

    # x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    # x = Convolution2D(kernel_size=(3, 3), filters=8*nf, strides=1, activation='relu', data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    x = ZeroPadding2D(((1, 1), (1, 1)))(x)
    x = Convolution2D(kernel_size=(3, 3), filters=8*nf, strides=1, activation='relu', data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.3)(x)

    # x = Convolution2D(kernel_size=(3, 3), filters=8*nf, strides=1, padding='valid', activation='relu',
    #                   data_format='channels_last')(x)
    # if batch_norm:
    #     x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Convolution2D(kernel_size=(1, 1), filters=8*nf, strides=1, padding='valid', activation='relu',
                      data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(kernel_size=(1, 1), filters=config.n_class, strides=1, data_format='channels_last')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    out = Activation(activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.lr)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=[metric])
    return model


def get_dummy_model(config):
    inp = Input(config.shape)

    x = Convolution2D(kernel_size=(5, 5), filters=64, strides=1, activation='relu', data_format='channels_last')(inp)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Convolution2D(kernel_size=(3, 3), filters=96, strides=1, activation='relu', data_format='channels_last')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Convolution2D(kernel_size=(3, 3), filters=96, strides=1, activation='relu', data_format='channels_last')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Convolution2D(kernel_size=(3, 3), filters=96, strides=1, activation='relu', data_format='channels_last')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(config.n_class, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.lr)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model


def get_seresnet18_model(config):
    return SEResNet18(lr=config.lr, input_shape=config.shape, classes=config.n_class)


def linear_decay_lr(start_epoch, end_epoch):
    def result_function(epoch, lr):
        if epoch <= start_epoch:
            return lr
        else:
            return lr*(end_epoch - epoch)/(end_epoch - start_epoch)

    return result_function


def SEResNet(lr, input_shape=None,
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filter: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            width: width multiplier for the network (for Wide ResNets)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                  default_size=224,
    #                                  min_size=32,
    #                                  data_format=K.image_data_format(),
    #                                  require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    x = BatchNormalization()(img_input)
    x = _create_se_resnet(classes, x, include_top, initial_conv_filters,
                          filters, depth, width, bottleneck, weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

    # load weights
    opt = optimizers.Adam(lr)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def SEResNet18(lr=0.001, input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(lr=lr, input_shape=input_shape,
                    depth=[2, 2, 2, 2],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet34(lr=0.001, input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(lr=lr, input_shape=input_shape,
                    depth=[3, 4, 6, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def _resnet_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block without bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = -1

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block with bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, bottleneck, weight_decay, pooling):
    '''Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''
    channel_axis = -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (3, 3), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x


if __name__ == '__main__':

    # 1st place config
    # config = Config((1, 64000), 0.001, 41)
    # mod = get_1st_place_model(config)

    # 2nd place config
    config = Config((128, 384, 1), 0.001, 4)
    mod = get_2nd_place_model(config, f1)
    print(mod.summary())
