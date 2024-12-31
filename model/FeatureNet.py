from pyexpat import features
import tensorflow as tf
import keras
import numpy as np
from scipy.interpolate import interp1d
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from keras.layers import Flatten, Reshape, TimeDistributed, BatchNormalization, Resizing
from tensorflow.keras.optimizers import Adam

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



'''
A Feature Extractor Network
'''

def build_FeatureNet(opt, channels=10, sequence_length=3000, class_num=2):
    activation = tf.nn.relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(sequence_length, 1), name='input_signal')
    print('input_signal:', input_signal.shape)


    ######### CNNs with small filter size at the first layer #########
    cnn0 = Conv1D(kernel_size=50,
                  filters=32,
                  strides=6,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=16, strides=16)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    cnn3 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn5 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=8, strides=8)
    s = cnn6(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]), ))  # Flatten
    s = cnn7(s)
    print('s:', s.shape)

    ######### CNNs with large filter size at the first layer #########
    cnn8 = Conv1D(kernel_size=400,
                  filters=64,
                  strides=50,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=8, strides=8)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    cnn11 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn13 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=4, strides=4)
    l = cnn14(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]), ))
    l = cnn15(l)
    print('l:', l.shape)

    feature = keras.layers.concatenate([s, l])

    fea_part = Model(input_signal, feature)

    ##################################################

    input = Input(shape=(channels, sequence_length), name='input_signal')
    reshape = Reshape((channels, sequence_length, 1))  # Flatten
    input_re = reshape(input)
    fea_all = TimeDistributed(fea_part)(input_re)

    merged = Flatten()(fea_all)
    merged = Dropout(0.5)(merged)
    merged = Dense(64)(merged)
    merged = Dense(class_num)(merged)

    fea_softmax = Activation(activation='softmax')(merged)

    # FeatureNet with softmax
    fea_model = Model(input, fea_softmax)
    fea_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    # FeatureNet without softmax
    pre_model = Model(input, fea_all)
    pre_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    return fea_model, pre_model


def resize_sequence(X, target_length):
    """
    将输入数据 X 的序列长度调整为 target_length。
    X 的形状为 (N, C, L)，调整后为 (N, C, target_length)。
    """
    N, C, L = X.shape
    resized_X = np.zeros((N, C, target_length), dtype=X.dtype)
    for i in range(N):
        for j in range(C):
            # 使用 scipy.interpolate.interp1d 进行插值
            f = interp1d(np.arange(L), X[i, j], kind='linear', fill_value="extrapolate")
            resized_X[i, j] = f(np.linspace(0, L - 1, target_length))
    return resized_X


if __name__ == '__main__':
    print('start')
    # opt = Adam(learning_rate=0.001)

    # fea_model, pre_model = build_FeatureNet(opt=opt, channels=10, time_second=30, freq=100)

    # # 打印模型摘要
    # print("FeatureNet with softmax:")
    # fea_model.summary()
    # print("\nFeatureNet without softmax:")
    # pre_model.summary()

    # # 假设我们有一些随机生成的数据来模拟输入
    # import numpy as np
    # # 生成模拟数据
    # input_shape = (1, 10, 30 * 100)  # 1个样本，10个通道，每个通道30秒*100Hz
    # dummy_input = np.random.random(input_shape)

    # # 使用生成的数据进行一次前向传播
    # dummy_output_softmax = fea_model.predict(dummy_input)
    # dummy_output = pre_model.predict(dummy_input)

    # # 打印输出的形状
    # print("Input shape:", dummy_input.shape)
    # print("Output shape with softmax:", dummy_output_softmax.shape)
    # print("Output shape without softmax(feature):", dummy_output.shape)

    C = 6
    L = 896
    fix_length = 3000
    batch_size = 8
    epochs = 2
    class_num = 5

    N = 204  # Number of samples
    X = np.random.rand(N, C, L)
    y = np.random.randint(0, class_num, size=(N, class_num))

    print(X.shape, y.shape)

    if L != fix_length:
        X = resize_sequence(X, fix_length)

    # Build the model
    opt = Adam(learning_rate=0.001)
    fea_model, pre_model = build_FeatureNet(opt, channels=C, sequence_length=fix_length, class_num=class_num)

    # Train the model
    fea_model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Evaluate the model
    loss, acc = fea_model.evaluate(X, y)
    print(f"Test loss: {loss}, Test accuracy: {acc}")

    output = fea_model.predict(X, batch_size=batch_size)
    features = pre_model.predict(X, batch_size=batch_size)
    print("Output shape:", output.shape)
    print("Features shape:", features.shape)

'''
Input: (100, 10, 3000)
Output with softmax: (100, 2)
Features: (100, 10, 256)

Input: (204, 61, 405)
Output with softmax: ()
Features: ()

'''