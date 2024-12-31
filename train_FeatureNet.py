from statistics import mean
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from model.FeatureNet import build_FeatureNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

batch_size = 16
epochs = 10
learning_rate = 0.001

fix_length = 3000

def load_data(data_path, file_name):
    # Load data
    data = pickle.load(open(data_path + file_name, 'rb'))
    X = data['x']
    y = data['y']

    return X, y

def resize_sequence(X, target_length):
    N, C, L = X.shape
    resized_X = np.zeros((N, C, target_length), dtype=X.dtype)
    for i in range(N):
        for j in range(C):
            # 使用 scipy.interpolate.interp1d 进行插值
            f = interp1d(np.arange(L), X[i, j], kind='linear', fill_value="extrapolate")
            resized_X[i, j] = f(np.linspace(0, L - 1, target_length))
    return resized_X

def data_generator(X, y, batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            batch_X = resize_sequence(batch_X, fix_length)
            yield batch_X, batch_y


def scale(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std = np.maximum(std, 1.0)
    data = (data - mean) / std
    return data



if __name__ == '__main__':
    print('start')
    data_path = 'UEA_dataset/Heartbeat/'
    train_file = 'Heartbeat_TRAIN.pkl'
    test_file = 'Heartbeat_TEST.pkl'

    class_num=2

    X_train, y_train = load_data(data_path, train_file)
    X_test, y_test = load_data(data_path, test_file)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    # trans y into one-hot
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=class_num)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=class_num)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    C = X_train.shape[1]
    L = X_train.shape[2]

    # shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)

    # Resize sequence
    if L != fix_length:
        X_train = resize_sequence(X_train, fix_length)
        X_test = resize_sequence(X_test, fix_length)
    
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # ------------------- Split data -------------------
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # train_gen = data_generator(X_train, y_train, batch_size)
    # test_gen = data_generator(X_test, y_test, batch_size)
    # val_gen = data_generator(X_val, y_val, batch_size)

    # scale data
    X_train = scale(X_train)
    X_val = scale(X_val)

    # Build model
    opt = Adam(learning_rate=learning_rate)
    fea_model, pre_model = build_FeatureNet(opt, channels=C, sequence_length=fix_length, class_num=class_num)

    # Train model
    fea_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    # fea_model.fit(train_gen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs, validation_data=val_gen, validation_steps=len(X_val) // batch_size)

    # Evaluate model
    train_loss, train_acc = fea_model.evaluate(X_train, y_train, batch_size=batch_size)
    # train_loss, train_acc = fea_model.evaluate(train_gen, steps=len(X_train) // batch_size)
    print(f"Train loss: {train_loss}, Train acc: {train_acc}")
    loss, acc = fea_model.evaluate(X_test, y_test, batch_size=batch_size)
    # loss, acc = fea_model.evaluate(test_gen, steps=len(X_test) // batch_size)
    print(f"Test loss: {loss}, Test acc: {acc}")

    # save the features
    train_features = pre_model.predict(X_train, batch_size=batch_size)
    test_features = pre_model.predict(X_test, batch_size=batch_size)
    # train_features = pre_model.predict(train_gen, steps=len(X_train) // batch_size)
    # test_features = pre_model.predict(test_gen, steps=len(X_test) // batch_size)
    # train_y = y_train[:len(train_features)]
    # test_y = y_test[:len(test_features)]
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)
    pickle.dump({'x': train_features, 'y': y_train}, open(data_path + 'train_features.pkl', 'wb'))
    pickle.dump({'x': test_features, 'y': y_test}, open(data_path + 'test_features.pkl', 'wb'))


