import os
import scipy.io
import numpy as np
import keras
from keras.utils import np_utils


def combine_file(num):
    if num < 10:
        path = 'New_Train/' + 'C' + str(num)
    else:
        path = 'New_Train/' + 'P' + str(num)

    if os.path.exists(path + 'D.mat'):
        data = scipy.io.loadmat(path + 'D.mat')
        data = data['BD']
        if os.path.exists(path + 'N.mat'):
            data_n = scipy.io.loadmat(path + 'N.mat')
            data_n = data_n['BD']
            data = np.vstack((data, data_n))
    else:
        data = scipy.io.loadmat(path + 'N.mat')
        data = data['BD']

    return data


def slide_window(data):
    t = data.shape[0]
    n = t // 45 - 3
    data_overlap = np.zeros((n, 180, 33))

    for i in range(n):
        data_overlap[i] = data[(i * 45): (i * 45 + 180)]

    return data_overlap


def compute_label(data):
    n = data.shape[0]
    label = np.zeros(n)
    for i in range(n):
        label[i] = 1 if sum(data[i][:, 32]) > 90 else 0
    return keras.utils.np_utils.to_categorical(label, 2)


def jittering(instance, var):
    for i in range(30):
        instance[:, i] += np.random.normal(0, var, size=(180, ))
    return instance


def data_augmentation(data):
    augmentation = []
    for i in range(len(data)):
        if sum(data[i][:, 32]) > 90:
            augmentation.append(jittering(data[i], 0.05))
            augmentation.append(jittering(data[i], 0.10))
            augmentation.append(jittering(data[i], 0.15))
    return augmentation


def load_data(num):
    data_validation = combine_file(num)
    data_training = []
    for i in range(1, 24):
        if i != num:
            training = combine_file(i)
            data_training.append(training)
    data_training = np.concatenate(data_training, axis=0)

    data_training = slide_window(data_training)
    data_validation = slide_window(data_validation)

    data_after_augmentation = data_augmentation(data_training)
    data_training = np.concatenate((data_training, data_after_augmentation), axis=0)

    np.random.shuffle(data_training)
    np.random.shuffle(data_validation)

    x_train = data_training[:, :, 0: 30]
    x_valid = data_validation[:, :, 0: 30]
    y_train = compute_label(data_training)
    y_valid = compute_label(data_validation)

    return x_train, x_valid, y_train, y_valid
