import os
import numpy as np
import keras
import copy
import random
import json
import librosa
from keras.utils import np_utils


def audio_load(num):
    y, sr = librosa.load('G:/AlphaPose/examples/res/audio/' + str(num) + '.wav', sr=25600)
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)
    return mfccs.T


def combine_dance_file(num):
    with open('G:/AlphaPose/examples/res/debug/' + str(num) + '-1.json', 'r') as f1:
        load1 = json.load(f1)
    data1 = load1[0]['keypoints']
    for i in range(1, len(load1) - 2):
        data1 = np.vstack((data1, load1[i]['keypoints']))

    if os.path.exists('G:/AlphaPose/examples/res/debug/' + str(num) + '-2.json'):
        with open('G:/AlphaPose/examples/res/video/' + str(num) + '-2.json', 'r') as f2:
            load2 = json.load(f2)
        data2 = load2[0]['keypoints']
        for i in range(1, len(load2)):
            data2 = np.vstack((data2, load2[i]['keypoints']))
        data1 = np.vstack((data1, data2))

        if os.path.exists('G:/AlphaPose/examples/res/debug/' + str(num) + '-3.json'):
            with open('G:/AlphaPose/examples/res/video/' + str(num) + '-3.json', 'r') as f3:
                load3 = json.load(f3)
            data3 = load3[0]['keypoints']
            for i in range(1, len(load3)):
                data3 = np.vstack((data3, load3[i]['keypoints']))
            data1 = np.vstack((data1, data3))

    return data1


def data_aug(data, var):
    aug = copy.copy(data)
    for i in range(len(data)):
        for j in range(42):
            aug[i][0][:, j] = data[i][0][:, j] + np.random.normal(0, var, size=(150, ))
    return aug


def load_dance_data():
    train = []
    valid = []
    d = [1, 2, 3, 4, 10]

    for i in range(1, 11):
        label = 0 if i in d else 1
        data = combine_dance_file(i)
        data1 = data[:, 0: 3]
        data2 = data[:, 15: 51]
        data3 = (data[:, 15: 18] + data[:, 18: 21]) / 2
        data = np.hstack((data1, data3, data2))
        n = data.shape[0] // 30 - 4
        if i < 9:
            for j in range(n):
                train.append([data[(j * 30): (j * 30 + 150)], label])
        else:
            for j in range(n):
                valid.append([data[(j * 30): (j * 30 + 150)], label])

    augmentation1 = data_aug(train, 0.05)
    augmentation2 = data_aug(train, 0.10)
    augmentation3 = data_aug(train, 0.15)
    train.extend(augmentation1)
    train.extend(augmentation2)
    train.extend(augmentation3)

    augmentation4 = data_aug(valid, 0.05)
    augmentation5 = data_aug(valid, 0.10)
    augmentation6 = data_aug(valid, 0.15)
    valid.extend(augmentation4)
    valid.extend(augmentation5)
    valid.extend(augmentation6)

    random.shuffle(train)
    random.shuffle(valid)

    x_train = np.zeros((len(train), 150, 42))
    y_train = np.zeros(len(train))
    for i in range(len(train)):
        x_train[i] = train[i][0]
        y_train[i] = train[i][1]

    x_valid = np.zeros((len(valid), 150, 42))
    y_valid = np.zeros(len(valid))
    for i in range(len(valid)):
        x_valid[i] = valid[i][0]
        y_valid[i] = valid[i][1]

    y_train = keras.utils.np_utils.to_categorical(y_train, 2)
    y_valid = keras.utils.np_utils.to_categorical(y_valid, 2)

    return x_train, x_valid, y_train, y_valid
