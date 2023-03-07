from utility_db_outliers import load_dataset
from models_lib import load_custom_model_for_ds
import h5py
from metrics2 import *
from metrics import *
from general_setting import *
import time
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
# ------------
from utility_methods2 import *
from sklearn.metrics import roc_auc_score

from tensorflow.keras.utils import to_categorical
import gzip
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from heapq import nsmallest
from collections import Counter
from scipy import stats
from sklearn import svm
import random
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# from setup_paths import *
from sklearn.decomposition import FastICA, PCA, KernelPCA

BATCH_SIZE = 100
SAVE_RESULTS = True
id_name = ID_DS_LIST[0]  # selects the ID dataset.
id_model = ID_MODEL_LIST[0]  # select the deep model used for training ID dataset.
print(id_name, id_model)


if id_name == "MNIST" and id_model == "LeNet":
    ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-l0', 'cw-li', \
               'cw-l2_target', 'cw-l0_target', 'cw-li_target', 'deepfool', \
               'fgsm015', 'fgsm020', 'fgsm030', 'fgsm035', 'fgsm040', 'fgsm045']
elif id_name == "CIFAR10":
    if id_model == "VGG":
        ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-l0', 'cw-li', \
                   'cw-l2_target', 'cw-l0_target', 'deepfool', \
                   'fgsm01', 'fgsm005', 'fgsm015', 'fgsm020', 'fgsm025', 'fgsm030']
    elif id_model == "ResNet":
        ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-l0', \
                   'cw-li', 'cw-l2_target', 'cw-l0_target', 'deepfool', \
                   'fgsm001', 'fgsm003', 'fgsm005', 'fgsm008', 'fgsm010']
elif id_name == "SVHN":
    ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-l0', \
               'cw-li', 'cw-l2_target', 'cw-l0_target', 'deepfool', \
               'fgsm001', 'fgsm003', 'fgsm005', 'fgsm008', 'fgsm010', 'fgsm013']

if id_name == "MNIST" and id_model == "LeNet":
    layers = (0, 1, 2, 3, 6, 7)
elif id_name == "CIFAR10":
    if id_model == "VGG":
        layers = (0, 3, 5, 6, 9, 11, 12, 15, 18, 20, 21, 24, 27, 30, 33, 36)
    elif id_model == "ResNet":
        layers = (
        3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48, 52, 55, 60, 63, 67, 70, 74, 77, 81, 84, 88, 91, 95, 98,
        102, 105, 110, 113, 117, 120, 124, 127, 131, 134, 138, 141, 145, 148)
elif id_name == "SVHN":
    if id_model == "VGG":
        layers = (0, 3, 5, 6, 9, 11, 12, 15, 18, 20, 21, 24, 27, 30, 33, 36)
    elif id_model == "ResNet":
        layers = (3, 6, 10, 13, 17, 20, 24, 27, 32, 35, 39, 42, 46, 49, 54, 57, 61, 64)


(org_traing_data, org_training_labels), (id_eva_data, org_testing_labels) = load_dataset(id_name)
org_traing_data_processed = preprocess_images(id_name, org_traing_data, id_model, verbose=True)
id_eva_data_processed = preprocess_images(id_name, id_eva_data, id_model, verbose=True)
train_db = org_traing_data_processed
test_db = id_eva_data_processed

y_test = to_categorical(org_testing_labels, 10)
y_train = to_categorical(org_training_labels, 10)
x_train = org_traing_data_processed
x_test = test_db = id_eva_data_processed
org_model = load_custom_model_for_ds(id_name, id_model)


for lay_i in range(len(layers)):
    lay_id = layers[lay_i]
    db_name = 'train'
    tmp_l = org_model.layers[lay_id]
    modelx_1 = Model(inputs=org_model.input, outputs=tmp_l.output)
    features_in = modelx_1.predict(x_train)
    features_1 = postprocess_features(features_in)
    tmp_l = org_model.layers[-2]
    modelx_2 = Model(inputs=org_model.input, outputs=tmp_l.output)
    features_2 = modelx_2.predict(x_train)
    x_train_cat = np.hstack((features_1, features_2))
    # np.save('./ADDM/datas_layers/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, db_name, lay_id), x_train_cat)
    num_units = x_train_cat.shape[1]

    # Get derived models for each layer
    model = Sequential()
    model.add(Dense(input_dim=num_units, units=50, activation='relu'))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train_cat, y_train, batch_size=256, epochs=20)
    model.save('./ADDM/Derived_Models/%s/%s_%s_layer_%s.h5' % (id_name, id_name, id_model, lay_id))

    # model = load_model('./ADDM/Derived_Models/%s/%s_%s_layer_%s.h5' % (id_name, id_name, id_model, lay_id))

    tmp_l = model.layers[0]
    feature_model_0 = Model(inputs=model.input, outputs=tmp_l.output)
    tmp_l = model.layers[1]
    feature_model_1 = Model(inputs=model.input, outputs=tmp_l.output)
    tmp_l = model.layers[2]
    feature_model_2 = Model(inputs=model.input, outputs=tmp_l.output)

    # Collect the features of sunclasses for training datasets (搜集ID小类别的16维特征   noise_train)
    for num in range(11):
        train_noise = np.load('./datas_train/%s/%s_%s_train_noise_%s.npy' % (id_name, id_name, id_model, num),
                              allow_pickle=True)
        noise_features = [[] for i in range(10)]
        for class_k in range(10):
            a = np.asarray(train_noise[class_k])
            features_in = modelx_1.predict(a)
            features_1 = postprocess_features(features_in)
            features_2 = modelx_2.predict(a)
            noise_features[class_k] = np.hstack((features_1, features_2))

        neus_cat = [[] for i in range(10)]
        for class_k in range(10):
            x0 = feature_model_0.predict(noise_features[class_k].reshape(-1, num_units))
            x1 = feature_model_1.predict(noise_features[class_k].reshape(-1, num_units))
            x2 = feature_model_2.predict(noise_features[class_k].reshape(-1, num_units))
            neus_cat[class_k] = np.hstack((x0, x1, x2))
        np.save('./ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_%s.npy' % (id_name, id_name, id_model, lay_id, num),
                neus_cat)

    ## the first variance (分析std  noise)
    std_noise = [[] for i in range(11)]
    for num in range(11):
        neu_train_noise = np.load(
            './ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_%s.npy' % (id_name, id_name, id_model, lay_id, num),
            allow_pickle=True)
        std = [[] for i in range(10)]
        for class_k in range(10):
            ne_ = neu_train_noise[class_k]
            sd = []
            for j in range(ne_.shape[1]):
                datas = ne_[:, j]
                data_std = np.std(datas)
                sd.append(data_std)
            std[class_k] = sd
        std_noise[num] = std

    ## the second variance (在std_noise基础上分析std)
    std_std = [[] for i in range(10)]
    for class_k in range(10):
        ne_ = np.asarray(std_noise)[:, class_k]
        sd = []
        for j in range(ne_.shape[1]):
            datas = ne_[:, j]
            data_std = np.std(datas)
            sd.append(data_std)
        std_std[class_k] = sd

    ## the order of neurons for above two variance analysis (在分析std后得到的排序)
    index_order = [[] for i in range(10)]
    for class_k in range(10):
        index_order[class_k] = np.argsort(std_std[class_k])  # std降序排列，最前面的std最小
    np.save('./ADDM/Index_Order/%s/%s_%s_index_order_layer_%s.npy' % (id_name, id_name, id_model, lay_id), index_order)

#  Collect the neuron features from derived model for each layer of target model (搜集Anomaly的每一层neurons的特征)
for attack in ATTACKS:
    ds_name = attack
    print(ds_name)

    for lay_i in range(len(layers)):
        lay_id = layers[lay_i]
        db_name = 'test'
        x_test_cat = np.load(
            './ADDM/datas_layers/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, db_name, lay_id),
            allow_pickle=True)
        num_units = x_test_cat.shape[1]

        model = load_model('./ADDM/Derived_Models/%s/%s_%s_layer_%s.h5' % (id_name, id_name, id_model, lay_id))
        tmp_l = model.layers[0]
        feature_model_0 = Model(inputs=model.input, outputs=tmp_l.output)
        tmp_l = model.layers[1]
        feature_model_1 = Model(inputs=model.input, outputs=tmp_l.output)
        tmp_l = model.layers[2]
        feature_model_2 = Model(inputs=model.input, outputs=tmp_l.output)
        test_features = np.load(
            './ADDM/datas_layers/%s/%s_%s_%s_layer_%s_ID.npy' % (id_name, id_name, id_model, ds_name, lay_id),
            allow_pickle=True)
        adv_features = np.load(
            './ADDM/datas_layers/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id),
            allow_pickle=True)

        test_neus_cat = [[] for i in range(10)]
        adv_neus_cat = [[] for i in range(10)]
        for class_i in range(10):
            x0 = feature_model_0.predict(test_features[class_i].reshape(-1, num_units))
            x1 = feature_model_1.predict(test_features[class_i].reshape(-1, num_units))
            x2 = feature_model_2.predict(test_features[class_i].reshape(-1, num_units))
            test_neus_cat[class_i] = np.hstack((x0, x1, x2))

            x0 = feature_model_0.predict(adv_features[class_i].reshape(-1, num_units))
            x1 = feature_model_1.predict(adv_features[class_i].reshape(-1, num_units))
            x2 = feature_model_2.predict(adv_features[class_i].reshape(-1, num_units))
            adv_neus_cat[class_i] = np.hstack((x0, x1, x2))

        np.save('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_ID.npy' % (id_name, id_name, id_model, ds_name, lay_id),
                test_neus_cat)
        np.save('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id),
                adv_neus_cat)


