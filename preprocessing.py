from utility_db_outliers import load_dataset
from models_lib import load_custom_model_for_ds
import h5py
from metrics2 import *
from metrics import *
from general_setting import *
import time
import tensorflow as tf
from tensorflow.keras import layers
from utility_methods2 import *
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model, Model

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

means = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# sigmas = [0.05,0.02,0.1,0.08,0.03,0.2,0.04]  #mnist/svhn
sigmas = [0.05, 0.2, 0.15, 0.07, 0.09, 0.1, 0.03]  #cifar10

mean = random.sample(means, 1)
sigma = random.sample(sigmas, 1)
if id_name == 'MNIST' or 'SVHN':
    clip_min = 0.0
    clip_max = 1.0
else:
    clip_min = -2.0
    clip_max = 2.0
def noise_(xx, mean, sigma):
    noise = np.random.normal(mean, sigma, (32,32,3))
    gaussian_out = xx + noise
    gaussian_train = np.clip(gaussian_out, clip_min, clip_max)
    return gaussian_train

# Generate training datasets with  noise
trainimgs = np.load('./datas_new/trainimgs_%s_%s.npy' % (id_model, id_name), allow_pickle=True)
for num in range(1, 11):
    noise_train = [[] for i in range(10)]
    mean = random.sample(means, 1)
    sigma = random.sample(sigmas, 1)
    for i in range(10):
        noise_train[i] = noise_(np.asarray(trainimgs[i]), mean, sigma)
    np.save('./datas_train/%s/%s_%s_train_noise_%s.npy'%(id_name,id_name,id_model,num),noise_train)

def extract_layer_features(in_model, in_data, in_layer, in_batch_size, shorten_model=False):
    for i_d in range(in_data.shape[0] // in_batch_size):
        batch_data = in_data[i_d * in_batch_size:(i_d + 1) * in_batch_size]
        if shorten_model == False:
            batch_features = extract_features(in_model, batch_data, in_layer)
        else:
            batch_features = in_model.predict(batch_data)
        batch_features_processed = postprocess_features(batch_features)
        if i_d == 0:
            features = batch_features_processed
        else:
            features = np.vstack((features, batch_features_processed))

    if (i_d + 1) * in_batch_size < in_data.shape[0]:
        batch_data = in_data[(i_d + 1) * in_batch_size:in_data.shape[0]]

        if shorten_model == False:
            batch_features = extract_features(in_model, batch_data, in_layer)
        else:
            batch_features = in_model.predict(batch_data)

        batch_features_processed = postprocess_features(batch_features)
        if i_d == 0:
            features = batch_features_processed
        else:
            features = np.vstack((features, batch_features_processed))
    return features


def extract_features(in_model, in_img_perturbed, in_layer_inx):
    l = in_model.layers[in_layer_inx]
    aux_model = Model(inputs=in_model.input, outputs=l.output)
    return aux_model.predict(in_img_perturbed)


def postprocess_features(in_features):
    if len(in_features.shape) == 4:
        output = np.sum(in_features, axis=(1, 2))
        output = output / (in_features.shape[1] * in_features.shape[1])
    if len(in_features.shape) == 2:
        output = in_features
    return output


# Classify subclasses (划分小类别)
for attack in ATTACKS:
    ds_name = attack
    print(ds_name)
    x_test = np.load('./advs_new/%s_%s_normal.npy' % (id_name, id_model), allow_pickle=True)
    y_test = np.load('./advs_new/%s_%s_normal_la.npy' % (id_name, id_model), allow_pickle=True)
    x_test_adv = np.load('./advs_new/%s_%s_%s.npy' % (id_name, id_model, ds_name), allow_pickle=True)
    pred_adv = org_model.predict(x_test_adv)
    la_ = np.argmax(pred_adv, axis=1)
    success = la_ != y_test
    class_adv = [[] for i in range(10)]
    class_test = [[] for i in range(10)]
    for i in range(len(pred_adv)):
        if success[i]:
            class_adv[la_[i]].append(x_test_adv[i])
            class_test[y_test[i]].append(x_test[i])
    np.save('./ClassDatas_Adv/%s/%s_%s_%s_ID.npy' % (id_name, id_name, id_model, ds_name),class_test)
    np.save('./ClassDatas_Adv/%s/%s_%s_%s.npy' % (id_name, id_name, id_model, ds_name),class_adv)


#  Collect the features for each layer of target model (搜集Anomaly的每一层neurons的特征)
for attack in ATTACKS:
    ds_name = attack
    class_test = np.load('./ClassDatas_Adv/%s/%s_%s_%s_ID.npy'%(id_name,id_name, id_model,ds_name),allow_pickle=True)
    class_adv = np.load('./ClassDatas_Adv/%s/%s_%s_%s.npy'%(id_name,id_name, id_model,ds_name),allow_pickle=True)

    for lay_i in range(len(layers)):
        lay_id = layers[lay_i]
        # tmp_l = org_model.layers[lay_id]
        modelx_1 = Model(inputs=org_model.input, outputs=tmp_l.output)
        tmp_l = org_model.layers[-2]
        modelx_2 = Model(inputs=org_model.input, outputs=tmp_l.output)

        test_features = [[] for i in range(10)]
        adv_features = [[] for i in range(10)]
        for class_k in range(10):
            a = np.asarray(class_test[class_k])
            features_in = modelx_1.predict(a)
            features_1 = postprocess_features(features_in)
            features_2 = modelx_2.predict(a)
            test_features[class_k] = np.hstack((features_1, features_2))

            b = np.asarray(class_adv[class_k])
            features_in = modelx_1.predict(b)
            features_1 = postprocess_features(features_in)
            features_2 = modelx_2.predict(b)
            adv_features[class_k] = np.hstack((features_1, features_2))
        np.save('./ADDM/datas_layers/%s/%s_%s_%s_layer_%s_ID.npy' % (id_name, id_name, id_model, ds_name,lay_id),test_features)
        np.save('./ADDM/datas_layers/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, ds_name,lay_id),adv_features)
