from utility_db_outliers import load_dataset
from models_lib import load_custom_model_for_ds
from metrics2 import *
# ------------
from utility_methods2 import *

from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression

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
        layers = (3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41, 45, 48, 52, 55, 60, 63, 67, 70, 74, 77, 81, 84, 88, 91, 95, 98,
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


def build_one_class_svm(train_images, test_images=None, ood_images=None, show_eval=False, nu_value=0.1, kernel='rbf', \
                        gamma_value="scale"):
    # kernel=['linear', 'poly', 'rbf', 'sigmoid']
    ss = StandardScaler()
    ss.fit(train_images)
    train_images_ss = ss.transform(train_images)
    clf = svm.OneClassSVM(nu=nu_value, kernel=kernel, gamma=gamma_value)
    clf.fit(train_images_ss)
    return clf, ss


def detect_ood_svm(in_org_model, in_model, in_mix_id_odd):
    features_vector = in_mix_id_odd
    features_vector_norm = in_model[-1].transform(features_vector)
    scores = in_model[-2].score_samples(features_vector_norm)
    return scores


def prepare_models_svm(in_org_model, train_features):
    osvm, ss = build_one_class_svm(train_features)
    f_model = (in_org_model, osvm, ss)
    return f_model


def combine_inliners_outliers(inliers, outliers, i_label=1, o_label=0):
    temp_outliers = outliers
    temp_inliers = inliers
    if i_label == 1:
        i_labels = np.ones(temp_inliers.shape[0])
    else:
        i_labels = np.zeros(temp_inliers.shape[0])

    if o_label == 0:
        o_labels = np.zeros(temp_outliers.shape[0])
    else:
        o_labels = np.ones(temp_outliers.shape[0])

    mixed_labels = np.append(i_labels, o_labels)
    mixed_data = np.vstack((temp_inliers, temp_outliers))

    return mixed_data, mixed_labels




if id_name == 'SVHN':
    num_datas = 60
elif id_name == 'CIFAR10':
    num_datas = 50
elif id_name == "SVHN":
    num_datas = 90

AUC = [[[] for i in range(10)] for i in range(len(ATTACKS))]
scaless = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
for sca in range(10):
    scales = scaless[sca]
    da_logistic = [[[] for i in range(10)] for i in range(len(layers))]
    for k in range(len(ATTACKS)):
        ds_name = ATTACKS[k]
        for lay in range(len(layers)):
            lay_id = layers[lay]
            class_adv = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), allow_pickle=True)

            for i in range(10):
                if (len(class_adv[i])) < num_datas:
                    x = np.asarray(class_adv[i])
                else:
                    x = np.asarray(class_adv[i][:num_datas])
                if k == 0:
                    x0 = x
                else:
                    x1 = x
                    x0 = da_logistic[lay][i]
                    x0 = np.vstack((x0, x1))
                da_logistic[lay][i] = x0
    np.save('./ADDM/logistic/%s/%s_%s_logistic_%s.npy' % (id_name, id_name, id_model, scales), da_logistic)

    # Logistic Regression
    sc_no_normalize = [[[] for i in range(10)] for i in range(len(layers))]
    ds_name = ATTACKS[1]
    da_logistic = np.load('./ADDM/logistic/%s/%s_%s_logistic_%s.npy' % (id_name, id_name, id_model, scales),allow_pickle=True)
    for lay in range(len(layers)):
        lay_id = layers[lay]
        # print('lay:', lay_id)
        train_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_std_%s.npy' % (id_name, id_name, id_model, lay_id, scales), allow_pickle=True)
        test_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_ID_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), allow_pickle=True)
        # adv_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std.npy' % (id_name,id_name, id_model, ds_name, lay_id),allow_pickle=True)
        adv_db = da_logistic[lay]
        lists = []
        for i in range(10):
            if (len(adv_db[i]) != 0):
                lists.append(i)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        FP = dict()
        TN = dict()
        TP = dict()
        FN = dict()
        temp = []
        la = []
        ROC = 0
        numclass = len(adv_db)
        scores = [[] for i in range(10)]
        mixed_labels = [[] for i in range(10)]
        for i in lists:
            train = np.asarray(train_db[i])
            test = np.asarray(test_db[i])
            adv = np.asarray(adv_db[i])
            detect_model = prepare_models_svm(org_model, train)
            mixed_data, mixed_labels[i] = combine_inliners_outliers(test[:num_datas * 15], adv)
            scores[i] = detect_ood_svm(org_model, detect_model, mixed_data)
        for i in lists:
            sc_no_normalize[lay][i] = scores[i]

    # Logistic Regression
    xx = sc_no_normalize

    yy = [[] for i in range(10)]
    for i in range(10):
        for j in range(len(layers)):
            if j == 0:
                y0 = xx[j][i].reshape(-1, 1)
            else:
                y1 = xx[j][i].reshape(-1, 1)
                y0 = np.hstack((y0, y1))
        yy[i] = y0

    lr = LogisticRegression(C=1e5)
    weights_no_normalize = [[] for i in range(10)]
    for i in range(10):
        scores_ = yy[i]
        lr.fit(scores_, mixed_labels[i].reshape(-1, 1))
        weights_no_normalize[i] = lr.coef_[0]
    np.save('./ADDM/logistic/%s/%s_%s_weights_%s.npy' % (id_name, id_name, id_model, scales), weights_no_normalize)

    # weights_no_normalize = np.load('./ADDM/logistic/%s/%s_%s_weights_%s.npy' % (id_name, id_name, id_model, scales),allow_pickle=True)

    ##Detect model三层Concat之后，用SVDD判断的结果  logistic regression  no normalization
    print('%s_%s(%s)' % (id_name, id_model, scales))
    # print('No Normalization的结果：tnr_at_95_tpr / detection_acc / Average AUROC')
    print('tnr_at_95_tpr / detection_acc / Average AUROC:')
    # for attack in ATTACKS:
    for att in range(len(ATTACKS)):
        attack = ATTACKS[att]
        ds_name = attack
        sc_normalize = [[[] for i in range(10)] for i in range(len(layers))]
        sc_no_normalize = [[[] for i in range(10)] for i in range(len(layers))]
        for lay in range(len(layers)):
            lay_id = layers[lay]
            train_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_std_%s.npy' % (id_name, id_name, id_model, lay_id, scales), allow_pickle=True)
            test_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_ID_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), allow_pickle=True)
            adv_db = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), allow_pickle=True)

            lists = []
            for i in range(10):
                if (len(adv_db[i]) != 0):
                    lists.append(i)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            FP = dict()
            TN = dict()
            TP = dict()
            FN = dict()
            temp = []
            la = []
            ROC = 0
            numclass = len(adv_db)

            scores = [[] for i in range(10)]
            mixed_labels = [[] for i in range(10)]
            for i in lists:
                train = np.asarray(train_db[i])
                test = np.asarray(test_db[i])
                ood = np.asarray(adv_db[i])
                detect_model = prepare_models_svm(org_model, train)
                mixed_data, mixed_labels[i] = combine_inliners_outliers(test, ood)
                scores[i] = detect_ood_svm(org_model, detect_model, mixed_data)
            for i in lists:
                sc_no_normalize[lay][i] = scores[i]

        sc_no_norm = [0] * 10
        for class_k in lists:
            for lay in range(len(layers)):
                sc_no_norm[class_k] += sc_no_normalize[lay][class_k] * weights_no_normalize[class_k][lay]

        for i in lists:
            fpr[i], tpr[i] = nums(sc_no_norm[i], mixed_labels[i])
            ROC = auc(fpr[i], tpr[i])
            test = np.asarray(test_db[i])
            lens = test.shape[0]
            FP[i], TN[i], TP[i], FN[i] = ErrorRateAt95Recall1(lens, sc_no_norm[i], mixed_labels[i])
            temp.append(ROC)
            la.append(mixed_labels[i])

        numbers_test = 0
        number_adv = 0
        for i in lists:
            numbers_test += len(test_db[i])
            number_adv += len(adv_db[i])
        numbers_all = numbers_test + number_adv

        ROC_AVE = 0
        for i in range(len(lists)):
            ROC_AVE += (len(la[i]) / numbers_all) * temp[i]

        NFP = 0
        NTN = 0
        NTP = 0
        NFN = 0
        for i in lists:
            NFP += FP[i]
        for i in lists:
            NTN += TN[i]
        for i in lists:
            NTP += TP[i]
        for i in lists:
            NFN += FN[i]
        print('The results of %s:' % (ds_name))
        print(float(NTN) / float(NFP + NTN + 1e-7) * 100, '/',
              (float(NTP) / float(NTP + NFN + 1e-7) + float(NTN) / float(NFP + NTN + 1e-7)) / 2 * 100, '/',
              ROC_AVE * 100)
        AUC[att][sca].append(ROC_AVE * 100)
        # print(AUC)
np.save('./ADDM/AUC/%s/%s_%s_auc.npy' % (id_name, id_name, id_model), AUC)
