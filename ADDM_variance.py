from utility_methods2 import *
import numpy as np

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

scaless = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
ratioss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
for kk in range(10):
    print('第%s次:' % (kk))
    ratios = ratioss[kk]
    scales = scaless[kk]

    for lay_i in range(len(layers)):
        lay_id = layers[lay_i]
        # print('第%s层的结果：'%(lay_id))
        index_order = np.load(
            './ADDM/Index_Order/%s/%s_%s_index_order_layer_%s.npy' % (id_name, id_name, id_model, lay_id),
            allow_pickle=True)
        activate_neus_ind = [[] for i in range(10)]
        for class_k in range(10):
            xx = index_order[class_k]  # std降序排列，最前面的std最小
            b = xx[int(len(xx) * ratios):]
            activate_neus_ind[class_k].append(b.squeeze())
        activate_neus_ind = np.asarray(activate_neus_ind).squeeze()

        # # train上的std
        neu_train = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_0.npy' % (id_name, id_name, id_model, lay_id),allow_pickle=True)
        train_neus_cat = [[] for i in range(10)]
        for class_k in range(10):
            num_sample = neu_train[class_k].shape[0]
            for j in range(num_sample):
                neu = []
                la_0 = activate_neus_ind[class_k]
                for k in range(len(la_0)):
                    ne = neu_train[class_k][j][la_0[k]]
                    neu.append(ne)
                train_neus_cat[class_k].append(neu)

        xx = [[] for i in range(10)]
        for i in range(10):
            xx[i] = np.asarray(train_neus_cat[i])
        np.save('./ADDM/neurons_layers_cat/%s/%s_%s_train_layer_%s_std_%s.npy' % (id_name,id_name, id_model, lay_id,scales), np.asarray(xx))

        # 攻击成功的样本以及对应的Adversarial Sample的std
        for attack in ATTACKS:
            ds_name = attack
            neu_test = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_ID.npy' % (id_name, id_name, id_model, ds_name, lay_id), allow_pickle=True)
            test_neus_cat = [[] for i in range(10)]
            for class_k in range(10):
                num_sample = neu_test[class_k].shape[0]
                for j in range(num_sample):
                    neu = []
                    la_0 = activate_neus_ind[class_k]
                    for k in range(len(la_0)):
                        ne = neu_test[class_k][j][la_0[k]]
                        neu.append(ne)
                    test_neus_cat[class_k].append(neu)

            xx = [[] for i in range(10)]
            for i in range(10):
                xx[i] = np.asarray(test_neus_cat[i])
            np.save('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_ID_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), np.asarray(xx))

            neu_adv = np.load('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id), allow_pickle=True)
            adv_neus_cat = [[] for i in range(10)]
            for class_k in range(10):
                num_sample = neu_adv[class_k].shape[0]
                for j in range(num_sample):
                    neu = []
                    la_0 = activate_neus_ind[class_k]
                    for k in range(len(la_0)):
                        ne = neu_adv[class_k][j][la_0[k]]
                        neu.append(ne)
                    adv_neus_cat[class_k].append(neu)

            xx = [[] for i in range(10)]
            for i in range(10):
                xx[i] = np.asarray(adv_neus_cat[i])
            np.save('./ADDM/neurons_layers_cat/%s/%s_%s_%s_layer_%s_std_%s.npy' % (id_name, id_name, id_model, ds_name, lay_id, scales), np.asarray(xx))

