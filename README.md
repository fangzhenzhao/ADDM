# ADDM：Adversarial Detection from Derived Models

## Software dependencies 

1. The source code is written in Python with Tensorflow (including Keras) and Jupyter. Please make sure you have these tools installed properly in your system.
   - The version of Python used: 3.6.9
   - The version of Tensorflow used: 1.12.0 / 2.3.1
   - The version of  Keras used: 2.2.4
   - The version of the notebook server is: 6.0.1

## Datasets

MNIST / CIFAR-10 / SVHN. 

By *id_name*,  you can set the dataset. The list of dataset is: ("MNIST", "CIFAR-10", "SVHN").

## Models

We pre-trained five neural networks (1) LeNet on MNIST;  (2) two VGG and ResNet models trained on Cifar10 and SVHN, respectively.  

By *id_model*, you can set the ID model. The list of ID models is: ("LeNet", "VGG", "ResNet")

You may download pre-trained models from 
https://1drv.ms/u/s!AmxoS1DPJxUZhyV8OAIvz6Yqji_w?e=TzoGWN

Please place them to './saved_models/'.

## Detecting Adversarial Samples

First, you need generate adversarial sample, see the link  https://github.com/xingjunm/lid_adversarial_subspace_detection.

Then, to experiment with our approach,  please perform the following steps:

- Execute "preprocessing.py" to get the features for each layer of target model;
- Execute "ADDM_neurons.py" to get the features of derived models;
- Execute "ADDM_variance.py" to get the useful features for evaluation ;
- Execute "ADDM_variance.py" to get the final results.
