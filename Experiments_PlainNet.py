# -*- coding: utf-8 -*-

""" Deep Residual Network.
Applying a Deep Residual Network to CIFAR-10 Dataset classification task.
References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import division, print_function, absolute_import

import tflearn
import argparse
import os

parser = argparse.ArgumentParser(description='ResNet.')
parser.add_argument('-n', metavar='--nBlocks',nargs=1,type=int,required=True,
                    help='Parametro n segun paper. #layers = 2 + (6n)')
parser.add_argument('-e', metavar='--nBlocks',nargs=1,type=int,required=True,
                    help='Cantidad de epochs')
parser.add_argument('-a', metavar='--act',nargs=4,type=str,required=False,
                    help='Configuracion de activaciones')

parser.add_argument('-exp', metavar='--experimento',nargs=2,type=str,required=False,
                    help='Id de experimento y funcion de activacion a aplicar')

parser.add_argument('-b', metavar='--block-mode',nargs=1,type=str,required=False,
                    help='Id de experimento y funcion de activacion a aplicar')

parser.add_argument('-suf2', metavar='--suffix2',nargs=1,type=str,required=False,
                    help='Sufijo 2 al nombre del modelo')

args = vars(parser.parse_args())

# Residual blocks
# 20 layers: n=3
# 32 layers: n=5,
# 44 layers: n=7,
# 56 layers: n=9,
# 110 layers: n=18
experimentos = {
    'default':['relu','relu','relu','relu'],
    'E1':[None,None,None,None],
    'E2':[None,None,'relu','relu'],
    'E3':['relu','relu',None,None],
    'E4':[None,'relu','relu','relu'],
    'E5':[None,'relu',None,None],
    'E6':['relu',None,None,None]
}
n = args['n'][0]
EPOCHS=args['e'][0]
activations = args['a']
experimento = args['exp']

suffix2 = args['suf2']


if args['b'] == None:
    block_mode = 'reference'
else:
    block_mode = args['b'][0]

if block_mode.upper() == 'REFERENCE':
    bn_position = 'before'
    last_actv = True
elif block_mode.upper() == 'BN_AA':
    bn_position = 'after'
    last_actv = True
elif block_mode.upper() == 'NO_RELU':
    bn_position = 'before'
    last_actv = False

if experimento != None:
    id_exp = experimento[0]
    activ_fn = experimento[1]
    activations = experimentos[id_exp]
    activations = [activ_fn if v is None else v for v in activations]
    suffix = id_exp+'_'+activ_fn
elif activations == None:
    activations = experimentos['default']
    suffix = 'default'
else:
    suffix = 'acts='+'_'.join(activations)

layers = (n*6)+2

# Data loading
from tflearn.datasets import cifar10
from modulos import my_plain_block


(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y,nb_classes=10)
testY = tflearn.data_utils.to_categorical(testY,nb_classes=10)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net,activation=activations[0])
net = my_plain_block(net, n, 16,activation=activations[1],bn_position=bn_position,last_actv=last_actv)
net = my_plain_block(net, 1, 32, downsample=True,activation=activations[2],bn_position=bn_position,last_actv=last_actv)
net = my_plain_block(net, n-1, 32,activation=activations[2],bn_position=bn_position,last_actv=last_actv)
net = my_plain_block(net, 1, 64, downsample=True,activation=activations[3],bn_position=bn_position,last_actv=last_actv)
net = my_plain_block(net, n-1, 64,activation=activations[3],bn_position=bn_position,last_actv=last_actv)
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='./checkpoints/', tensorboard_verbose=0,tensorboard_dir='log',
                    clip_gradients=0.)


model_name = 'PLAIN_'+str(layers)+'_layers_'+str(EPOCHS)+'_epochs_BM='+block_mode+'_'+suffix

if suffix2 != None:
    model_name = model_name +'_'+suffix2[0]

model.fit(X, Y, n_epoch=EPOCHS, validation_set=(testX, testY),
          snapshot_epoch=True,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar10_'+model_name)

model.save('./models_trained/resnet_cifar10_'+model_name+'_model')