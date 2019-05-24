
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math
import time
import numpy as np

from lib.model_MIL import *
from lib.data_loader import *

Flags = tf.app.flags

#Flags.DEFINE_integer('dimension', 100, 'the dimension of instance space')
#Flags.DEFINE_integer('max_iteration', 10000, 'the max iteration of training')
Flags.DEFINE_integer('max_epoch', 10, 'the max epoch of training')
Flags.DEFINE_integer('output_units_of_layer1', 256, 'the number of output unit of dense layer1')
Flags.DEFINE_integer('output_units_of_layer2', 128, 'the number of output unit of dense layer2')
Flags.DEFINE_integer('output_units_of_layer3', 64, 'the number of output unit of dense layer3')
Flags.DEFINE_integer('output_units_of_layer4', 1, 'the number of output unit of dense layer4')
Flags.DEFINE_integer('summary_fre', 5000, 'frequency of print summary')
Flags.DEFINE_integer('decay_period',5000, 'decay period of learning rate')

Flags.DEFINE_float('r_of_LSE', 5,'the parameter r of LSE function')
Flags.DEFINE_float('weight', 10, 'weight of positive instance in loss function')
Flags.DEFINE_float('learning_rate', 0.001,'the learning rate for GD')
Flags.DEFINE_float('decay_rate',0,'the decay rate of training rate')

Flags.DEFINE_string('mode', 'train', 'the mode of the model, train or test on certain data set')
#Flags.DEFINE_string('net_structure', 'MI_net_DS', 'the network structure used to train')
Flags.DEFINE_string('database_name', 'article', 'the network structure used to train')
Flags.DEFINE_string('summary_dir', './lib/summary/', 'The dirctory to output the summary')
Flags.DEFINE_string('record_dir', './lib/summary/', 'The dirctory to output the learning curve')
Flags.DEFINE_string('output_dir', './lib/summary/', 'The dirctory to output the model')
Flags.DEFINE_string('PR_curve_dir', './lib/summary/', 'The dirctory to output the PR curve')

FLAGS = Flags.FLAGS

def test_mode(FLAGS,data,structure,epoch=0):
    if epoch == 0:
        epoch = FLAGS.max_epoch
    test_set = data['test']

    with tf.variable_scope('dense'):
        #tf.layers.Dense objects to build dense layers 
        dense_layers = {}
        if structure == 'MI_net_RC':
            dense_layers['layer1'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer1')
            dense_layers['layer2'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer2')
            dense_layers['layer3'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer3')
            dense_layers['layer4'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer4')
        else:
            dense_layers['layer1'] = tf.layers.Dense(FLAGS.output_units_of_layer1, activation=tf.nn.relu, name='dense_layer1')
            dense_layers['layer2'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer2')
            dense_layers['layer3'] = tf.layers.Dense(FLAGS.output_units_of_layer3, activation=tf.nn.relu, name='dense_layer3')
            dense_layers['layer4'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer4')
        if structure == 'MI_net_DS':
            dense_layers['layer5'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer5')
            dense_layers['layer6'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer6')

    #construct training net work

    learning_rate = FLAGS.learning_rate
    if FLAGS.decay_rate > 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               FLAGS.decay_period, FLAGS.decay_rate, staircase=True)
                                               
    
    
    feature_size = len(data['train'][0][0][0])
    print(feature_size)
    bag_features = tf.placeholder(shape=[None,feature_size],dtype='float32',name='features')
    bag_label = tf.placeholder(shape=[1],dtype='float32',name='label')
    bag = {'features':bag_features, 'label':bag_label}

    network = network_constructor( input_layer = bag, dense_dict = dense_layers, mode=FLAGS.mode, 
                                  net_structure= structure, r_LSE=FLAGS.r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)




    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    tf.Session(config=config)
    print('saver.restore(sess, "/tmp/model.ckpt")')
    







if FLAGS.mode == 'train':
    
    time1=time.time()
    data = data_loader(FLAGS.database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    
    structure='mi_net'
    epoch = FLAGS.max_epoch

    if epoch == 0:
        epoch = FLAGS.max_epoch
    test_set = data['test']

    with tf.variable_scope('dense'):
        #tf.layers.Dense objects to build dense layers 
        dense_layers = {}
        if structure == 'MI_net_RC':
            dense_layers['layer1'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer1')
            dense_layers['layer2'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer2')
            dense_layers['layer3'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer3')
            dense_layers['layer4'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer4')
        else:
            dense_layers['layer1'] = tf.layers.Dense(FLAGS.output_units_of_layer1, activation=tf.nn.relu, name='dense_layer1')
            dense_layers['layer2'] = tf.layers.Dense(FLAGS.output_units_of_layer2, activation=tf.nn.relu, name='dense_layer2')
            dense_layers['layer3'] = tf.layers.Dense(FLAGS.output_units_of_layer3, activation=tf.nn.relu, name='dense_layer3')
            dense_layers['layer4'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer4')
        if structure == 'MI_net_DS':
            dense_layers['layer5'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer5')
            dense_layers['layer6'] = tf.layers.Dense(FLAGS.output_units_of_layer4, name='dense_layer6')

    #construct training net work

    learning_rate = FLAGS.learning_rate
    if FLAGS.decay_rate > 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               FLAGS.decay_period, FLAGS.decay_rate, staircase=True)
                                               
    
    
    feature_size = len(data['train'][0][0][0])
    print(feature_size)
    bag_features = tf.placeholder(shape=[None,feature_size],dtype='float32',name='features')
    bag_label = tf.placeholder(shape=[1],dtype='float32',name='label')
    bag = {'features':bag_features, 'label':bag_label}

    network = network_constructor( input_layer = bag, dense_dict = dense_layers, mode=FLAGS.mode, 
                                  net_structure= structure, r_LSE=FLAGS.r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)




    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    tf.Session(config=config)
    print('saver.restore(sess, "/tmp/model.ckpt")')
