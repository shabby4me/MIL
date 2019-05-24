from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math
import time
import numpy as np
import gc

from random import shuffle

from lib.model_MIL import *
from lib.data_loader import *


Flags = tf.app.flags

#Flags.DEFINE_integer('dimension', 100, 'the dimension of instance space')
#Flags.DEFINE_integer('max_iteration', 10000, 'the max iteration of training')
Flags.DEFINE_integer('max_epoch', 30, 'the max epoch of training')
Flags.DEFINE_integer('output_units_of_layer1', 256, 'the number of output unit of dense layer1')
Flags.DEFINE_integer('output_units_of_layer2', 128, 'the number of output unit of dense layer2')
Flags.DEFINE_integer('output_units_of_layer3', 64, 'the number of output unit of dense layer3')
Flags.DEFINE_integer('output_units_of_layer4', 1, 'the number of output unit of dense layer4')
Flags.DEFINE_integer('summary_fre', 30000, 'frequency of print summary')
Flags.DEFINE_integer('decay_period',5000, 'decay period of learning rate')

Flags.DEFINE_float('r_of_LSE', 0.2,'the parameter r of LSE function')
Flags.DEFINE_float('weight', 30, 'weight of positive instance in loss function')
Flags.DEFINE_float('learning_rate', 0.001,'the learning rate for GD')
Flags.DEFINE_float('decay_rate',0,'the decay rate of training rate')
Flags.DEFINE_float('dropout_rate',0.5,'the dropout rate of training')

Flags.DEFINE_string('mode', 'train', 'the mode of the model, train or test on certain data set')
#Flags.DEFINE_string('net_structure', 'MI_net_DS', 'the network structure used to train')
Flags.DEFINE_string('database_name', 'RNAi', 'the network structure used to train')
Flags.DEFINE_string('summary_dir', './lib/summary/', 'The dirctory to output the summary')
Flags.DEFINE_string('record_dir', './lib/summary/', 'The dirctory to output the learning curve')
Flags.DEFINE_string('output_dir', './lib/summary/', 'The dirctory to output the model')
Flags.DEFINE_string('PR_curve_dir', './lib/summary/', 'The dirctory to output the PR curve')
Flags.DEFINE_string('ckpt_dir', './lib/stage 2-better/epoch_test_by_loss/', 'The directory to restore checkpoint from')
Flags.DEFINE_string('ckpt_name', 'model.ckpt-1114912', 'the name of checkpoint' )

FLAGS = Flags.FLAGS




def train(FLAGS,data,structure='',database_name='',epoch=0,r_of_LSE=0):
    if epoch == 0:
        epoch = FLAGS.max_epoch
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
    if database_name=='':
        database_name=FLAGS.database_name
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
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight,
                                   dropout_rate=FLAGS.dropout_rate)


    structure = database_name + '_' + structure + '_r_' + str(r_of_LSE)

    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    curve_summary = open(FLAGS.record_dir+structure+'/record.txt','w+')
    loss_PRarea_epoch = open(FLAGS.summary_dir+structure+'/loss.txt','w+')
    acc_summary.write('net_structure:'+ structure+'\n')
    acc_summary.write('max_epoch:'+str(epoch)+'\n')
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    acc_summary.write('input r:'+str(r_of_LSE)+'\n' )
    acc_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    curve_summary.write('net_structure:'+ structure+'\n')
    curve_summary.write('max_epoch:'+str(epoch)+'\n')
    curve_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    curve_summary.write('input r:'+str(r_of_LSE)+'\n' )
    curve_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        '''
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config,
                                                           checkpoint_dir=FLAGS.summary_dir)) as sess:
    '''                                                   
        sess.run(tf.global_variables_initializer())

        print('Optimization start!')
        t1 = time.time()
        ac_loss=0
        loss_epochs=[]
        PR_area_epochs=[]
        eval_loss_epochs=[]
        for epoch in range(epoch):
            loss_epoch=0
            ac_loss=0
            for step in range(len(data['train'])):
                
                result = sess.run(network, feed_dict={bag_features:data['train'][step][0],
                                                      bag_label:data['train'][step][1]})
                
                
                ac_loss = ac_loss+result['loss']
                loss_epoch = loss_epoch+result['loss']
                 
            print('In epoch', epoch,'the loss is', loss_epoch)
            
            acc_summary.write('\nIn epoch'+str(epoch)+'the loss is'+str(loss_epoch)+'\n')
            loss_epochs.append(loss_epoch)
            PR_area_epochs.append(PR_area(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label, mode='eval'))
            eval_loss_epochs.append(eval_set_loss(model=network['loss'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label))
            if epoch > 5 and (1+epoch)%5 == 0:
                saver.save(sess, FLAGS.output_dir+structure+'/model.ckpt', global_step=(1+epoch)*(1+step))     
        test(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file = acc_summary,curve_file=curve_summary)
        t2 = time.time()
        saver.save(sess, FLAGS.output_dir+structure+'/model.ckpt', global_step=(1+epoch)*(1+step))   

        for i in range(len(loss_epochs)):
            acc_summary.write(str(loss_epochs[i])+',')
            loss_PRarea_epoch.write(str(loss_epochs[i])+',')
        acc_summary.write('\n')
        loss_PRarea_epoch.write('\n')
        for i in range(len(eval_loss_epochs)):
            acc_summary.write(str(eval_loss_epochs[i])+',')
            loss_PRarea_epoch.write(str(eval_loss_epochs[i])+',')
        acc_summary.write('\n')
        loss_PRarea_epoch.write('\n')
        for i in range(len(PR_area_epochs)):
            acc_summary.write(str(PR_area_epochs[i])+',')
            loss_PRarea_epoch.write(str(PR_area_epochs[i])+',')

        
        print('run time:', (t2-t1)/60, 'min')
        print('Save the checkpoint')
        acc_summary.close()
        curve_summary.close()
        loss_PRarea_epoch.close()
        
        t1 = time.time()
        with open(FLAGS.PR_curve_dir+structure+'/PR_curve.txt','w+') as file:
            PR_curve(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
        t2 = time.time()
        print('PR curve run time:', (t2-t1)/60, 'min')
    print('Optimization done')


def train_final(FLAGS,data,structure='',database_name='',epoch=0,r_of_LSE=0):
    if epoch == 0:
        epoch = FLAGS.max_epoch
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
    if database_name=='':
        database_name=FLAGS.database_name
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
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight,
                                   dropout_rate=FLAGS.dropout_rate)


    structure = database_name + '_' + structure + '_r_' + str(r_of_LSE)

    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    curve_summary = open(FLAGS.record_dir+structure+'/record.txt','w+')
    loss_PRarea_epoch = open(FLAGS.summary_dir+structure+'/loss.txt','w+')
    acc_summary.write('net_structure:'+ structure+'\n')
    acc_summary.write('max_epoch:'+str(epoch)+'\n')
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    acc_summary.write('input r:'+str(r_of_LSE)+'\n' )
    acc_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    curve_summary.write('net_structure:'+ structure+'\n')
    curve_summary.write('max_epoch:'+str(epoch)+'\n')
    curve_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    curve_summary.write('input r:'+str(r_of_LSE)+'\n' )
    curve_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        '''
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config,
                                                           checkpoint_dir=FLAGS.summary_dir)) as sess:
    '''                                                   
        sess.run(tf.global_variables_initializer())

        print('Optimization start!')
        t1 = time.time()
        ac_loss=0
        loss_epochs=[]
        PR_area_epochs=[]
        eval_loss_epochs=[]
        for epoch in range(epoch):
            loss_epoch=0
            ac_loss=0
            for step in range(len(data['eval'])):
                
                result = sess.run(network, feed_dict={bag_features:data['eval'][step][0],
                                                      bag_label:data['eval'][step][1]})
                
                
                ac_loss = ac_loss+result['loss']
                loss_epoch = loss_epoch+result['loss']

            for step in range(len(data['train'])):
                
                result = sess.run(network, feed_dict={bag_features:data['train'][step][0],
                                                      bag_label:data['train'][step][1]})
                
                
                ac_loss = ac_loss+result['loss']
                loss_epoch = loss_epoch+result['loss']
                 
            print('In epoch', epoch,'the loss is', loss_epoch)
            
              
        
        t2 = time.time()
        saver.save(sess, FLAGS.output_dir+structure+'/model.ckpt', global_step=(1+epoch)*(1+step))   

        

        
        print('run time:', (t2-t1)/60, 'min')
        print('Save the checkpoint')
        acc_summary.close()
        curve_summary.close()
        loss_PRarea_epoch.close()
        
        t1 = time.time()
        with open(FLAGS.PR_curve_dir+structure+'/PR_curve.txt','w+') as file:
            PR_curve(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
        t2 = time.time()
        print('PR curve run time:', (t2-t1)/60, 'min')
    print('Optimization done')


def evaluation(FLAGS,data,structure,r_of_LSE=0,ckpt_name=''):
    
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
    if ckpt_name=='':
        ckpt_name=FLAGS.ckpt_name
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
    print('feature size:',feature_size)
    bag_features = tf.placeholder(shape=[None,feature_size],dtype='float32',name='features')
    bag_label = tf.placeholder(shape=[1],dtype='float32',name='label')
    bag = {'features':bag_features, 'label':bag_label}

    network = network_constructor( input_layer = bag, dense_dict = dense_layers, mode=FLAGS.mode, 
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)


    structure = structure + '_r_' + str(r_of_LSE)

    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    
    acc_summary.write('net_structure:'+ structure+'\n')
    
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    acc_summary.write('r input:'+str(r_of_LSE)+'\n' )
    acc_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess,FLAGS.ckpt_dir+structure+'/'+ckpt_name)
        t1 = time.time()
        with open(FLAGS.PR_curve_dir+structure+'/PR_curve.txt','w+') as file:
            PR_curve(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
        t2 = time.time()
        print('PR curve run time:', (t2-t1)/60, 'min')
    print('Evaluation done')
    acc_summary.close()

def check(FLAGS,data,structure,r_of_LSE=0):
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
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
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)


    structure = structure + '_r_' + str(r_of_LSE)

    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    
    acc_summary.write('net_structure:'+ structure+'\n')
    
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    acc_summary.write('r input:'+str(r_of_LSE)+'\n' )
    acc_summary.write('r of LSE:'+str(1/r_of_LSE)+'\n' )
    
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess,FLAGS.ckpt_dir+structure+'/'+FLAGS.ckpt_name)
        t1 = time.time()
        with open(FLAGS.PR_curve_dir+structure+'/score_detail.txt','w+') as file:
            instance_inf(model=network['raw'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
        t2 = time.time()
        print('PR curve run time:', (t2-t1)/60, 'min')
    print('Evaluation done')
    acc_summary.close()



def train_detail(FLAGS,data,structure,epoch=0,r_of_LSE=0):
    if epoch == 0:
        epoch = FLAGS.max_epoch
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
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
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)



    structure= structure+'_detail'
    
    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    curve_summary = open(FLAGS.record_dir+structure+'/record.txt','w+')
    acc_summary.write('net_structure:'+ structure+'\n')
    acc_summary.write('max_epoch:'+str(epoch)+'\n')
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    curve_summary.write('net_structure:'+ structure+'\n')
    curve_summary.write('max_epoch:'+str(epoch)+'\n')
    curve_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        '''
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config,
                                                           checkpoint_dir=FLAGS.summary_dir)) as sess:
    '''                                                   
        sess.run(tf.global_variables_initializer())

        print('Optimization start!')
        t1 = time.time()
        ac_loss=0
        
        for epoch in range(epoch):
            loss_epoch=0
            for step in range(len(data['train'])):
                #fetches the training process datas to be recorded

                #save the result by save_frequency

                result = sess.run(network, feed_dict={bag_features:data['train'][step][0],
                                                      bag_label:data['train'][step][1]})
                '''
                print('step:',step,'label:',data['train'][step][1],'result[\'net\']=',result['net'],'result[\'loss\']=', result['loss'])
                if result['net']>1:
                    print('bag_value is larger than 1')
                
                
                if step%100 == 0:
                    print('label',data['label'][step])
                
                print('epoch',epoch,'setp ',step ,':','loss:',result['loss'],', label',data['train'][step][1], ', raw score:',result['raw'])
                '''
                ac_loss = ac_loss+result['loss']
                loss_epoch = loss_epoch+result['loss']
                
                if (1+step)%FLAGS.summary_fre == 0:
                    #saver.save(sess, FLAGS.output_dir+structure+'/model_'+str(epoch)+'.ckpt', global_step=(1+epoch)*(1+step))
                    #print('model saved')
                    print('epoch',epoch,'setp ',step ,':','loss:',result['loss'],', label',data['train'][step][1], ', ac_loss:',ac_loss)
                    #acc_summary.write('epoch:'+str(epoch)+', setp'+str(step) +':'+', loss:'+str(result['loss'])+', label'+str(data['train'][step][1])+ 'ac_loss:'+str(ac_loss)+'\n')
                    #test(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label, file=acc_summary,curve_file=curve_summary)
                    ac_loss = 0
                

                
                
            print('In epoch', epoch,'the loss is', loss_epoch)
            acc_summary.write('\nIn epoch'+str(epoch)+'the loss is'+str(loss_epoch)+'\n')

            with open(FLAGS.PR_curve_dir+structure+'/PR_curve_'+str(epoch)+'.txt','w+') as file:
                PR_curve(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
            print('Save the checkpoint')
            
            
        test(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file = acc_summary,curve_file=curve_summary)
        t2 = time.time()
        
        print('run time:', (t2-t1)/60, 'min')
        
        acc_summary.close()
        curve_summary.close()
        
        
        
    print('Optimization done')

def train_great_detail(FLAGS,data,structure,epoch=0,r_of_LSE=0):
    if epoch == 0:
        epoch = FLAGS.max_epoch
    if r_of_LSE == 0:
        r_of_LSE=FLAGS.r_of_LSE
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
                                  net_structure= structure, r_LSE=r_of_LSE, learning_rate=learning_rate, weight=FLAGS.weight)



    structure= structure+'_great_detail'
    
    saver = tf.train.Saver()
    os.makedirs(os.path.dirname(FLAGS.summary_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.record_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.PR_curve_dir+structure+'/'), exist_ok=True)
    os.makedirs(os.path.dirname(FLAGS.output_dir+structure+'/'), exist_ok=True)

    
    acc_summary = open(FLAGS.summary_dir+structure+'/log.txt','w+')
    curve_summary = open(FLAGS.record_dir+structure+'/record.txt','w+')
    acc_summary.write('net_structure:'+ structure+'\n')
    acc_summary.write('max_epoch:'+str(epoch)+'\n')
    acc_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    curve_summary.write('net_structure:'+ structure+'\n')
    curve_summary.write('max_epoch:'+str(epoch)+'\n')
    curve_summary.write('learning_rate:'+str(FLAGS.learning_rate)+'\n' )
    if FLAGS.decay_rate > 0:
        acc_summary.write('decay_period'+str(FLAGS.decay_period)+ ', decay_rate'+str(FLAGS.decay_rate)+'\n')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        '''
    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config,
                                                           checkpoint_dir=FLAGS.summary_dir)) as sess:
    '''                                                   
        sess.run(tf.global_variables_initializer())

        print('Optimization start!')
        t1 = time.time()
        ac_loss=0
        
        for epoch in range(epoch):
            loss_epoch=0
            for step in range(len(data['train'])):
                #fetches the training process datas to be recorded

                #save the result by save_frequency

                result = sess.run(network, feed_dict={bag_features:data['train'][step][0],
                                                      bag_label:data['train'][step][1]})
                if step<4 and epoch>2:
                    with open(FLAGS.PR_curve_dir+structure+'/PR_curve_'+str(epoch)+'_'+str(step)+'.txt','w+') as file:
                        PR_curve(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file=file)
                
                

                
                
            print('In epoch', epoch,'the loss is', loss_epoch)
            
            
            
        test(model=network['net'], data=data, sess=sess, bag_features=bag_features, bag_label=bag_label,file = acc_summary,curve_file=curve_summary)
        t2 = time.time()
        
        print('run time:', (t2-t1)/60, 'min')
        
        acc_summary.close()
        curve_summary.close()
        
        
        
    print('Optimization done')

if FLAGS.mode == 'train':

    #train(FLAGS,data,structure='',database_name='',epoch=0,r_of_LSE=0)
    r = 0.2
    

    '''
    
    #for KO , omission and RNAi dataset
    for database_name in ['omission','RNAi']:
        time1=time.time()
        data = data_loader(database_name,cross_mod=1)
        time2=time.time()
        print('load data and process cost', (time2-time1)/60, 'min')

        #shuffle the training sequence
        #shuffle(data['train'])
        
        train(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r)
        train(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r)
        train(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r)#
        train(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r)
        train(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r)#
        train(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r)
        del data
        gc.collect()
    
    
    

    '''
    
    
    '''
    #without dropout
    database_name='KO'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    #shuffle the training sequence
    #shuffle(data['train'])
    train(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=7)
    train(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=7)
    train(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=11)#
    train(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=5)
    train(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=13)#
    train(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=14)
    del data
    gc.collect()

    '''
    
    #with dropout
    database_name='KO'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    #shuffle the training sequence
    #shuffle(data['train'])
    train(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=14)
    train(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=15)
    train(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=17)#
    train(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=12)
    train(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=26)#
    train(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=25)
    del data
    gc.collect()
    
    '''
    database_name='RNAi'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    train(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=10)
    train(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=2)
    train(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=4)#
    train(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=2)
    train(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=6)#
    train(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=4)
    del data
    gc.collect()
    
    database_name='omission'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    train(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=5)
    train(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=4)
    train(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=21)#
    train(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=8)
    train(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=2)#
    train(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=26)
    del data
    gc.collect()
    '''
    '''
    #with dropout
    database_name='RNAi'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    train_final(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=25)
    train_final(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=19)
    train_final(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=30)#
    train_final(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=19)
    train_final(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=2)#
    train_final(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=17)
    del data
    gc.collect()
    
    database_name='omission'
    time1=time.time()
    data = data_loader(database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')
    train_final(FLAGS,data,database_name=database_name,structure='mi_net',r_of_LSE=r,epoch=19)
    train_final(FLAGS,data,database_name=database_name,structure='MI-net',r_of_LSE=r,epoch=26)
    train_final(FLAGS,data,database_name=database_name,structure='MI_net_DS',r_of_LSE=r,epoch=27)#
    train_final(FLAGS,data,database_name=database_name,structure='MI_net_RC',r_of_LSE=r,epoch=26)
    train_final(FLAGS,data,database_name=database_name,structure='mi_net_simple',r_of_LSE=r,epoch=21)#
    train_final(FLAGS,data,database_name=database_name,structure='MI-net_simple',r_of_LSE=r,epoch=9)
    del data
    gc.collect()
    
    '''

    
    
   
if FLAGS.mode == 'eval':
    
    time1=time.time()
    data = data_loader(FLAGS.database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')

    r=5
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)
    evaluation(FLAGS,data,'mi_net_simple',r_of_LSE=r,epoch=0)


if FLAGS.mode == 'check':
    time1=time.time()
    data = data_loader(FLAGS.database_name)
    time2=time.time()
    print('load data and process cost', (time2-time1)/60, 'min')

    
    for r in [0.2,1,5]:        
        check(FLAGS,data,'mi_net_simple',r_of_LSE=r)#
        
        
