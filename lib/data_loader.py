from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
import random
import gc

def data_loader(database_name,cross_mod=0):
    #load for musk dataset for test
    if database_name.startswith('musk'):
        data = sio.loadmat('./dataset/'+database_name+'.mat')
        features = data['x']['data'][0,0]
        bag_names = data['x']['ident'][0,0]['milbag'][0,0]
        labels = data['x']['nlab'][0,0][:,0] - 1

        #normalization
        feature_mean = np.mean(features, axis=0, keepdims=True)+1e-6
        feature_std = np.std(features, axis=0, keepdims=True)+1e-6
        features = np.divide(features-feature_mean, feature_std)

        input_instance_index = {}
        for id, bag_name in enumerate(bag_names):
            if bag_name in input_instance_index:
                input_instance_index[bag_name].append(id)
            else:
                input_instance_index[bag_name] = [id]
        bag_features = []
        for bag_name, instances_index in input_instance_index.items():
            bag_feature = ([],[])
            ins_lab = []
            for instance_index in instances_index:
                bag_feature[0].append(features[instance_index])
                ins_lab.append((labels[instance_index]))
            bag_feature[1].append(np.max(ins_lab).reshape([]))
            bag_features.append(bag_feature)

        num_of_bags = len(bag_features)
        random_list = list(range(num_of_bags))
        random.shuffle(random_list)
        train_list = random_list[2*int(num_of_bags/10):]
        eval_list = random_list[int(num_of_bags/10):2*int(num_of_bags/10)]
        test_list = random_list[0:int(num_of_bags/10)]

        dataset = {}
        dataset['train'] = [bag_features[bag] for bag in train_list]
        dataset['eval'] = [bag_features[bag] for bag in eval_list]
        dataset['test'] = [bag_features[bag] for bag in test_list]
        print('data load finished,', len(bag_features),'bags are loaded')
    #load the KO dataset
    if database_name == 'KO':
        
        fname = 'C:/Users/lyz62/Downloads/Multiple Instance Data/KO/Yaozhong/KO.data'
        
        
        
        cnt = 0
        with open(fname) as f:
            line = f.readline()
            data = []
            for i in range(64420): # number counted by preprocessing
                data.append(([],[],[])) 
            while not line == '':
                # delete period and newline at the end then split by comma
                line = line.strip().strip('.').split(',')
                # transfer to int
                for step in range(len(line)):
                    line[step] = int(line[step])
                # merge into bags
                label = line.pop()
                bag_name = line[0]
                features = line[2:]
                data[bag_name][0].append(features)
                data[bag_name][2].append(label)
                line = f.readline()
                cnt = cnt+1
                if cnt%10000 == 0:
                    print('10000 lines read!')
        for bag in data:
            bag[1].append(max(bag[2]))
            

        num_of_bags = len(data)
        train_set = []
        evaluation_set = []
        test_set = []
        #divide into test, training and tuning set
        for i in range(len(data)-1,-1,-1):
            if i>= 49801 and i<=62121:
                test_set.append(data.pop())
            else:
                if i%10 == cross_mod:
                    evaluation_set.append(data.pop())
                else:
                    train_set.append(data.pop())

        
        dataset = {'train':train_set,'test':test_set,'eval':evaluation_set}

        print('data load finished,', num_of_bags,'bags are loaded')

        return dataset

    if database_name == 'omission':
        
        fname = 'C:/Users/lyz62/Downloads/Multiple Instance Data/omission/Yaozhong/omission.data'
        labels = []
        bag_names = []
        features = []
        cnt = 0
        with open(fname) as f:
            line = f.readline()
            data = []
            for i in range(64416): # number counted by preprocessing
                data.append(([],[],[])) 
            while not line == '':
                # delete period and newline at the end then split by comma
                line = line.strip().strip('.').split(',')
                # transfer to int
                for step in range(len(line)):
                    line[step] = int(line[step])
                # merge into bags
                label = line.pop()
                bag_name = line[0]
                features = line[2:]
                data[bag_name][0].append(features)
                data[bag_name][2].append(label)
                line = f.readline()
                cnt = cnt+1
                if cnt%10000 == 0:
                    print('10000 lines read!')
        for bag in data:
            bag[1].append(max(bag[2]))
        

        num_of_bags = len(data)
        train_set = []
        evaluation_set = []
        test_set = []

        for i in range(len(data)-1,-1,-1):
            if i>= 49796 and i<=62117:
                test_set.append(data.pop())
            else:
                if i%10 == cross_mod:
                    evaluation_set.append(data.pop())
                else:
                    train_set.append(data.pop())

        
        dataset = {'train':train_set,'test':test_set,'eval':evaluation_set}

        print('data load finished,', num_of_bags,'bags are loaded')

        return dataset

    if database_name == 'RNAi':
        
        fname = 'C:/Users/lyz62/Downloads/Multiple Instance Data/RNAi/Yaozhong/RNAi.data'        
        labels = []
        bag_names = []
        features = []
        cnt = 0
        with open(fname) as f:
            line = f.readline()
            data = []
            for i in range(64419): # number counted by preprocessing
                data.append(([],[],[])) 
            while not line == '':
                # delete period and newline at the end then split by comma
                line = line.strip().strip('.').split(',')
                # transfer to int
                for step in range(len(line)):
                    line[step] = int(line[step])
                # merge into bags
                label = line.pop()
                bag_name = line[0]
                features = line[2:]
                data[bag_name][0].append(features)
                data[bag_name][2].append(label)
                line = f.readline()
                cnt = cnt+1
                if cnt%10000 == 0:
                    print('10000 lines read!')
        for bag in data:
            bag[1].append(max(bag[2]))
            

        num_of_bags = len(data)
        train_set = []
        evaluation_set = []
        test_set = []

        for i in range(len(data)-1,-1,-1):
            if i>= 49800 and i<=62119:
                test_set.append(data.pop())
            else:
                if i%10 == cross_mod:
                    evaluation_set.append(data.pop())
                else:
                    train_set.append(data.pop())

        
        dataset = {'train':train_set,'test':test_set,'eval':evaluation_set}

        print('data load finished,', num_of_bags,'bags are loaded')

        return dataset





