from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np



def log_sum_exp_pooling(logits,r):
    exp_logits = tf.exp(tf.scalar_mul(tf.constant(1/r,dtype='float32'),logits))
    sum_of_exp_logits = tf.scalar_mul(tf.constant(r,dtype='float32'),tf.log(tf.reduce_mean(exp_logits)))
    return sum_of_exp_logits



def loss(y_true,y_pred,weight):
    binary_crossentropy = -(weight*y_true*tf.log(y_pred+1e-4)+(1-y_true)*tf.log(1-y_pred+1e-4))
    return binary_crossentropy

def mil_max_pooling(instances,r):
    exp_ins = tf.exp(tf.scalar_mul(tf.constant(1/r,dtype='float32'),instances))
    pooling_ins = tf.scalar_mul(tf.constant(r,dtype='float32'),tf.log(1e-4+tf.reduce_mean(exp_ins,0)))
    return pooling_ins


def network_constructor( input_layer, dense_dict, mode, net_structure, r_LSE, learning_rate, weight, dropout_rate=0):
    
    #mi_net constructor
    if net_structure == 'mi_net':
        
        net = dense_dict['layer1'](input_layer['features'])
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense1:',net)
        net = dense_dict['layer2'](net)
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense2:',net)
        net = dense_dict['layer3'](net)
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense3:',net)
        net = dense_dict['layer4'](net)
        print('dense4:',net)
        
        net = tf.sigmoid(net)
        raw_score = net
        print('instance_score:', net)
        '''
        net = dense_dict['layer4'](input_layer['features'])
        net = tf.sigmoid(net)
        print('dense1:',net)
        '''
        net = log_sum_exp_pooling(logits=net,r=r_LSE)
        print('bag_score:', net)
        print('network construction complete')

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network={'net':net, 'loss':loss_func, 'train':train, 'raw':raw_score}

    if net_structure == 'mi_net_simple':
        
        net = dense_dict['layer4'](input_layer['features'])
        net = tf.sigmoid(net)
        raw_score = net
        print('dense1:',net)
        
        net = log_sum_exp_pooling(logits=net,r=r_LSE)
        print('bag_score:', net)
        print('network construction complete')

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network={'net':net, 'loss':loss_func, 'train':train, 'raw': raw_score}

    if net_structure == 'MI-net':
        net = dense_dict['layer1'](input_layer['features'])
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense1:',net)
        net = dense_dict['layer2'](net)
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense2:',net)
        net = dense_dict['layer3'](net)
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense3:',net)
        net = tf.sigmoid(net)*2-1
        net = tf.reshape(mil_max_pooling(instances=net, r=r_LSE), [1,dense_dict['layer3'].units])
        print('MIL pooling layer:',net)

        net = tf.reshape(dense_dict['layer4'](net),())
        print('dense4:',net)
        raw_score = net
        net = tf.sigmoid(net)
        print('instance_score:', net)

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network = {'net':net, 'loss':loss_func, 'train':train, 'raw':raw_score}

    if net_structure == 'MI-net_simple':
        
        net = dense_dict['layer3'](input_layer['features'])
        print('dense3:',net)
        net = tf.sigmoid(net)*2-1
        net = tf.reshape(mil_max_pooling(instances=net, r=r_LSE), [1,dense_dict['layer3'].units])
        print('MIL pooling layer:',net)

        net = tf.reshape(dense_dict['layer4'](net),())
        print('dense4:',net)
        net = tf.sigmoid(net)
        print('instance_score:', net)

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network = {'net':net, 'loss':loss_func, 'train':train}

    if net_structure == 'MI_net_DS':
        net = dense_dict['layer1'](input_layer['features'])
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense1:',net)

        mil_pool_1 = tf.reshape(mil_max_pooling(instances=net, r=r_LSE),[1,dense_dict['layer1'].units])
        print('MIL pooling layer1:', mil_pool_1)
        net = dense_dict['layer2'](net)
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense2:',net)

        mil_pool_2 = tf.reshape(mil_max_pooling(instances=net, r=r_LSE),[1,dense_dict['layer2'].units])
        print('MIL pooling layer2', mil_pool_2)
        net = dense_dict['layer3'](net)
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense3:',net)

        net = tf.reshape(mil_max_pooling(instances=net, r=r_LSE),[1,dense_dict['layer3'].units])
        print('MIL pooling layer3:',net)

        net = dense_dict['layer4'](net)
        print('dense4:',net)

        DS_1 = dense_dict['layer5'](mil_pool_1)
        DS_1 = tf.nn.dropout(DS_1,keep_prob=1-dropout_rate)
        print('Deep supervision 1',DS_1)
        DS_2 = dense_dict['layer6'](mil_pool_2)
        DS_2 = tf.nn.dropout(DS_2,keep_prob=1-dropout_rate)
        print('Deep supervision 2',DS_2)

        net = tf.divide(tf.add_n([net,DS_1,DS_2]),3)
                           
        net = tf.sigmoid(net)
        net = tf.reshape(net,())
        print('instance_score:', net)

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network={'net':net, 'loss':loss_func, 'train':train}

    if net_structure == 'MI_net_RC':
        net = dense_dict['layer1'](input_layer['features'])
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense1:',net)

        mil_pool_1 = mil_max_pooling(instances=net, r=r_LSE)
        print('MIL pooling layer1:', mil_pool_1)
        net = dense_dict['layer2'](net)
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense2:',net)

        mil_pool_2 = mil_max_pooling(instances=net, r=r_LSE)
        print('MIL pooling layer2', mil_pool_2)
        net = dense_dict['layer3'](net)
        net = tf.sigmoid(net)*2-1
        net = tf.nn.dropout(net,keep_prob=1-dropout_rate)
        print('dense3:',net)

        net = mil_max_pooling(instances=net, r=r_LSE)
        net = tf.reshape(net + mil_pool_1 + mil_pool_2,[1,dense_dict['layer3'].units])
        print('MIL pooling sum up:',net)

        net = dense_dict['layer4'](net)
        print('dense4:',net)
        net = tf.sigmoid(net)
        net = tf.reshape(net,())
        print('instance_score:', net)

        loss_func = tf.reshape(loss(input_layer['label'],net,weight),())
        print('loss function:',loss_func)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        network={'net':net, 'loss':loss_func, 'train':train}

    
    



    return network

def test(model,data,sess, bag_features, bag_label,file,curve_file):
    size = len(data['test'])
    cnt_p = 0.0
    cnt_tp = 0.0
    cnt_t = 0.0
    for step in range(size):
        result = sess.run(model, feed_dict={bag_features:data['test'][step][0],bag_label:data['test'][step][1]})
        #bag_label:data['label'][step]})
        if data['test'][step][1][0] ==1.0:
            cnt_p = cnt_p + 1
        if abs(result-data['test'][step][1][0]) < 0.5:
            if data['test'][step][1][0] ==1.0:
                cnt_tp = cnt_tp+1
            cnt_t = cnt_t+1
    #compute precision and recall
    if not cnt_p == 0:
        print('positive:',cnt_p,'true pos:',cnt_tp,'pos_rate:',cnt_tp/cnt_p)
        print('negative:',size-cnt_p,'true negative:',(cnt_t-cnt_tp), 'neg_rate:',(cnt_t-cnt_tp)/(size-cnt_p))
        file.write('positive:'+str(cnt_p)+', true pos:'+str(cnt_tp)+', pos_rate:'+str(cnt_tp/cnt_p)+'\n')
        file.write('negative:'+str(size-cnt_p)+', true negative:'+str(cnt_t-cnt_tp)+', neg_rate:'+str((cnt_t-cnt_tp)/(size-cnt_p))+'\n')
        if size-cnt_p-cnt_t+2*cnt_tp>0:
            precision = cnt_tp/(size-cnt_p-cnt_t+2*cnt_tp)
            recall = cnt_tp/cnt_p
            if precision+recall == 0:
                PR = 0
            else:
                PR = 2*precision*recall/(precision+recall)
            print('test set size:',size,'accuracy:',cnt_t/size, 'precision:', precision,'recall:', recall ,'PR value:', PR )
            file.write('test set size:'+str(size)+', accuracy:'+str(cnt_t/size)+', precision:'+str(precision)+', recall:'+str(recall)+', PR value:'+str(PR)+'\n'+'\n')
            curve_file.write(str(precision)+', '+str(recall)+','+str(PR)+',')
        else:
            precision = 0
            recall = cnt_tp/cnt_p
            PR = 0
            print('test set size:',size,'accuracy:',cnt_t/size, 'precision: no instance is predicted to be positive','recall',cnt_tp/cnt_p)
            file.write('test set size:'+str(size)+', accuracy:'+str(cnt_t/size)+', precision: no instance is predicted to be positive'+', recall:'+str(cnt_tp/cnt_p)+'\n'+'\n')
            curve_file.write(str(precision)+', '+str(recall)+','+str(PR)+',')
        
        file.flush()
    else:
        print('positive:',cnt_p,'true pos:',cnt_tp,'rate:',cnt_t/size)
        file.write('positive:'+str(cnt_p)+'true pos:'+str(cnt_tp)+'rate:'+str(cnt_t/size)+'\n')
        file.flush()

        
    return cnt_t/size

def eval_set_loss(model,data,sess, bag_features, bag_label):
    size = len(data['eval'])
    loss = 0
    for step in range(size):
        result = sess.run(model, feed_dict={bag_features:data['eval'][step][0],bag_label:data['eval'][step][1]})
        loss = loss+result

    return loss

def PR_curve(model,data,sess,bag_features, bag_label, file, mode=''):
    if mode == '' or mode =='test':
        size = len(data['test'])
        cnt_p = 0.0
        cnt_tp = 0.0
        cnt_t = 0.0
        for step in range(size):
            result = sess.run(model, feed_dict={bag_features:data['test'][step][0],bag_label:data['test'][step][1]})
            #bag_label:data['label'][step]})
            
            file.write(str(result)+','+ str(data['test'][step][1][0])+ ',')
    if mode =='eval':
        size = len(data['eval'])
        cnt_p = 0.0
        cnt_tp = 0.0
        cnt_t = 0.0
        for step in range(size):
            result = sess.run(model, feed_dict={bag_features:data['eval'][step][0],bag_label:data['eval'][step][1]})
            #bag_label:data['label'][step]})
            
            file.write(str(result)+','+ str(data['eval'][step][1][0])+ ',')

def PR_area(model,data,sess,bag_features, bag_label,mode=''):
    if mode == '' or mode =='test':
        size = len(data['test'])
        prSeq=[]
        for step in range(size):
            result = sess.run(model, feed_dict={bag_features:data['test'][step][0],bag_label:data['test'][step][1]})
            #bag_label:data['label'][step]})
            prSeq.append([result,data['test'][step][1][0]])
    if mode =='eval':
        size = len(data['eval'])
        prSeq=[]
        for step in range(size):
            result = sess.run(model, feed_dict={bag_features:data['eval'][step][0],bag_label:data['eval'][step][1]})
            #bag_label:data['label'][step]})
            prSeq.append([result,data['eval'][step][1][0]])


    
    flag = False
    true_pos=0
    pos = 0
    point_index=0
    precision=[]
    recall=[]
    
    prSeq=np.array(prSeq)
    prSeq=prSeq[np.argsort(prSeq[:,0])[::-1]]
    for i in range(len(prSeq)):
        if prSeq[i,1] == 1:
            pos=pos+1
    for i in range(len(prSeq)):
        if prSeq[i,1] == 1:
            flag = True
            true_pos=true_pos+1
        elif flag:
            flag = False
            point_index=point_index+1
            p=true_pos/(i)
            precision.append(p)
            recall.append(true_pos/pos)
    if len(precision)==0:
        print('wtf??')
        
        area=0
    elif len(precision)<2:
        area = precision[0]+recall[0]
    else:
        area=recall[0]+precision[0]*recall[1]
        for i in range(1,len(precision)-1):
            area=area+precision[i]*(recall[i+1]-recall[i-1])
        area=area+precision[-1]*(1-recall[-2])
    return area
        

    
                          
def instance_inf(model,data,sess,bag_features, bag_label, file):
    size = len(data['test'])
    for step in range(size):
        #result = sess.run(model, feed_dict={bag_features:data['test'][step][0],bag_label:data['test'][step][1]})
        #bag_label:data['label'][step]})
        if data['test'][step][1][0] ==1.0:
            result = sess.run(model, feed_dict={bag_features:data['test'][step][0],bag_label:data['test'][step][1]})
            file.write('instance label: '+str(data['test'][step][2])+'\n')
            file.write('predicted value:'+str(result)+'\n')




