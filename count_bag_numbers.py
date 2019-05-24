from lib.data_loader import *
import gc

def count_number_of_bags():
    fname = 'C:/Users/lyz62/Downloads/Multiple Instance Data/KO/Yaozhong/KO.data'
    fname1= 'C:/Users/lyz62/Downloads/Multiple Instance Data/omission/Yaozhong//omission.data'
    fname2= 'C:/Users/lyz62/Downloads/Multiple Instance Data/RNAi/Yaozhong//RNAi.data'
    with open(fname) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line[0:10].split(',')[0])
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('KO:')
    print(len(input_instance_index))
    print(max_)
    print(min_)
    with open(fname1) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line[0:10].split(',')[0])
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('omission:')
    print(len(input_instance_index))
    print(max_)
    print(min_)
    with open(fname2) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line[0:10].split(',')[0])
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('RNAi')
    print(len(input_instance_index))
    print(max_)
    print(min_)

def test_set_range():
    tfname1='C:/Users/lyz62/Downloads/Multiple Instance Data/KO/Yaozhong/KO.test'
    tfname2='C:/Users/lyz62/Downloads/Multiple Instance Data/omission/Yaozhong/omission.test'
    tfname3='C:/Users/lyz62/Downloads/Multiple Instance Data/RNAi/Yaozhong/RNAi.test'

    with open(tfname1) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line)
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('KO:')
    print(len(input_instance_index))
    print(max_-min_+1)
    print(max_)
    print(min_)

    with open(tfname2) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line)
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('omission')
    print(len(input_instance_index))
    print(max_-min_+1)
    print(max_)
    print(min_)

    with open(tfname3) as f:
        input_instance_index = {}
        line = f.readline()
        max_ = 0
        min_ = 100000
        while not line == '':
            bag_name = int(line)
            if not bag_name in input_instance_index:
                input_instance_index[bag_name]=1
            if bag_name > max_:
                max_ = bag_name
            if bag_name < min_:
                min_ = bag_name
            line = f.readline()
    print('RNAi')
    print(len(input_instance_index))
    print(max_-min_+1)
    print(max_)
    print(min_)

def count_load_numbers(name):
    data=data_loader(name)
    cnt=0
    for bags in data['train']:
        if bags[1]==[1]:
            cnt=cnt+1
    print('number of training instances:')
    print(len(data['train']))
    print('number of positive training instances:')
    print(cnt)
    
    cnt1=0
    for bags in data['eval']:
        if bags[1]==[1]:
            cnt1=cnt1+1
    print('number of tuning instances:')
    print(len(data['eval']))
    print('number of positive tuning instances:')
    print(cnt1)

    cnt1=cnt+cnt1
    print('number of instances in training and tuning set:')
    print(len(data['train'])+len(data['eval']))
    print('number of positive instances in training and tuning set:')
    print(cnt1)

    cnt=0
    for bags in data['test']:
        if bags[1]==[1]:
            cnt=cnt+1
    print('number of test instances:')
    print(len(data['test']))
    print('number of positive test instances:')
    print(cnt)

    cnt=cnt+cnt1
    print('# instances')
    print(len(data['train'])+len(data['eval'])+len(data['test']))
    print('# positive instances')
    print(cnt)

    #del data
    #gc.collect()

#count_load_numbers('KO')
#count_load_numbers('omission')
count_load_numbers('RNAi')
