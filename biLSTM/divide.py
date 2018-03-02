# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 下午7:36
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : divide.py
# @Software: PyCharm Community Edition




import random
import jieba

train_path = 'C:/Users/zenRRan/Desktop/Stance-Detection/biLSTM/Data/Chinese/NLPCC2016_Stance_Detection_Task_A_Traindata.txt'
test_path = 'C:/Users/zenRRan/Desktop/Stance-Detection/biLSTM/Data/Chinese/NLPCC_2016_Stance_Detection_Task_A_gold.txt'
testfile = open(test_path, 'r', encoding='utf-8')
trainfile = open(train_path, 'r', encoding='utf-8')
testtext = []
traintext = []
for line in testfile.readlines():
    line = line.strip().split()
    testtext.append(line[1:])
for line in trainfile.readlines():
    line = line.strip().split()
    traintext.append(line[1:])
testfile.close()
trainfile.close()

# random.shuffle(text)
new_train_path = 'C:/Users/zenRRan/Desktop/Stance-Detection/biLSTM/Data/Chinese/train.sd'
new_test_path = 'C:/Users/zenRRan/Desktop/Stance-Detection/biLSTM/Data/Chinese/test.sd'
testNewFile = open(new_test_path, 'w', encoding='utf-8')
trainNewFile = open(new_train_path, 'w', encoding='utf-8')
# dev = text[:400]
# train = text[400:]
# dev = sorted(dev, key=lambda dev:dev[0])
# train = sorted(train, key=lambda train:train[0])
for line in testtext:
    line = ' '.join(line)
    line = jieba.cut(line)
    line = [item for item in filter(lambda x: x != ' ', line)]
    line = ' '.join(line)
    testNewFile.write(line + '\n')
for line in traintext:
    line = ' '.join(line)
    line = jieba.cut(line)
    line = [item for item in filter(lambda x: x != ' ', line)]
    line = ' '.join(line)
    trainNewFile.write(line + '\n')
testNewFile.close()
trainNewFile.close()

# sent = ['phone', 'se']
# sent = jieba.cut(sent)
# print(sent)
# sent = [item for item in filter(lambda x: x != '', sent)]
# print(sent)
# sent = [item for item in filter(lambda x: x != ' ', sent)]
# print(' '.join(sent))
