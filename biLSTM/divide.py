# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 下午7:36
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : divide.py
# @Software: PyCharm Community Edition




import random
import jieba

testfile = open("/Users/zhenranran/Desktop/zenRRan.github.com/Stance-Detection/biLSTM/Data/Chinese/NLPCC2016_Stance_Detection_Task_A_Testdata.txt", "r", encoding="utf-8")
trainfile = open('/Users/zhenranran/Desktop/zenRRan.github.com/Stance-Detection/biLSTM/Data/Chinese/NLPCC2016_Stance_Detection_Task_A_Traindata.txt', "r", encoding="utf-8")
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
testNewFile = open("/Users/zhenranran/Desktop/zenRRan.github.com/Stance-Detection/biLSTM/Data/Chinese/test.sd", "w", encoding="utf-8")
trainNewFile = open("/Users/zhenranran/Desktop/zenRRan.github.com/Stance-Detection/biLSTM/Data/Chinese/train.sd", "w", encoding="utf-8")
# dev = text[:400]
# train = text[400:]
# dev = sorted(dev, key=lambda dev:dev[0])
# train = sorted(train, key=lambda train:train[0])
for line in testtext:
    testNewFile.write(" ".join(line)+"\n")
for line in traintext:
    trainNewFile.write(" ".join(line)+"\n")
testNewFile.close()
trainNewFile.close()
