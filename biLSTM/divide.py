# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 下午7:36
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : divide.py
# @Software: PyCharm Community Edition




import random

file = open("/Users/zhenranran/Desktop/raw.txt", "r", encoding="utf-8")
text = []
for line in file.readlines():
    line = line.strip().split(" ")
    text.append(line[1:])
random.shuffle(text)
devFile = open("/Users/zhenranran/Desktop/dev2.sd", "w", encoding="utf-8")
trainFile = open("/Users/zhenranran/Desktop/train2.sd", "w", encoding="utf-8")
dev = text[:400]
train = text[400:]
dev = sorted(dev, key=lambda dev:dev[0])
train = sorted(train, key=lambda train:train[0])
for line in dev:
    devFile.write(" ".join(line)+"\n")
for line in train:
    trainFile.write(" ".join(line)+"\n")
devFile.close()
trainFile.close()
file.close()
