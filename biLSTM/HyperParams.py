# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 下午1:46
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : HyperParams.py
# @Software: PyCharm Community Edition

import collections


class HyperParams:
    def __init__(self):
        self.maxSentlen = 0
        self.labelSize = 0
        self.EmbedSize = 64
        self.Steps = 50
        self.learningRate = 0.001
        self.dropout = 0.5
        self.wordNum = 0
        self.topicSize = 0

        #biLSTM
        self.hiddenSize = 128
        self.hiddenNum = 32

        #CNN
        self.kernelSizes = [2, 3, 4]
        self.kernelNum = 100


        self.mainAddress = '/Users/zhenranran/Desktop/zenRRan.github.com/Stance Detection/CNN/Data/'
        self.trainFile = self.mainAddress+"train.sd"
        self.devFile = self.mainAddress + "dev.sd"
        self.testFile = self.mainAddress + "test.sd"

        self.trainLen = 0
        self.devLen = 0
        self.testLen = 0

        self.wordAlpha = Alphabet()
        self.labelAlpha = Alphabet()
    def printArgs(self):
        print("----------args----------")
        print("labelSize    ", self.labelSize)
        print("EmbedSize    ", self.EmbedSize)
        print("Steps        ", self.Steps)
        print("learningRate ", self.learningRate)
        print("dropout      ", self.dropout)
        print("wordNum      ", self.wordNum)
        print("topicSize    ", self.topicSize)
        print("hiddenSize   ", self.hiddenSize)
        print("hiddenNum    ", self.hiddenNum)
        print("trainLen     ", self.trainLen)
        print("devLen       ", self.devLen)
        print("testLen      ", self.testLen)
        print()

class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = collections.OrderedDict()

    def initial(self, stat, cutoff=0):
        for key in stat:
            if stat[key] > cutoff:
                self.from_string(key)
        self.m_b_fixed = True

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.string2id[str] = newid
                self.id2string.append(str)
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def from_id(self, id, definedStr=''):
        if id >= self.m_size or int(id) < 0:
            return definedStr
        else:
            return self.id2string[id]
    def write(self, path):
        fopen = open(path, encoding="utf-8", mode='w')
        for key in self.string2id:
            fopen.write(key+"   "+str(self.string2id[key])+'\n')
        fopen.close()

    def read(self, path):
        fopen = open(path, encoding="utf-8", mode='r')
        for line in fopen:
            info = line.split(" ")
            self.string2id[info[0]] = info[1]
            self.id2string.append(info[0])
        self.m_b_fixed = True
        self.m_size = len(self.string2id)
        fopen.close()

    def clean(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = collections.OrderedDict()

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True


