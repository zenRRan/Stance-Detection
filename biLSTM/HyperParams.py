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
        self.setSentlen = 1
        self.labelSize = 0
        self.EmbedSize = 64
        self.Steps = 40
        self.learningRate = 0.001
        self.dropout = 0.5
        self.wordNum = 0
        self.topicSize = 0
        self.batchSize = 16
        self.wordCutOff = 0
        self.using_pred_emb = True
        self.using_Chinese_data =True
        self.using_English_data = False
        self.topicWordNum = 0
        self.pred_emd_dim = 64

        #biLSTM
        self.hiddenSize = 100
        self.hiddenNum = 1

        #CNN
        self.kernelSizes = [2, 3, 4]
        self.kernelNum = 100

        lg = ''
        if self.using_English_data:
            lg = '英文'
        else:
            lg = '中文'
        self.mainAddress = 'D:/语料/立场检测/'
        self.trainFile = self.mainAddress + lg + "/train.sd"
        if self.using_English_data:
            self.devFile = self.mainAddress + lg + "/dev.sd"
        self.testFile = self.mainAddress + lg + "/test.sd"
        if self.using_English_data:
            self.writeFileName = '../eng_data.txt'
        else:
            self.writeFileName = '../chn_data.txt'

        self.eng_pred_embedding_25_path = 'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.25d.txt'
        self.eng_pred_embedding_50_path = 'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.50d.txt'
        self.eng_pred_embedding_100_path = 'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.100d.txt'
        self.eng_pred_embedding_200_path = 'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.200d.txt'

        self.chn_pred_embedding_64_path = 'D:/Pred_Emdding/news12g_bdbk20g_nov90g_dim64.txt'

        self.trainLen = 0
        self.devLen = 0
        self.testLen = 0

        self.wordAlpha = Alphabet()
        self.labelAlpha = Alphabet()
        self.topicAlpha = Alphabet()

    def args(self):
        args = "----------args----------\n"
        # args += "labelSize      " + str(self.labelSize) + '\n'
        args += "EmbedSize      " + str(self.EmbedSize) + '\n'
        args += "Steps          " + str(self.Steps) + '\n'
        args += "learningRate   " + str(self.learningRate) + '\n'
        args += "dropout        " + str(self.dropout) + '\n'
        args += "batchSize      " + str(self.batchSize) + '\n'
        # args += "wordNum        " + str(self.wordNum) + '\n'
        args += "wordCutOff     " + str(self.wordCutOff) + '\n'
        # args += "topicSize      " + str(self.topicSize) + '\n'
        args += "hiddenSize     " + str(self.hiddenSize) + '\n'
        args += "hiddenNum      " + str(self.hiddenNum) + '\n'
        # args += "trainLen       " + str(self.trainLen) + '\n'
        # args += "devLen         " + str(self.devLen) + '\n'
        # args += "setSentlen     " + str(self.setSentlen) + '\n'
        args += "using_pred_emb " + str(self.using_pred_emb) + '\n'
        if self.using_pred_emb:
            args += "pred_emd_dim   " + str(self.pred_emd_dim) + '\n'
        lg = ''
        if self.using_Chinese_data:
            lg = "Chinese"
        else:
            lg = "English"
        # args += "testLen        " + str(self.testLen) + '\n'
        args += "language       " + lg + '\n\n'


        return args

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


