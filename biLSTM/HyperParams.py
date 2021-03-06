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
        self.max_sent_len = 0
        self.set_sent_len = 1
        self.label_size = 0
        self.embed_size = 64
        self.Steps = 30
        self.lr = 0.001
        self.dropout = 0.5
        self.word_num = 0
        self.topic_size = 0
        self.batch_size = 13
        self.word_cut_off = 0
        self.using_pred_emb = True
        self.using_Chinese_data =True
        self.using_English_data = False
        self.topic_word_num = 0
        self.pred_emd_dim = 64
        self.decay = 1e-8
        self.lr_decay = True
        self.clip_grad = False

        #biLSTM
        self.biLSTM = True
        self.biLSTM_hidden_size = 120
        self.biLSTM_hidden_num = 1

        #biGRU
        self.biGRU = False
        self.biGRU_hidden_size = 100
        self.biGRU_hidden_num = 1

        #CNN
        self.CNN = False
        self.kernelSizes = [2, 3, 4]
        self.kernelNum = 100

        #Optimizer
        self.Adam = True
        self.SGD = False

        lg = ''
        if self.using_English_data:
            lg = '英文'
        else:
            lg = '中文'
        self.main_address = 'D:/语料/立场检测/'
        self.train_file = self.main_address + lg + "/train.sd"
        if self.using_English_data:
            self.dev_file = self.main_address + lg + "/dev.sd"
        self.test_file = self.main_address + lg + "/test.sd"
        if self.using_English_data:
            self.write_file_name = '../eng_data.txt'
        else:
            self.write_file_name = '../test_data.txt'

        eng_pred_embedding_path_list = ['D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.25d.txt',
                                        'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.50d.txt',
                                        'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.100d.txt',
                                        'D:/Pred_Emdding/glove.twitter.27B/glove.twitter.27B.200d.txt']
        self.eng_pred_embedding_path = eng_pred_embedding_path_list[1]
        self.chn_pred_embedding_path = 'D:/Pred_Emdding/news12g_bdbk20g_nov90g_dim64.txt'

        self.save_pred_emd_path = '../Data/Chinese/pred_emd.txt'
        if self.using_English_data:
            self.save_pred_emd_path = self.Eng_pred_embedding_path

        self.train_len = 0
        self.dev_len = 0
        self.test_len = 0

        self.word_alpha = Alphabet()
        self.label_alpha = Alphabet()
        self.topic_alpha = Alphabet()

    def args(self):
        args = "----------args----------\n"
        # args += "label_size=" + str(self.label_size) + '\n'
        args += "embed_size=" + str(self.embed_size) + '\n'
        args += "Steps=" + str(self.Steps) + '\n'
        args += "lr=" + str(self.lr) + '\n'
        args += "dropout=" + str(self.dropout) + '\n'
        args += "batch_size=" + str(self.batch_size) + '\n'
        # args += "word_num        " + str(self.word_num) + '\n'
        args += "word_cut_off=" + str(self.word_cut_off) + '\n'
        # args += "topic_size=" + str(self.topic_size) + '\n'
        if self.biLSTM:
            args += "biLSTM_hidden_size=" + str(self.biLSTM_hidden_size) + '\n'
            args += "biLSTM_hidden_num=" + str(self.biLSTM_hidden_num) + '\n'
        if self.biGRU:
            args += "biGRU_hidden_size=" + str(self.biGRU_hidden_size) + '\n'
            args += "biGRU_hidden_num=" + str(self.biGRU_hidden_num) + '\n'
        if self.SGD:
            args += "Optimizer=" + str(self.SGD) + '\n'
        if self.Adam:
            args += "Optimizer=" + str(self.Adam) + '\n'
        # args += "train_len=" + str(self.train_len) + '\n'
        # args += "dev_len=" + str(self.dev_len) + '\n'
        # args += "set_sent_len=" + str(self.set_sent_len) + '\n'
        args += "lr_decay=" + str(self.lr_decay) + '\n'
        args += "clip_grad=" + str(self.clip_grad) + '\n'
        args += "using_pred_emb=" + str(self.using_pred_emb) + '\n'
        if self.using_pred_emb:
            args += "pred_emd_dim=" + str(self.pred_emd_dim) + '\n'
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


