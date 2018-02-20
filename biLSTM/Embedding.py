# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 下午8:22
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Embedding.py
# @Software: PyCharm Community Edition

import torch
import numpy as np
import torch.nn as nn


def load_predtrained_embedding(self, path, words_dic, avg=False, zeros=False):
    padID = words_dic['<pad>']
    embeding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if len(line) <= 1:
                print("load_predtrained_embedding text is wrong!  -> len(line) <= 1")
                break
            else:
                embeding_dim = len(line) - 1
                break
    word_size = len(words_dic)
    print("The word size is ", word_size)
    print("The dim of predtrained embedding is ", embeding_dim, "\n")

    embedding = np.zeros((word_size, embeding_dim))
    in_word_list = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            index = words_dic.get(line[0])
            if index:
                vector = np.array(line[1:], dtype='float32')
                embedding[index] = vector
                in_word_list.append(index)

    if avg is True:
        embedding = np.zeros((word_size, embeding_dim))
        avg_col = np.sum(embedding, axis=0) / len(in_word_list)
        for i in range(word_size):
            if i in in_word_list and words_dic[i] != padID:
                embedding[i] = avg_col

    return torch.from_numpy(embedding).float()














