# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:51
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : CNN.py
# @Software: PyCharm Community Edition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Model(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.dropout = args.dropout
        self.word_num = args.word_num
        self.embed_size = args.embed_size
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.label_size = args.label_size

        self.embeddings = nn.Embedding(self.word_num, self.embed_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_num, (K, self.embed_size), padding=(K // 2, 0)) for K in self.kernel_sizes])
        self.linear = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.label_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input):
        out = self.embeddings(input)
        out = F.tanh(out)
        l = []
        out = out.unsqueeze(1)
        for conv in self.convs:
            l.append(F.tanh(conv(out)).squeeze(3))
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        out = torch.cat(l, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out