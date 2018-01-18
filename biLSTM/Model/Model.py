# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:51
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Model.py
# @Software: PyCharm Community Edition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.embedings = nn.Embedding(args.wordNum, args.EmbedSize)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.kernelNum, (K, args.EmbedSize), padding=(K // 2, 0)) for K in args.kernelSizes])
        self.linear = nn.Linear(len(args.kernelSizes)*args.kernelNum, args.labelSize)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input):
        out = self.embedings(input)
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