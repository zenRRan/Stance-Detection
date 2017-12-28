# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:51
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Model.py
# @Software: PyCharm Community Edition



from torch.autograd import Variable
import torch.optim as oprim
from Reader import Reader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
from HyperParams import HyperParams
import collections


class Pooling(nn.Module):

    def __init__(self, vocabSize, EmbedSize, labelSize):
        super(Pooling, self).__init__()
        self.embedings = nn.Embedding(vocabSize, EmbedSize)
        self.linear = nn.Linear(EmbedSize, labelSize)

    def forward(self, input):
        out = self.embedings(input)
        pooling = F.max_pool1d(out.view(1, out.size(2), out.size(1)), kernel_size=out.size(1)).view(1, 100)
        out = self.linear(pooling)
        return out