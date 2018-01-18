# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:51
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Evaluate.py
# @Software: PyCharm Community Edition

import torch
import torch.nn as nn
from torch.autograd import Variable


class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        self.precision = self.correct_num / self.predict_num
        self.recall = self.correct_num / self.gold_num
        self.fscore = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        print("precision: ", self.precision, ", recall: ", self.recall, ", fscore: ", self.fscore)

