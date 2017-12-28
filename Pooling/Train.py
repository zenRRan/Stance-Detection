# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 下午12:54
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Train.py
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
import numpy as np
from Model import Pooling




class Labeler:

    def __init__(self):
        self.HyperParams = HyperParams()
        self.word_stat_dic = {}
        self.label_stat_dic = {}


    def createAlphabet(self, text):
        print("creating Alphabet......")
        for line in text:
            for word in line:
                if word not in self.word_stat_dic:
                    self.word_stat_dic[word] = 1
                else:
                    self.word_stat_dic[word] += 1
        self.HyperParams.wordAlpha.initial(self.word_stat_dic)
        # print(self.word_stat_dic)
        print("Alphabet has created.")
    def seq2id(self, seq):
        id = []
        for word in seq:
            id.append(self.HyperParams.wordAlpha.from_string(word))
        # for _ in range(maxLen-len(seq)):
        #     id.append(0)
        return id
    def label2seq(self, l):
        G = []
        for line in l:
            if line == "0":
                G.append([1, 0, 0, 0, 0])
            elif line == "1":
                G.append([0, 1, 0, 0, 0])
            elif line == "1":
                G.append([0, 0, 1, 0, 0])
            elif line == "1":
                G.append([0, 0, 0, 1, 0])
            else:
                G.append([0, 0, 0, 0, 1])
        return G

Labeler = Labeler()
ReaderDev = Reader(Labeler.HyperParams.mainAddress+'data/raw.clean.dev', False)
Reader = Reader(Labeler.HyperParams.mainAddress+'data/raw.clean.train', False)
textTrain, labelTrain, labelType, maxLen = Reader.getData()
sents = Reader.getWholeText()


textDev, _, _, _ = ReaderDev.getData()
sentsDev = ReaderDev.getWholeText()

print("Train size "+str(len(textTrain)))

print("Dev size "+str(len(sentsDev)))

print("labelTrain size "+str(len(labelTrain)))
print("labelType is ", labelType)
print("maxLen is ", maxLen)

Labeler.createAlphabet(textTrain+textDev)
print("wordAlpha size ", Labeler.HyperParams.wordAlpha.m_size)
print("Embeding size ", Labeler.HyperParams.EmbedSize)
# Labeler.HyperParams.wordAlpha.write(Labeler.HyperParams.mainAddress+"Pooling/wordDic")

vocabSize = Labeler.HyperParams.wordAlpha.m_size
EmbedSize = Labeler.HyperParams.EmbedSize
LearningRate = Labeler.HyperParams.learningRate
Steps = Labeler.HyperParams.Steps




# for line in textTrain:
#     sents.append(Labeler.seq2id(line))

model = Pooling(vocabSize, EmbedSize, len(labelType))

Optimizer = oprim.Adam(model.parameters(), lr=LearningRate)

Loss = []
def accuracy(model, sents):
    C = 0
    for sent in sents:
        Optimizer.zero_grad()
        feature = Variable(torch.LongTensor([Labeler.seq2id(sent[:-2])]))
        label =  Variable(torch.LongTensor([int(sent[-1])]))
        Y = model(feature)
        C += torch.max(Y, 1)[1].data == int(sent[-1])
    return C.numpy()[0]/len(sents)*100, C.numpy()[0], len(sents)

for step in range(Steps):
    totalLoss = torch.Tensor([0])
    cnt = 0
    trainCorrect = 0
    for sent in sents:
        Optimizer.zero_grad()
        feature = Variable(torch.LongTensor([Labeler.seq2id(sent[:-2])]))
        label =  Variable(torch.LongTensor([int(sent[-1])]))
        Y = model(feature)
        Loss = F.cross_entropy(Y, label)
        Loss.backward()
        Optimizer.step()

        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)

        totalLoss += Loss.data
        trainCorrect += torch.max(Y, 1)[1].view(label.size()).data == int(sent[-1])
    TrainAcc = trainCorrect.numpy()[0]/len(sents) * 100
    DevAcc, DevCorrect, DevNum = accuracy(model, sentsDev)
    print("Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Dev  acc: {:.4f}%{}/{}".format(step,
                                                                                    totalLoss.numpy()[0],
                                                                                          TrainAcc,
                                                                                          trainCorrect.numpy()[0],
                                                                                          len(sents),
                                                                                          DevAcc[0],
                                                                                          DevCorrect[0],
                                                                                          int(DevNum)))






























# def sub(pred, gold):
#     list = []
#     for i in range(len(pred)):
#          list.append(pred[i] - gold[i])
#          print(i, " ", pred[i] - gold[i])
#     return list








# for i in range(len(sents)):
#     P = []
#     sent = sents[1]
#     input = Variable(torch.LongTensor(sent))
#     embed = embeding(input)
#     pooling = F.max_pool1d(embed.view(1, Labeler.HyperParams.EmbedSize, len(sent)), kernel_size=len(sent)).view(100, 1)
#     W = Variable(torch.randn(len(labelType), Labeler.HyperParams.EmbedSize))
#     Y = torch.mm(W, pooling).view((1, len(labelType)))
#     P = F.softmax(Y)
#     # print(P)
#     # print(G[i])
#     print("P len ", P)
#     print("pooling ", pooling)
#     a = sub(P, G[i])
#     print(sub(P, G[i]))
#     W = W - torch.mul(0.01, torch.mm(sub(P, G[i]), pooling))

































