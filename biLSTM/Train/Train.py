# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 下午12:54
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Train.py
# @Software: PyCharm Community Edition

import torch
import torch.nn.functional as F
import torch.optim as oprim
from torch.autograd import Variable

import Reader
from HyperParams import HyperParams
from Model import biLSTM

class Labeler:

    def __init__(self):
        self.HyperParams = HyperParams()
        self.word_stat_dic = {}
        self.label_stat_dic = {}
        self.topics = ['atheism', 'feminist movement', 'hillary clinton',
                  'legalization of abortion', 'climate change is a real concern']

    def createAlphabet(self, text):
        print("Creating Alphabet......")
        for line in text:
            for word in line[:-2]:
                if word not in self.word_stat_dic:
                    self.word_stat_dic[word] = 1
                else:
                    self.word_stat_dic[word] += 1

            if line[-1] not in self.label_stat_dic:
                self.label_stat_dic[line[-1]] = 1
            else:
                self.label_stat_dic[line[-1]] += 1



        self.HyperParams.wordAlpha.initial(self.word_stat_dic)
        self.HyperParams.labelAlpha.initial(self.label_stat_dic)
        self.HyperParams.wordNum = self.HyperParams.wordAlpha.m_size + 1
        self.HyperParams.labelSize = self.HyperParams.labelAlpha.m_size
        print("Created over")

        # print("wordNum: ", self.HyperParams.wordNum)
        # print("label: ", self.HyperParams.labelSize)

    def seq2id(self, seq):
        id = []
        for word in seq:
            degit = self.HyperParams.wordAlpha.from_string(word)
            if degit >= 0:
                id.append(degit)
            else:
                '''
                    add unknow str
                '''
                id.append(self.HyperParams.wordAlpha.m_size)
        # for _ in range(maxLen-len(seq)):
        #     id.append(0)
        return id
    def label2id(self, label):
        id = self.HyperParams.labelAlpha.from_string(label)
        if id != -1:
            return id
        else:
            print("wrong: def seq2id_label -> ", seq)
            return None

    def processingRawStanceData(self, textList):
        for line in [textList]:
            if line[0] == self.topics[0]:
                return 0, textList[1:-1], textList[-1]
            elif " ".join(line[:2]) == self.topics[1]:
                return 1, textList[2:-1], textList[-1]
            elif " ".join(line[:2]) == self.topics[2]:
                return 2, textList[2:-1], textList[-1]
            elif " ".join(line[:3]) == self.topics[3]:
                return 3, textList[3:-1], textList[-1]
            elif " ".join(line[:6]) == self.topics[4]:
                return 4, textList[6:-1], textList[-1]
            else:
                return -1
    def train(self, trainFile, devFile, testFile):

        readerTrain = Reader.reader(trainFile)
        readerDev = Reader.reader(devFile)
        readerTest = Reader.reader(testFile)

        sentsTrain = readerTrain.getWholeText()
        sentsDev = readerDev.getWholeText()
        sentsTest = readerTest.getWholeText()


        self.HyperParams.trainLen = len(sentsTrain)
        self.HyperParams.devLen = len(sentsDev)
        self.HyperParams.testLen = len(sentsTest)

        self.createAlphabet(sentsTrain+sentsDev)
        self.HyperParams.topicSize = len(self.topics)

        self.HyperParams.printArgs()

        LearningRate = self.HyperParams.learningRate
        Steps = self.HyperParams.Steps

        model = biLSTM.Model(self.HyperParams)
        Optimizer = oprim.Adam(model.parameters(), lr=LearningRate)


        def accuracy(model, sents):
            C = 0
            for sent in sents:
                topic, text, label = self.processingRawStanceData(sent)
                text = self.seq2id(text)
                label = self.label2id(label)
                topic = Variable(torch.LongTensor([[topic]]))
                text = Variable(torch.LongTensor([text]))
                label = Variable(torch.LongTensor([label]))

                Y = model(topic, text)
                C += (torch.max(Y, 1)[1].data == label.data[0]).sum()
            return float(C)/len(sents)*100, C, len(sents)

        for step in range(Steps):
            totalLoss = torch.Tensor([0])
            cnt = 0
            trainCorrect = 0
            # random.shuffle(sentsTrain)
            sentsTrain = sentsTrain[:100]
            for sent in sentsTrain:
                model.train()
                Optimizer.zero_grad()

                topic, text, label = self.processingRawStanceData(sent)
                text = self.seq2id(text)
                label = self.label2id(label)
                topic = Variable(torch.LongTensor([[topic]]))
                text = Variable(torch.LongTensor([text]))
                label = Variable(torch.LongTensor([label]))

                Y = model(topic, text)

                Loss = F.cross_entropy(Y, label)
                Loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                Optimizer.step()

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

                totalLoss += Loss.data
                trainCorrect += (torch.max(Y, 1)[1].view(label.size()).data == label.data[0]).sum()

            TrainAcc = float(trainCorrect)/len(sentsTrain) * 100
            DevAcc, DevCorrect, DevNum = accuracy(model, sentsDev[:10])
            TestAcc, TestCorrect, TestNum = accuracy(model, sentsTest[:10])
            print("Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Dev  acc: {:.4f}%{}/{}     Test  acc: {:.4f}%{}/{}".format(step,
                                                                                                                                    totalLoss.numpy()[0],
                                                                                                                                    TrainAcc,
                                                                                                                                    trainCorrect,
                                                                                                                                    len(sentsTrain),
                                                                                                                                    DevAcc,
                                                                                                                                    DevCorrect,
                                                                                                                                    int(DevNum),
                                                                                                                                    TestAcc,
                                                                                                                                    TestCorrect,
                                                                                                                                    int(TestNum)
                                                                                                                                    ))
l = Labeler()
l.train(l.HyperParams.trainFile, l.HyperParams.devFile, l.HyperParams.testFile)