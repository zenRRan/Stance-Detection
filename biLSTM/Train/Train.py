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
import random
from Evaluate import Eval
from Common import unk_key
from Common import padding_key
from Common import English_topics
from Common import Chinese_topics
import collections

class Labeler:

    def __init__(self):
        self.HyperParams = HyperParams()
        self.word_stat_dic = collections.OrderedDict()
        self.label_stat_dic = collections.OrderedDict()
        self.topic_stat_dic = collections.OrderedDict()

        if self.HyperParams.using_English_data:
            self.topics = English_topics
        else:
            self.topics = Chinese_topics

        self.padID = 0
        self.unkID = 0
        self.i = 0

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

        for line in self.topics:
            line = line.strip().split()
            for word in line:
                if word not in self.topic_stat_dic:
                    self.topic_stat_dic[word] = 1
                else:
                    self.topic_stat_dic[word] += 1

        self.HyperParams.wordAlpha.from_string(unk_key)
        self.HyperParams.wordAlpha.from_string(padding_key)

        self.word_stat_dic[unk_key] = self.HyperParams.wordCutOff + 1
        self.word_stat_dic[padding_key] = self.HyperParams.wordCutOff + 1

        self.HyperParams.wordAlpha.initial(self.word_stat_dic, self.HyperParams.wordCutOff)
        self.HyperParams.labelAlpha.initial(self.label_stat_dic)
        self.HyperParams.topicAlpha.initial(self.topic_stat_dic)

        self.padID = self.HyperParams.wordAlpha.from_string(padding_key)
        self.unkID = self.HyperParams.wordAlpha.from_string(unk_key)

        self.HyperParams.wordNum = self.HyperParams.wordAlpha.m_size
        self.HyperParams.labelSize = self.HyperParams.labelAlpha.m_size
        self.HyperParams.topicWordNum = self.HyperParams.topicAlpha.m_size
        # print(self.HyperParams.labelAlpha.id2string)
        # print(self.label_stat_dic)
        print("Created over")

        # print("wordNum: ", self.HyperParams.wordNum)
        # print("label: ", self.HyperParams.labelSize)

    def seq2id(self, seqs):
        idList = []
        maxLen = 0

        for seq in seqs:
            maxLen = max(maxLen, len(seq))
        for seq in seqs:
            id = []
            for word in seq:
                degit = self.HyperParams.wordAlpha.from_string(word)
                if degit >= 0:
                    id.append(degit)
                else:
                    id.append(self.unkID)
            for _ in range(maxLen-len(seq)):
                id.append(self.padID)
            idList.append(id)
        return idList

    def label2id(self, labels):
        idList = []
        for label in labels:
            id = self.HyperParams.labelAlpha.from_string(label)
            if id != -1:
                idList.append(id)
            else:
                print("Wrong: label2id id = -1!")
                return []

        return idList

    def processingRawStanceData(self, textList):
        topics = []
        texts = []
        labels = []
        if self.HyperParams.using_English_data:
            for line in textList:
                if line[0] == self.topics[0]:
                    topics.append(line[0])
                    texts.append(line[1:-1])
                    labels.append(line[-1])
                elif " ".join(line[:2]) == self.topics[1]:
                    topics.append(line[:2])
                    texts.append(line[2:-1])
                    labels.append(line[-1])
                elif " ".join(line[:2]) == self.topics[2]:
                    topics.append(line[:2])
                    texts.append(line[2:-1])
                    labels.append(line[-1])
                elif " ".join(line[:3]) == self.topics[3]:
                    topics.append(line[:3])
                    texts.append(line[3:-1])
                    labels.append(line[-1])
                elif " ".join(line[:6]) == self.topics[4]:
                    topics.append(line[:6])
                    texts.append(line[6:-1])
                    labels.append(line[-1])
                else:
                    print("wrong: def -> processingRawStanceData"+'\n'+' '.join(line))
                    return -1
        else:
            for line in textList:
                self.i += 1
                # if not(line[-1] == 'favor' or line[-1] == 'against' or line[-1] == 'none'):
                    # print(self.i, line[-1])

                if ' '.join(line[:2]) == self.topics[0]:
                    topics.append(line[:2])
                    texts.append(line[2:-1])
                    labels.append(line[-1])
                elif line[0] == self.topics[1]:
                    topics.append(line[0])
                    texts.append(line[1:-1])
                    labels.append(line[-1])
                elif " ".join(line[:6]) == self.topics[2]:
                    topics.append(line[:6])
                    texts.append(line[6:-1])
                    labels.append(line[-1])
                elif ' '.join(line[:2]) == self.topics[3]:
                    topics.append(line[:2])
                    texts.append(line[2:-1])
                    labels.append(line[-1])
                elif " ".join(line[:3]) == self.topics[4]:
                    topics.append(line[:3])
                    texts.append(line[3:-1])
                    labels.append(line[-1])
                else:
                    print("wrong: def -> processingRawStanceData"+'\n'+' '.join(line))
                    return -1

        return topics, texts, labels

    def cutSentFromText(self, text):
        newText = []
        for line in text:
            newText.append(line[:self.HyperParams.setSentlen])
        return newText

    def train(self, trainFile, devFile=None, testFile=None):

        readerTrain = Reader.reader(trainFile)
        readerTest = Reader.reader(testFile)

        sentsTrain = readerTrain.getWholeText()
        sentsTest = readerTest.getWholeText()

        # sentsTrain = self.cutSentFromText(sentsTrain)
        # sentsTest = self.cutSentFromText(sentsTest)

        self.HyperParams.trainLen = len(sentsTrain)
        self.HyperParams.testLen = len(sentsTest)

        if self.HyperParams.using_English_data:
            readerDev = Reader.reader(devFile)
            sentsDev = readerDev.getWholeText()
            sentsDev = self.cutSentFromText(sentsDev)
            self.HyperParams.devLen = len(sentsDev)

        if self.HyperParams.using_English_data:
            self.createAlphabet(sentsTrain+sentsDev)
        else:
            self.createAlphabet(sentsTrain)
        self.HyperParams.topicSize = len(self.topics)

        args = self.HyperParams.args()

        print(args)

        LearningRate = self.HyperParams.learningRate
        Steps = self.HyperParams.Steps

        model = biLSTM.Model(self.HyperParams)
        # print(model)
        Optimizer = oprim.Adam(model.parameters(), lr=LearningRate)

        def accuracy(model, sents):
            pred_right_num_idx = 0
            pred_num_idx = 1
            gold_num_idx = 2

            evalList = [[0, 0, 0] for _ in range(self.HyperParams.labelSize)]

            topic, text, label = self.processingRawStanceData(sents)
            topic = self.seq2id(topic)
            text = self.seq2id(text)
            label = self.label2id(label)

            topic = Variable(torch.LongTensor(topic))
            text = Variable(torch.LongTensor(text))
            label = Variable(torch.LongTensor(label))

            Y = model(topic, text)
            C = (torch.max(Y, 1)[1].view(label.size()).data == label.data).sum()

            pred_list = torch.max(Y, 1)[1].view(label.size()).data.tolist()
            label_list = label.data.tolist()

            for i in range(len(evalList)):
                for j in range(len(label_list)):
                    if label_list[j] == i:
                        evalList[i][gold_num_idx] += 1
                        if label_list[j] == pred_list[j]:
                            evalList[i][pred_right_num_idx] += 1
                    if pred_list[j] == i:
                        evalList[i][pred_num_idx] += 1
            P_R_F1_list = [Eval(pred_right_num=evalList[i][pred_right_num_idx],
                                pred_num=evalList[i][pred_num_idx],
                                gold_num=evalList[i][gold_num_idx]).P_R_F1
                           for i in range(len(evalList))]

            return float(C)/len(sents)*100, C, len(sents), P_R_F1_list
        def getTextBatchList(text, batch):
            textBatchlist = []
            textBatchNum = len(text) // batch
            if len(text) % batch != 0:
                textBatchNum += 1
            if textBatchNum - 1 < 0:
                print("wrong: func getTextBatchList's text's length is 0!!!")
                return []
            end = 0
            for i in range(textBatchNum-1):
                begin = end
                end += batch
                textBatchlist.append(text[begin:end])
            textBatchlist.append(text[end:len(text)])
            return textBatchlist

        file = open(self.HyperParams.writeFileName, 'a+')
        file.write(args)
        file.close()

        sentsTrain = sentsTrain
        if self.HyperParams.using_English_data:
            sentsDev = sentsDev
        sentsTest = sentsTest
        batchSize = self.HyperParams.batchSize

        best_F1 = 0
        best_acc = 0
        for step in range(Steps):
            file = open(self.HyperParams.writeFileName, 'a+')
            totalLoss = torch.Tensor([0])
            cnt = 0
            trainCorrect = 0
            # random.shuffle(sentsTrain)
            textBatchList = getTextBatchList(sentsTrain, batchSize)


            for batch in textBatchList:
                model.train()
                Optimizer.zero_grad()

                topic, text, label = self.processingRawStanceData(batch)
                topic = self.seq2id(topic)
                text = self.seq2id(text)
                label = self.label2id(label)

                topic = Variable(torch.LongTensor(topic))
                text = Variable(torch.LongTensor(text))
                label = Variable(torch.LongTensor(label))

                Y = model(topic, text)

                Loss = F.cross_entropy(Y, label)
                Loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                Optimizer.step()

                cnt += 1
                if cnt % 500 == 0:
                    print(cnt)

                totalLoss += Loss.data
                trainCorrect += (torch.max(Y, 1)[1].view(label.size()).data == label.data).sum()

            totalLoss /= len(sentsTrain)
            TrainAcc = float(trainCorrect)/len(sentsTrain) * 100

            FAVOR_index = self.HyperParams.labelAlpha.string2id["favor"]
            AGAINST_index = self.HyperParams.labelAlpha.string2id["against"]
            if self.HyperParams.using_English_data:
                DevAcc, DevCorrect, DevNum, P_R_F1_dev_list =  accuracy(model, sentsDev)
            TestAcc, TestCorrect, TestNum, P_R_F1_test_list = accuracy(model, sentsTest)
            if self.HyperParams.using_English_data:
                dev_mean_F1 = (P_R_F1_dev_list[FAVOR_index][2] + P_R_F1_dev_list[AGAINST_index][2]) / 2
            test_mean_F1 = (P_R_F1_test_list[FAVOR_index][2] + P_R_F1_test_list[AGAINST_index][2]) / 2

            if best_F1 < test_mean_F1:
                best_F1 = test_mean_F1
                best_acc = TestAcc
            if self.HyperParams.using_Chinese_data:
                output = "Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Test  acc: {:.4f}%{}/{}  F1={:.4f}".format(step,
                                                                                                                           totalLoss.numpy()[0],
                                                                                                                           TrainAcc,
                                                                                                                           trainCorrect,
                                                                                                                           len(sentsTrain),
                                                                                                                           TestAcc,
                                                                                                                           TestCorrect,
                                                                                                                           int(TestNum),
                                                                                                                           test_mean_F1)
            else:
                output = "Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Dev  acc: {:.4f}%{}/{}     Test  acc: {:.4f}%{}/{}  F1={:.4f}".format(
                    step,
                    totalLoss.numpy()[0],
                    TrainAcc,
                    trainCorrect,
                    len(sentsTrain),
                    DevAcc,
                    DevCorrect,
                    int(DevNum),
                    TestAcc,
                    TestCorrect,
                    int(TestNum),
                    test_mean_F1)

            print(output)
            file.write(output+"\n")
            file.close()

        file = open(self.HyperParams.writeFileName, 'a+')
        output = 'Total: best F1 = ' + str(best_F1) + ' acc = ' + str(best_acc)
        print(output)
        file.write(output + "\n")
        file.close()

# def printList(list):
#     for line in list:
#         print(" ".join(line))
l = Labeler()
if l.HyperParams.using_English_data:
    l.train(l.HyperParams.trainFile, l.HyperParams.devFile, l.HyperParams.testFile)
else:
    l.train(l.HyperParams.trainFile, testFile=l.HyperParams.testFile)