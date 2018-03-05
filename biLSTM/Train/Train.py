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
from Model import biGRU
from Model import CNN
import random
from Evaluate import Eval
from Common import unk_key
from Common import padding_key
from Common import English_topics
from Common import Chinese_topics
import random
random.seed(23)
torch.manual_seed(23)

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

        self.HyperParams.word_alpha.from_string(unk_key)
        self.HyperParams.word_alpha.from_string(padding_key)

        self.word_stat_dic[unk_key] = self.HyperParams.word_cut_off + 1
        self.word_stat_dic[padding_key] = self.HyperParams.word_cut_off + 1

        self.HyperParams.word_alpha.initial(self.word_stat_dic, self.HyperParams.word_cut_off)
        self.HyperParams.label_alpha.initial(self.label_stat_dic)
        self.HyperParams.topic_alpha.initial(self.topic_stat_dic)

        self.padID = self.HyperParams.word_alpha.from_string(padding_key)
        self.unkID = self.HyperParams.word_alpha.from_string(unk_key)

        self.HyperParams.word_num = self.HyperParams.word_alpha.m_size
        self.HyperParams.label_size = self.HyperParams.label_alpha.m_size
        self.HyperParams.topic_word_num = self.HyperParams.topic_alpha.m_size
        # print(self.HyperParams.label_alpha.id2string)
        # print(self.label_stat_dic)
        print("Created over")

        # print("wordNum: ", self.HyperParams.wordNum)
        # print("label: ", self.HyperParams.labelSize)

    def seq2id(self, seqs):
        id_list = []
        max_len = 0

        for seq in seqs:
            max_len = max(max_len, len(seq))
        for seq in seqs:
            id = []
            for word in seq:
                degit = self.HyperParams.word_alpha.from_string(word)
                if degit >= 0:
                    id.append(degit)
                else:
                    id.append(self.unkID)
            for _ in range(max_len-len(seq)):
                id.append(self.padID)
            id_list.append(id)
        return id_list

    def label2id(self, labels):
        id_list = []
        for label in labels:
            id = self.HyperParams.label_alpha.from_string(label)
            if id != -1:
                id_list.append(id)
            else:
                print("Wrong: label2id id = -1!")
                return []

        return id_list

    def processingRawStanceData(self, text_list):
        topics = []
        texts = []
        labels = []
        if self.HyperParams.using_English_data:
            for line in text_list:
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
            for line in text_list:
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
        new_text = []
        for line in text:
            new_text.append(line[:self.HyperParams.set_sent_len])
        return new_text

    def train(self, train_file, dev_file=None, test_file=None):

        reader_train = Reader.reader(train_file, language='chn')
        reader_test = Reader.reader(test_file, language='chn')

        sents_train = reader_train.getWholeText()
        sents_test = reader_test.getWholeText()

        # sentsTrain = self.cutSentFromText(sentsTrain)
        # sentsTest = self.cutSentFromText(sentsTest)

        self.HyperParams.train_len = len(sents_train)
        self.HyperParams.test_len = len(sents_test)

        if self.HyperParams.using_English_data:
            reader_dev = Reader.reader(dev_file, language='eng')
            sents_dev = reader_dev.getWholeText()
            sents_dev = self.cutSentFromText(sents_dev)
            self.HyperParams.dev_den = len(sents_dev)

        if self.HyperParams.using_English_data:
            self.createAlphabet(sents_train+sents_dev)
        else:
            self.createAlphabet(sents_train)
        self.HyperParams.topic_size = len(self.topics)

        args = self.HyperParams.args()

        print(args)

        lr = self.HyperParams.lr
        Steps = self.HyperParams.Steps

        model = None
        if self.HyperParams.biLSTM:
            print("using biLSTM...")
            model = biLSTM.Model(self.HyperParams)
        if self.HyperParams.biGRU:
            print("using biGRU...")
            model = biGRU.Model(self.HyperParams)
        if self.HyperParams.CNN:
            print("using CNN...")
            model = CNN.Model(self.HyperParams)
        if model == None:
            print("please select a model!")
            return

        # print(model)
        # param = [i for i in model.parameters() if i.requires_grad]
        # param = [i for i in model.parameters() if i.sparse]
        # sparseParam = [i for i in model.parameters() if not i.sparse]
        # Optimizer = oprim.Adam(param, lr=LearningRate)
        # SparseOprimizer = oprim.SparseAdam(sparseParam)
        # model.
        # Optimizer = oprim.Adam(model.parameters(), lr=LearningRate, weight_decay=self.HyperParams.decay)

        Optimizer = None
        if self.HyperParams.Adam:
            Optimizer = oprim.Adam(model.parameters(), lr=lr)
        if self.HyperParams.SGD:
            Optimizer = oprim.SGD(model.parameters(), lr=lr)
        if Optimizer == None:
            print("please select a model!")
            return



        def accuracy(model, sents):
            pred_right_num_idx = 0
            pred_num_idx = 1
            gold_num_idx = 2

            evalList = [[0, 0, 0] for _ in range(self.HyperParams.label_size)]

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

        file = open(self.HyperParams.write_file_name, 'a+')
        file.write(args)
        file.close()

        sents_train = sents_train
        if self.HyperParams.using_English_data:
            sents_dev = sents_dev
        sents_test = sents_test
        batch_size = self.HyperParams.batch_size

        best_F1 = 0
        best_acc = 0
        for step in range(Steps):
            file = open(self.HyperParams.write_file_name, 'a+')
            total_loss = torch.Tensor([0])
            cnt = 0
            train_correct = 0
            random.shuffle(sents_train)
            text_batch_list = getTextBatchList(sents_train, batch_size)


            for batch in text_batch_list:
                model.train()
                Optimizer.zero_grad()
                # SparseOprimizer.zero_grad()

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
                if self.HyperParams.clip_grad:
                    torch.nn.utils.clip_grad_norm(model.parameters(), 10)
                Optimizer.step()

                cnt += 1
                if cnt % 500 == 0:
                    print(cnt)

                total_loss += Loss.data
                train_correct += (torch.max(Y, 1)[1].view(label.size()).data == label.data).sum()
            if self.HyperParams.lr_decay:
                adjust_learning_rate(Optimizer, self.HyperParams.lr / (1 + (step*3.01 + 1) * self.HyperParams.decay))

            total_loss /= len(sents_train)
            train_acc = float(train_correct)/len(sents_train) * 100

            FAVOR_index = self.HyperParams.label_alpha.string2id["favor"]
            AGAINST_index = self.HyperParams.label_alpha.string2id["against"]
            if self.HyperParams.using_English_data:
                dev_acc, dev_correct, dev_num, P_R_F1_dev_list =  accuracy(model, sents_dev)
            test_acc, test_correct, test_num, P_R_F1_test_list = accuracy(model, sents_test)
            if self.HyperParams.using_English_data:
                dev_mean_F1 = (P_R_F1_dev_list[FAVOR_index][2] + P_R_F1_dev_list[AGAINST_index][2]) / 2
            test_mean_F1 = (P_R_F1_test_list[FAVOR_index][2] + P_R_F1_test_list[AGAINST_index][2]) / 2

            if best_F1 < test_mean_F1:
                best_F1 = test_mean_F1
                best_acc = test_acc
            if self.HyperParams.using_Chinese_data:
                output = "Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Test  acc: {:.4f}%{}/{}  F1={:.4f}".format(step,
                                                                                                                           total_loss.numpy()[0],
                                                                                                                           train_acc,
                                                                                                                           train_correct,
                                                                                                                           len(sents_train),
                                                                                                                           test_acc,
                                                                                                                           test_correct,
                                                                                                                           int(test_num),
                                                                                                                           test_mean_F1)
            else:
                output = "Step: {} - loss: {:.6f}  Train  acc: {:.4f}%{}/{}     Dev  acc: {:.4f}%{}/{}     Test  acc: {:.4f}%{}/{}  F1={:.4f}".format(
                    step,
                    total_loss.numpy()[0],
                    train_acc,
                    train_correct,
                    len(sents_train),
                    dev_acc,
                    dev_correct,
                    int(dev_num),
                    test_acc,
                    test_correct,
                    int(test_num),
                    test_mean_F1)

            print(output)
            file.write(output+"\n")
            file.close()

        file = open(self.HyperParams.write_file_name, 'a+')
        output = 'Total: best F1 = ' + str(best_F1) + ' acc = ' + str(best_acc)
        print(output)
        file.write(output + "\n")
        file.close()

# def printList(list):
#     for line in list:
#         print(" ".join(line))
def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

l = Labeler()
if l.HyperParams.using_English_data:
    l.train(l.HyperParams.train_file, l.HyperParams.dev_file, l.HyperParams.test_file)
else:
    l.train(l.HyperParams.train_file, test_file=l.HyperParams.test_file)