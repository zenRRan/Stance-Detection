# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/1/18 上午9:59
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : biLSTM.py
# @Software: PyCharm Community Edition

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.embedingTopic = nn.Embedding(args.topicSize, args.EmbedSize)
        self.embedingText = nn.Embedding(args.wordNum, args.EmbedSize)

        self.biLSTM = nn.LSTM(
            args.EmbedSize,
            args.hiddenSize,
            num_layers=args.hiddenNum,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(args.hiddenSize * 4, args.hiddenSize // 2)
        self.linear2 = nn.Linear(args.hiddenSize // 2, args.labelSize)

    def forward(self, topic, text):
        topic = self.embedingTopic(topic)
        text = self.embedingText(text)

        topic, _ = self.biLSTM(topic)   #[1, 1, 200]
        text,  _ = self.biLSTM(text)    #[1, 17, 200]

        # print("----biLSTM----")
        # print("topic.size ", topic.size())
        # print("text.size ", text.size())

        topic = torch.transpose(topic, 1, 2)
        text = torch.transpose(text, 1, 2)

        topic = F.tanh(topic)
        text = F.tanh(text)

        topic = F.max_pool1d(topic, topic.size(2))  #[1, 200, 1]
        text = F.max_pool1d(text, text.size(2))     #[1, 200, 1]

        # print("----maxpooling----")
        # print("topic.size ", topic.size())
        # print("text.size ", text.size())
        topic_text = torch.cat([topic, text], 1)    #[1, 400, 1]

        # print("----cat----")
        # print("topic_text.size ", topic_text.size())

        topic_text = topic_text.squeeze(2)          #[1, 400]

        # print("----squeeze(2)----")
        # print("topic_text.size ", topic_text.size())

        output = self.linear1(topic_text)

        # print("----linear1----")
        # print("output.size ", output.size())
        output = F.tanh(output)
        output = self.linear2(output)
        # print("----linear2----")
        # print("output.size ", output.size())

        return output






