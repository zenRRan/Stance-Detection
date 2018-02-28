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
import Embedding

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.EmbedSize = args.EmbedSize
        self.embedingTopic = nn.Embedding(args.topicWordNum, self.EmbedSize)
        self.embedingText = nn.Embedding(args.wordNum, self.EmbedSize)
        if args.using_pred_emb:
            path = ''
            if args.pred_emd_dim == 25:
                self.EmbedSize = 25
                path = args.pred_embedding_25_path
            elif args.pred_emd_dim == 50:
                self.EmbedSize = 50
                path = args.pred_embedding_50_path
            elif args.pred_emd_dim == 100:
                self.EmbedSize = 100
                path = args.pred_embedding_100_path
            elif args.pred_emd_dim == 200:
                self.EmbedSize = 200
                path = args.pred_embedding_200_path
            self.embedingTopic = nn.Embedding(args.topicWordNum, self.EmbedSize)
            self.embedingText = nn.Embedding(args.wordNum, self.EmbedSize)
            load_emb_text = Embedding.load_predtrained_emb_avg(path, args.wordAlpha.string2id, padding=True)
            load_emb_topic = Embedding.load_predtrained_emb_avg(path, args.topicAlpha.string2id, padding=False)
            self.embedingTopic.weight.data.copy_(load_emb_topic)
            self.embedingText.weight.data.copy_(load_emb_text)

        self.biLSTM = nn.LSTM(
            self.EmbedSize,
            args.hiddenSize,
            dropout=args.dropout,
            num_layers=args.hiddenNum,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(args.hiddenSize * 4, args.hiddenSize // 2)
        self.linear2 = nn.Linear(args.hiddenSize // 2, args.labelSize)

    def forward(self, topic, text):
        topic = self.embedingText(topic)
        text = self.embedingText(text)

        topic, _ = self.biLSTM(topic)   #[1, 1, 200]
        text,  _ = self.biLSTM(text)    #[1, 17, 200]


        topic = torch.transpose(topic, 1, 2)
        text = torch.transpose(text, 1, 2)

        topic = F.tanh(topic)
        text = F.tanh(text)

        topic = F.max_pool1d(topic, topic.size(2))  #[1, 200, 1]
        text = F.max_pool1d(text, text.size(2))     #[1, 200, 1]

        topic_text = torch.cat([topic, text], 1)    #[1, 400, 1]

        topic_text = topic_text.squeeze(2)          #[1, 400]

        output = self.linear1(topic_text)
        # output = F.tanh(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output






