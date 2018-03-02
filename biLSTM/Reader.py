# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:50
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Reader.py
# @Software: PyCharm Community Edition



import re
import random

class reader:

    def __init__(self, filename, needFresh=True):
        self.text = []
        self.label = []
        self.labelType = []
        self.fileText = []
        self.maxLen = 0
        print("Reading "+filename)
        file = open(filename, "r",encoding='utf-8')
        self.fileText = file.readlines()
        file.close()
        if needFresh:
            for i in range(len(self.fileText)):
                # print(i, " raw ", self.fileText[i])
                self.fileText[i] = self.freshData(self.fileText[i])
                # print(i, " new ", self.fileText[i])

        for i in range(len(self.fileText)):
            line = self.fileText[i]
            # print("line = ", line)
            lineList = line.strip().split(" ")
            self.fileText[i] = lineList
            self.text.append(lineList[:-2])
            self.maxLen = max(self.maxLen, len(lineList[:-2]))
            self.label.append(lineList[-1])
            if lineList[-1] not in self.labelType:
                self.labelType.append(lineList[-1])


    def getData(self):
        return self.text, self.label, self.labelType

    def getWholeText(self):
        # random.shuffle(self.fileText)
        return self.fileText

    def freshData(self, string):
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


# data = reader('/Users/zhenranran/Desktop/zenRRan.github.com/Stance Detection/CNN/Data/dev.sd')