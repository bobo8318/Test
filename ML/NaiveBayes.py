import numpy as np
from math import log
import re

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','has','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage','help','please'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']
                   ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
pass


def createVocabList(dataset):
    """创建词汇表 将所有出现的词放到一个集合中"""
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)
    pass
    return list(vocabSet)
pass

def setOfWords2Vec(vocabList,inputSet):
    """根据词汇表 生成出现向量
    vocabList 包含所有词的词汇表
    inputSet 待生成向量的原始词列表
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("")
    return returnVec
pass

def trainNB0(trainMatrix,trainCategory):
    """根据训练集 生成每个词（每列）在不同分类下的出现概率
        计算在每个分类下 每个词出现的总数 /所有词出现的总数

        trainMatrix 训练集
        trainCategory 分类向量
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbsuive = np.sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords);p1Num = np.ones(numWords)
    p0Demon = 2.0;p1Demon = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            print("trainMatrix %d"%i)
            print(trainMatrix[i])
            print("p1Num %d"%i)
            print(p1Num)
            p1Demon += np.sum(trainMatrix[i])
            print("p1Demon %d"%i)
            print(p1Demon)
        else:
            p0Num += trainMatrix[i]
            p0Demon += np.sum(trainMatrix[i])
    pass
    p1vect = log(p1Num/p1Demon)
    p0vect = log(p0Num/p0Demon)
    return p0vect,p1vect,pAbsuive
pass

def bagOfWordsToVectMN(vocabList,inputSet):
    """词袋模型 计算个词的出现次数"""
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1;
        pass
    pass
pass

def classifyNB(vec2classify,p0vec,p1vec,pClass1):
    """根据之前算出的每个词出现的概率向量 计算出 待分类向量vec2classify的分类概率
    vec2classify 待分类向量
    p0vec 0分类所有概率向量
    p1vec 1分类所有概率向量
    pClass1 1分类出现的概率
    """
    p1 = sum(vec2classify*p1vec)+log(pClass1)
    p0 = sum(vec2classify*p0vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
pass

def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]
pass