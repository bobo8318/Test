from math import log
import operator

def calcShannonEnt(dataSet):
    """计算信息熵"""
    numEntries = len(dataSet)
    labelCount = {}
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        pass
        labelCount[currentLabel] += 1
    pass
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    pass
    return shannonEnt
pass

def spilDataSet(dataset,axis,value):
    """ 数据分割 取出axis 索引下 值为value的所有向量
     dataset 数据集
     axis 分割属性 索引
     value 分割值
     extend 扩展 将参数中的每个元素 单独放入向量中
     append 添加 参数作为一个元素添加到向量中
     """
    retDataSet = []
    for featVect in dataset:
        if featVect[axis] == value:
            readuceVect = featVect[:axis]
            readuceVect.extend(featVect[axis+1:])
            retDataSet.append(readuceVect)
        pass
    pass
    return retDataSet;
pass

def chooseBestFeatureToSplit(dataset):
    """数据集的列数 -1 用于循环计算分割后的熵 """
    numFeatures = len(dataset[0])-1
    """初始熵 """
    baseEntory = calcShannonEnt(dataset)
    bestInfoGain = 0.0; bestFeatur = -1
    """对每列进行计算 熵 取增益最大 """
    for i in range(numFeatures):
        """将dataset 中 第 i列元素 全部取出 生成新的向量 """
        featList = [example[i] for example in dataset ]
        """ 该列所有元素的 唯一值 去掉重复"""
        uniqueVals = set(featList)
        newEntory = 0.0
        """根据i列的 去重唯一值对 dataset进行分割 并求出分割后的信息熵增益"""
        for value in uniqueVals:
            subDataSet = spilDataSet(dataset,i,value)
            prob = len(subDataSet)/float(len(dataset))
            newEntory += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntory - newEntory
        """取最大增益 """
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeatur = i
    return bestFeatur
pass

def majorityCnt(classList):
    """ 当划分到最后一列数据后 扔有为完全分类的节点 将剩余节点判断为类型最多的 类型"""
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    pass
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(0),reverse=True)
    return sortedClassCount[0][0]
pass

def creatTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    """剩下的元素都一样 停止"""
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    """到最后一列 返回最后的叶子节点类型"""
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(spilDataSet(dataset,bestFeat,value),subLabels)

    return myTree
pass