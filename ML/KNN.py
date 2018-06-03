import numpy as np
import operator

def autoNorm(dataset):
    """ newvalue = oldvalue-minvalue/maxvalue-minvalue"""
    """一列的最大值 最小值 范围 向量"""
    minvalues = dataset.min(0)
    maxvalues = dataset.max(0)
    ranges = maxvalues - minvalues
    """同大小的0矩阵"""
    normDataSet = np.zeros(np.shape(dataset))
    """行数"""
    m = normDataSet.shape[0]

    normDataSet = dataset - np.tile(minvalues,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return  normDataSet,ranges,minvalues
pass

def KNN(intx,dataset,labels,k):
    """intx 分类向量
      dataset 训练样本
      labels 标签
      k 选取近邻数量
      """
    datasize = dataset.shape[0]
    """计算距离"""
    diffMat = np.tile(intx,(datasize,1)) - dataset;
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndices = distance.argsort()
    classCount = {}
    """选择距离最小的k个点 对分类进行统计"""
    for i in range (k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] =  classCount.get(voteILabel,0)+1
    pass
    """对统计后的分类进行排序"""
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    """最近的一个返回"""
    return sortedClassCount[0][0]
pass