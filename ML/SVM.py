import random

from numpy import mat, shape, zeros, multiply


def loadDataSet(fileName):
    """
    :param 加载数据文件:
    :return: 数据文件 类别标签
    """
    dataMat = [] ; labelMat = []
    fr = open(fileName)
    for line in fr.readline():
        dataMat.append([float(line[0],float(line[1]))])
        labelMat.append(float(line[2]));
        pass
    return dataMat,labelMat
    pass

def selectJRand(i,m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
        pass
    return j;
    pass

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    pass
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):

    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0;m,n = shape(dataMatrix)
    alphas = mat(zeros(m,1))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanges = 0;
        for i in range(m) :
            fXi = float(multiply(alphas,labelMat)).T*(dataMatrix*dataMatrix[i,:].T)+b
            Ei = fXi - float(labelMat[i])
            pass
        pass
    pass
