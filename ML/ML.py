import numpy as np
import pandas as pd
class Perceptron(object):

    def __init__(self,eta=0.01,n_iter=10):
        """
        eta：学习率
        n_iter：权重向量训练次数
        w_:权重向量
        errors_:错误次数
        """
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self,x,y):
        """
        输入训练样本，培训神经元，x 输入样本向量，y对应样本分类
        x：shape[n_samples,n_features]
        x:[[1,2,3],[4,5,6]]
        """

        """
        初始化权重向量为零
        加1 是w0
        """
        self.w_ = np.zero(1+x.shape[1]);
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0;
            for xi,target in zip(x,y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update;

                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass

            pass
        pass
    def net_input(self,x):
        """
        :param x:
        :return:
        z = w0*i+....
        """
        return np.dot(x,self.w_[1:])+self.w_[0]
        pass

    def predict(self,x):
        return np.where(self.net_input(x) >= 0.0 ,1,-1)
        pass
    pass
pass
"""
import  matplotlib.pyplot as plt
file ="E:/cp/lottery.csv"
df = pd.read_csv(file,header=None)
y = df.loc[0:100,0].values
y = np.where(y == 2003008,-1,1)
x = df.iloc[0:100,[1,2]].values
print(x)
plt.scatter(x,y,color='red',marker='o',label='ttt')
"""

"""
将数据集按照splitRatio 比例随机分成 训练集 和测试集
"""
import random
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
pass

dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset, splitRatio)
print('Split {0} rows into train with {1} and test with {2}').format(len(dataset), train, test)

