import numpy as np
from tools import plotTools
from sklearn import datasets
from sklearn import model_selection as ms
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2


class Bayes():
    def __init__(self):
        pass
    # 创建数据集
    def createData(self):
        self.X, self.y = datasets.make_blobs(100, 2, centers=2, random_state=1701, cluster_std=2)#聚类数据生成器
        self.X = self.X.astype(np.float32)
        pass
    # 显示数据集
    def showData(self):
        plt.style.use("ggplot")
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50)
        plt.show()
        pass
    # 划分数据集
    def splitData(self):
        self.x_train, self.x_test, self.y_train, self.y_test = ms.train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        pass
    # 创建贝叶斯分类器
    def createModel(self):
        self.model_norm = cv2.ml.NormalBayesClassifier_create()
        self.model_norm.train(self.x_train, cv2.ml.ROW_SAMPLE, self.y_train)
    # 进行预测
    def predict(self, testdata):
        if testdata == None:
            
            _, self.y_pred = self.model_norm.predict(self.x_test)
        else:
            print('use test data')
            _, self.y_pred = self.model_norm.predict(testdata)
        return self.y_pred
    # 进行评估
    def score(self):
        return metrics.accuracy_score(self.y_test, self.y_pred)
    # 决策边界
    def getXHypo(self):
        #print("-----------------------------------")
        #print(self.x_test)
        self.x_hypo = plotTools.plot_decision_boundary(self.x_test)
        #print("-----------------------------------")
        return self.x_hypo
    # show result plt
    def showResultPlt(self,xx,yy,zz):
        plt.contourf(xx,yy,zz, cmap=plt.cm.coolwarm, alpha=0.8)# 画等高线
        plt.scatter(self.x_test[:, 0], self.x_test[:, 1], c=self.y_test, s=200)
        plt.show()
    def getZZ(self, ret, xx):
        if isinstance(ret, tuple):#cv2 result
            print("cv2")
            zz = ret[1]
        else:# sklearn result
            print("sklearn")
            zz = ret
        zz = zz.reshape(xx.shape)
        return zz
if __name__ == ("__main__"):
    
    bayes = Bayes()
    bayes.createData()
    bayes.splitData()
    bayes.createModel()
    #bayes.predict()
    #print(bayes.getXHypo())
    xx, yy, x_hypo = bayes.getXHypo()
    ret = bayes.predict(x_hypo)
    print("---------------------------")
    print(ret.shape)
    print("---------------------------")
    zz = bayes.getZZ(ret, xx)
    bayes.showResultPlt(xx, yy, zz)
    
