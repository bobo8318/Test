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
    def predict(self):
        _, self.y_pred = self.model_norm.predict(self.x_test)
        return self.y_pred
    # 进行评估
    def score(self):
        return metrics.accuracy_score(self.y_test, self.y_pred)
    # 决策边界
    def getXHypo(self):
        print("-----------------------------------")
        print(self.x_test)
        self.x_hypo = plotTools.plot_decision_boundary(self.x_test)
        print("-----------------------------------")
        return self.x_hypo
if __name__ == ("__main__"):
    
    bayes = Bayes()
    bayes.createData()
    bayes.splitData()
    bayes.createModel()
    #bayes.predict()
    print(bayes.getXHypo())
