# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score 
from sklearn import preprocessing

# 加载iris数据集
iris = datasets.load_iris()
# 读取特征
X = iris.data
# 读取分类标签
y = iris.target

X = preprocessing.scale(X)
print(X)
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3, random_state=42);

# 定义分类器
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(train_x,train_y);
print(knn.score(test_x,test_y))
# 进行交叉验证数据评估, 数据分为5部分, 每次用一部分作为测试集
scores = cross_val_score(knn, X, y, cv = 5, scoring = 'accuracy')
# 输出5次交叉验证的准确率
print(scores)

