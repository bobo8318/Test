# -*- coding: utf-8 -*-
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import cross_val_score 


# 将资料分割成train与test的模块
from sklearn.model_selection import train_test_split

# 生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification 

# Support Vector Machine中的Support Vector Classifier
from sklearn.svm import SVC 

# 可视化数据的模块
import matplotlib.pyplot as plt 

a = np.array([
            [10,2.4,3.6],
            [-100,5,-2],
            [120,20,40],
        ],dtype=np.float64)
print(preprocessing.scale(a))
'''
# n_samples：指定样本数
# n_features：指定特征数
# n_classes：指定几分类
# random_state：随机种子，使得随机状可重
'''
X, y = make_classification(
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2, 
    random_state=22, n_clusters_per_class=1, 
    scale=100)
#X = preprocessing.scale(X)
X = preprocessing.minmax_scale(X,feature_range=(-1,1))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)
clf = SVC()
clf.fit(X_train,y_train)
#print(clf.score(X_test,y_test))
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

# 1. 基于mean和std的标准化
#scaler = preprocessing.StandardScaler().fit(a)
#print(scaler.transform(a))

scores = cross_val_score(clf, X, y, cv = 5, scoring = 'accuracy')
print(scores)
print(clf.score(X_test,y_test))