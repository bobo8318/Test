# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 23:54:04 2018

@author: My
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 加载iris数据集
iris = load_iris()
# 读取特征
X = iris.data
# 读取分类标签
y = iris.target

X = preprocessing.scale(X)

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3, random_state=42);



print(np_utils.to_categorical(iris['target']))
