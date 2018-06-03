# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:22:18 2018

@author: My
"""

import numpy as np
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD#随机梯度下降算法
from keras.datasets import mnist  
from sklearn.datasets import load_iris#鸢尾花数据集
iris = load_iris()
print(iris['target'])
from sklearn.preprocessing import LabelBinarizer
print(LabelBinarizer().fit_transform(iris["target"]))
#from sklearn.eross_validation import train_test_split
#train_data,test_data,tarin_target,test_target = train_test_split(iris.data,iris.target,test_size)


