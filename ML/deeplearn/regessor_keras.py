# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:31:06 2018

@author: My
"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # 可视化模块

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()


X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points

model = Sequential()
model.add(Dense(units=1, input_dim=1))# x y 是一维的
#model.add(Dense(units=1))# 新的全连接层会自动将上层的output 设为本层的input

model.compile(loss='mse', optimizer='sgd')
# mse 均方误差 sgd 随机梯度下降法

print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)#散点图
plt.plot(X_test, Y_pred)#直线图
plt.show()