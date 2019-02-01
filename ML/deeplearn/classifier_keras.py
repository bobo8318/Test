# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 07:28:58 2018

@author: My
"""
import keras
print(keras.__version__)
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize  X_train.shape[0]所有数据行数 -1自动识别多少列
X_test = X_test.reshape(X_test.shape[0], -1) / 255.   # normalize  X_train.shape[0]所有数据行数 -1自动识别多少列

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#建立神经网络
'''
model = Sequential()
model.add(Dense(units=32, input_dim=784))# 第一层神经网络
model.add(Activation('relu'))# 激活函数
model.add(Dense(units=10))# 第二层神经网络
model.add(Activation('softmax'))# 激活函数
'''
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# metrics 里面可以放入需要计算的 cost，accuracy，score 等

# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)


#测试模型
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)