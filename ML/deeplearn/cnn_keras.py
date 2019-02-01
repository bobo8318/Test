# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:20:58 2018

@author: My
"""
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam
from keras.utils import np_utils

model = Sequential()


model.add(Conv2D(
filters = 32, #第一层先设置32个滤波器
kernel_size=(5,5),#设置滤波器大小为5*5
padding = 'same',#选择滤波器的扫描方式，即是否考虑边缘
input_shape = (1,28,28)#初次设置卷积层需要设置输入的形状
))
model.add(Activation('relu'))


model.add(MaxPooling2D(
pool_size = (2,2),#设置为2*2的池化块
strides = (2,2),#向右向下的步长
padding = 'same'
))

model.add(Conv2D(64,(5,5),padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(strides = (2,2),padding = 'same'))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('solftmax'))


