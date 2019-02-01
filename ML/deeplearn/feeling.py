# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 22:24:53 2018

@author: My
"""

import tflearn
from tflearn.data_utils import to_categorical,pad_sequences
from tflearn.datasets import imdb
import numpy as np

#path 是存储的路经，pkl 是 byte stream 格式，用这个格式在后面比较容易转换成 list 或者 tuple。 n_words 为从数据库中取出来的词个数
train,test, _ = imdb.load_data(path='imdb.pkl',n_words=10000,valid_portion=0.1)

trainX,trainY = train
testX,testY = test
#print(trainX[0:5])
#pad sequence 将 inputs 转化成矩阵形式，并用 0 补齐到最大维度，这样可以保持维度的一致性
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
#print(trainX[0:5])

# 输入层，batch size 设为 None，length＝100=前面的max sequence length
net = tflearn.input_data([None, 100])
# 上一层的输出作为下一层的输入，input_dim 是前面设定的从数据库中取了多少个单词，output_dim 就是得到 embedding 向量的维度
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
# 模型用的 LSTM，可以保持记忆，dropout 为了减小过拟合
net = tflearn.lstm(net, 128, dropout=0.8)
'''
fully_connected 是指前一层的每一个神经元都和后一层的所有神经元相连， 
将前面 LSTM 学习到的 feature vectors 传到全网络中，可以很轻松地学习它们的非线性组合关系 
激活函数用 softmax 来得到概率值
'''
net = tflearn.fully_connected(net, 2, activation='softmax')

# 最后应用一个分类器，定义优化器，学习率，损失函数
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')

# Training
#模型初始化
model = tflearn.DNN(net, tensorboard_verbose=0)

#show_metric=True 可以看到过程中的准确率
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32)