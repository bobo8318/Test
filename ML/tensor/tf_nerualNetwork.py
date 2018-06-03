# -*- coding: utf-8 -*-
"""
Created on Mon May 14 07:26:19 2018

@author: My
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import flask as fl

Batch_Size = 8
seed = 23455

# 基于seed的随机数
rng = np.random.RandomState(seed)
X = rng.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X]

# 正向传播
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#反向传播
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)#梯度下降
#train_step = tf.train.MomentumOptimizer(0.01).minimize(loss)#
#train_step = tf.train.AdamOptimizer(0.01).minimize(loss)#

with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #训练前参数
        print("W1:\n",sess.run(w1))
        print("W1:\n",sess.run(w2))
        
        STEPS = 3000
        for i in range(STEPS):
            start = (i*Batch_Size) % 32
            end = start+Batch_Size
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        
        
         #训练后参数
        print("--------------------------------------\n")
        print("W1:\n",sess.run(w1))
        print("W1:\n",sess.run(w2))