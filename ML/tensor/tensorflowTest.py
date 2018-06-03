# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:50:54 2018

@author: My
"""

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random
# Parameters
learning_rate = 0.1
training_epochs = 2000
display_step = 50

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
w = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


activation = tf.add(tf.multiply(X, w), b)#y = wx+b
cost = tf.reduce_sum(tf.pow(activation-Y,2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
    print("Optimization Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), \
          "W=", sess.run(w), "b=", sess.run(b))

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


'''
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

print(matrix1)
print(matrix2)
product=tf.matmul(matrix1,matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
'''