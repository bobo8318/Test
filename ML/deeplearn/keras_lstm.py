# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 07:32:15 2018

@author: My
"""
from keras.models import Sequential
import numpy as np
from keras.layers import Dense,TimeDistributed,LSTM
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

def get_batch():
    global BATCH_START,TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

print(get_batch())
