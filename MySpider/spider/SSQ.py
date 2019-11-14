# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:16:47 2019

@author: My
"""
import pandas as pd

class SSQ(object):
    def __init__(self):
        self.data = "test"
        self.path = "ssq.csv"
    pass
    def importData(self):
        df = pd.read_csv(self.path,delimiter=' ')
 
        
        data = df.loc[0:,["期号", "日期" ,"红球1" ,"红球2", "红球3" ,"红球4" ,"红球5", "红球6" ,"篮球"]]
        return data
        
    pass
        

if __name__ == '__main__':
   ssq = SSQ()
   ssq.importData()

pass