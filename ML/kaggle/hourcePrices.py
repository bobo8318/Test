# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:58:29 2018

@author: My
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df_train = pd.read_csv("train.csv",index_col="Id")
df_test = pd.read_csv("test.csv",index_col="Id")
target = df_train['SalePrice']
#target_test = df_test['SalePrice']
#print(df_train.shape)
df_train = df_train.drop('SalePrice',axis=1)
#print(df_train.shape)
df_train['training_set'] = True
df_test['training_set'] = False
df_full = pd.concat([df_train,df_test])
df_full = df_full.interpolate()
df_full = pd.get_dummies(df_full)

df_train = df_full[df_full["training_set"]==True]
df_train = df_train.drop("training_set",axis=1)

df_test = df_full[df_full["training_set"]==False]
df_test = df_test.drop("training_set",axis=1)

rf = RandomForestRegressor(n_estimators=100,n_jobs=1)
rf.fit(df_train,target)
preds = rf.predict(df_test)
#print(rf.score(df_test))
my_submission = pd.DataFrame({"Id":df_test.index,"SalePrice":preds})
my_submission.to_csv("hp_submission.csv",index=False)

'''
在将训练集和测试集分别加载进 DataFrame 之后，
我保存了目标变量，并在 DataFrame 中删除它（因为我只想保留 DataFrame 中的独立变量和特征）。
随后，我在训练集和测试集中添加了一个新的临时列（'training_set'），
以便我们可以将它们连接在一起（将它们放在同一个 DataFrame 中），然后再将它们分开。
我们继续整合它们，填充缺失的数值，并通过独热编码（One-Hot Encoding）将分类特征转换为数字特征。
'''