# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:39:48 2018

@author: My
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn.svm import SVC


#导入数据
df_train = pd.read_csv("../data/train_digitRecognizer.csv")
df_test = pd.read_csv("../data/test_digitRecognizer.csv")
df_target = df_train["label"]
df_train = df_train.drop("label",axis=1)
# 数据归一化 所有像素值>0 的值设为1
df_train[df_train>0] = 1
df_test[df_test>0] = 1

#df_test = df_test.


#model = KNeighborsClassifier(n_neighbors=3)
#model = RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=2,min_samples_leaf=1)
model = SVC()

#model.fit(df_train, df_target)

#preds = model.predict(df_test)
#写到csv
#my_submission = pd.DataFrame({"ImageId":df_test.index,"Label":preds})
#my_submission.to_csv("digitrecognizer_submission_randomforest.csv",index=False)

# 交叉验证
scors = cross_val_score(model, df_train, df_target, cv = 5)
print(scors)