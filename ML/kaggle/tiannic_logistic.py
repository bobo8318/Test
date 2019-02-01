# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:01:04 2018

@author: My
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict 

# 加载数据
df_train = pd.read_csv("train_titanic.csv",index_col="PassengerId")
df_test = pd.read_csv("test_titanic.csv",index_col="PassengerId")

df_target = df_train["Survived"]
#print(df_train.describe())

# 数据预处理：缺失值填充
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())#缺失值填充
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())#缺失值填充
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())#缺失值填充
#print(df_train["Sex"].unique())

#df_train.loc[df_train["Sex"]=="male","Sex"] = 0
#df_train.loc[df_train["Sex"]=="female","Sex"] = 1
#print(df_train["Sex"].unique())

df_train["Sex"] = pd.get_dummies(df_train["Sex"]) 
df_test["Sex"] = pd.get_dummies(df_test["Sex"]) 
#print(df_train["Sex"].unique())

df_train["Embarked"] = df_train["Embarked"].fillna("S")
df_test["Embarked"] = df_test["Embarked"].fillna("S")

#print(df_train["Embarked"].unique())

df_train.loc[df_train["Embarked"]=="S","Embarked"] = 0
df_train.loc[df_train["Embarked"]=="C","Embarked"] = 1
df_train.loc[df_train["Embarked"]=="Q","Embarked"] = 2

df_test.loc[df_test["Embarked"]=="S","Embarked"] = 0
df_test.loc[df_test["Embarked"]=="C","Embarked"] = 1
df_test.loc[df_test["Embarked"]=="Q","Embarked"] = 2

predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
df_train = df_train[predictors]
#print(df_train.describe())
#model = LinearRegression()# 线性回归
model = LogisticRegression()# 逻辑回归
kf = KFold(df_train.shape[0],n_folds=3,random_state=1)
predictions = []
#scores = cross_val_score(model, df_train, df_target, cv = 5)
#scores = cross_val_predict(model, df_train, df_target, cv = 5)
#print(scores.mean())
model.fit(df_train,df_target)

print(df_test[predictors].describe())

preds = model.predict(df_test[predictors])
my_submission = pd.DataFrame({"PassengerId":df_test.index,"Survived":preds})
my_submission.to_csv("titanic_submission.csv",index=False)
'''
for train,test in kf:#交叉验证 验证模型好坏
    train_predictors = df_train[predictors].iloc[train]# 交叉验证训练集数据
    train_target = df_train["Survived"].iloc[train]
    model.fit(train_predictors,train_target)
    
    test_predictions = model.predict(df_train[predictors].iloc[test,:])
    
    predictions.append(test_predictions)

predictions = np.concatenate(predictions,axis=0)
predictions[predictions>0.5] = 1
predictions[predictions<=0.5] = 0
accuracy = sum(predictions[predictions == df_train["Survived"]])/len(predictions)

print(accuracy)
'''