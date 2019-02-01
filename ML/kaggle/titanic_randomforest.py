# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 17:12:27 2018

@author: My
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict 

# 加载数据
df_train = pd.read_csv("train_titanic.csv",index_col="PassengerId")
df_test = pd.read_csv("test_titanic.csv",index_col="PassengerId")

df_target = df_train["Survived"]

# 数据预处理：缺失值填充
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())#缺失值填充
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())#缺失值填充
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())#缺失值填充

#
df_train["Sex"] = pd.get_dummies(df_train["Sex"]) 
df_test["Sex"] = pd.get_dummies(df_test["Sex"])

df_train["Embarked"] = df_train["Embarked"].fillna("S")
df_test["Embarked"] = df_test["Embarked"].fillna("S")

#print(df_train["Embarked"])

df_train["Embarked"] = pd.get_dummies(df_train["Embarked"])
df_test["Embarked"] = pd.get_dummies(df_test["Embarked"])
#print(df_train["Embarked"])
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
df_train = df_train[predictors]

model = RandomForestClassifier(random_state=1,n_estimators=10,min_samples_split=2,min_samples_leaf=1);
model.fit(df_train,df_target)

preds = model.predict(df_test[predictors])
my_submission = pd.DataFrame({"PassengerId":df_test.index,"Survived":preds})
my_submission.to_csv("titanic_submission_randomforest.csv",index=False)

#scores = cross_val_score(model, df_train, df_target, cv = 5)
#print(scores.mean())