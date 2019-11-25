import numpy as np
import cv2
from sklearn import tree
from sklearn import datasets
import sklearn.model_selection as ms
'''
from  sklearn.feature_extraction.text import CountVectorizer
data = []
vec = CountVectorizer()
x = vec.fit_transform(data)

vec.get_feature_names()
'''

data = [{},{}]

# 查出目标项
#target = [d['drug'] for d in data]
# 删除原数据表中的目标项
#[d.pop('drug') for d in data]

# 导出graphviz格式决策树文件

datas = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = ms.train_test_split(datas.data, datas.target, test_size=0.2, random_state=42)

dtc = tree.DecisionTreeClassifier()

dtc.fit(x_train, y_train)