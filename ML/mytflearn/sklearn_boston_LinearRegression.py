# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold

loaded_data = datasets.load_boston()

data_x = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_x,data_y)

'''
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)#产生数据

print(X.shape)
print(y.shape)

plt.scatter(X, y)
plt.show()
'''
#print(model.coef_)# y = 0.1x+0.3 中的0.1
#print(model.intercept_)# y = 0.1x+0.3 中的0.3

##print(model.get_params())#之前定义的参数

print(model.score(data_x,data_y))#准确率

