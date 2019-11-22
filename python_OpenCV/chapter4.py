from sklearn import preprocessing
import numpy as np

x = np.array([
    [1., -2., 2.],
    [3., 0., 0.],
    [0., 1., -1.]
])

# 标准化 每行均值约等于0 每一行的方差都为1
x_scaled = preprocessing.scale(x) 


# 特征归一化
x_normalized_l1 = preprocessing.normalize(x,norm='l1')
x_normalized_l2 = preprocessing.normalize(x,norm='l2')

# 特征缩放 默认 0-1
min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler = preprocessing.MinMaxScaler(feature_range(-10,10))
x_min_max = min_max_scaler.fit_transform(x)

# 特征二值化

y = np.array([
    [1., -2., 2.],
    [3., 0., 0.],
    [0., 1., -1.]
])

binarizer = preprocessing.Binarizer(threshold=1) # 比1大为1 小于等于1 为0
y_binarizer = binarizer.transform(y)

# 缺失数据处理
from numpy import nan
from sklearn.preprocessing import Imputer

z = np.array([
    [nan, 0, 3],
    [2, 9, -8],
    [1, nan, 1],
    [5, 2, 4],
    [7, 6, -3]
])

# 指定坐标轴上 mean 平均值 median 中值 most_frequest 频率最高值
imp = Imputer(strategy='mean')
z2 = imp.fit_transform(z)

# PCA
mean = [20, 20]
cov = [[5, 0], [25, 25]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T # 一个多元正态分布矩阵


x_ = np.vstack((x,y)).T # 按垂直方向（行顺序）堆叠数组构成一个新的数组
import cv2
mu, eig = cv2.PCACompute(x_, np.array([]))

import  matplotlib.pyplot as plt
plt.style.use('ggplot')

'''
plt.plot(x, y, 'o', zorder=1)
plt.quiver(mean[0], mean[1], eig[:, 0], eig[:, 1], zorder=3, scale=0.2, units='xy')
plt.text(mean[0] + 5 * eig[0, 0], mean[1] + 5 * eig[0, 1], 'u1', zorder = 5 , fontsize = 16, bbox = dict(facecolor='white', alpha=0.6))
plt.text(mean[0] + 7 * eig[1, 0], mean[1] + 4 * eig[1, 1], 'u1', zorder = 5 , fontsize = 16, bbox = dict(facecolor='white', alpha=0.6))
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()
'''

# cv2 旋转
x2 = cv2.PCAProject(x_, mu, eig)
# plt.axis([-20, 20, -10, 10])

# sklearn 进行独立成分分析
from sklearn import decomposition
ica = decomposition.FastICA()
x2 = ica.fit_transform(x_)

plt.plot(x2[:, 0], x2[:, 1], 'o')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.axis([-0.2, 0.2, -0.2, 0.2])
plt.show()