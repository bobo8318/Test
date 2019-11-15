import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

x = np.linspace(0,10,100)
y_true = np.sin(x) + np.random.rand(x.size)-0.5
y_pred = np.sin(x)

plt.style.use('ggplot')
plt.plot(x,y_pred,linewidth=4,label='model')
plt.plot(x,y_true,'o',label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')

plt.show()
# 计算准确率
# 均方误差
mse = np.mean((y_true-y_pred)**2)
mse_sk = metrics.mean_squared_error(y_true, y_pred)
# 数据离散度
fvu = np.var(y_true - y_pred) / np.var(y_true)
# 可释方差
fvu = 1.0 - fvu

fvu_sk = metrics.explained_variance_score(y_true,y_pred)

# 决定系数 R**2
r2 = 1.0 -mse / np.var(y_true)
r2_sk = metrics.r2_score(y_true,y_pred)
