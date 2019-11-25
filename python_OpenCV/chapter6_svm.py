import numpy as np
from sklearn import datasets
from sklearn import model_selection as ms
import cv2
import matplotlib.pyplot as plt

# 生成二分类测试数据 100行2列
x,y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, random_state=7816)

'''plt.scatter(x[:, 0], x[:, 1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()'''

# 数据预处理
x = x.astype(np.float32)
y = y*2 - 1 # -1 或 1
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

# 创建支持向量机
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred = svm.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# 可视化决策边界
def plot_decision_boundary(svm, x_test, y_test):
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() -1, x_test[:, 1].max() + 1
    h = 0.02 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_hypo = np.c_[xx.ravel().astype(np.float32),yy.ravel().astype(np.float32)]
    _, zz = svm.predict(x_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_test[:, 0], x_test[: ,1], c=y_test, s=200)

    return

#plot_decision_boundary(svm, x_test, y_test)

# 设置不同内核
kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF]

for idx, kernel in enumerate(kernels):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel)
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    _, y_pred = svm.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    plt.subplot(2, 2, idx+1)
    plot_decision_boundary(svm, x_test, y_test)
    plt.title('accuracy = %.2f' % accuracy)
    pass

# 自然环境下检测
# 获取数据
data =  "G:\test\opencv-machine-learning\notebooks\data\chapter6"
dataset = "pedestrians128x64"
datafile = 