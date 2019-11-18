import numpy as np
import cv2
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection

import matplotlib.pyplot as plt
plt.style.use("ggplot")

iris = datasets.load_iris()

idx = iris.target != 2
data = iris.data[idx].astype(np.float32)
target = iris.target[idx].astype(np.float32)

'''print(dir(iris))
print(iris.data.shape)
print(iris.feature_names)
print(np.unique(iris.target))



plt.scatter(data[:,0], data[:,1], c=target, cmap=plt.cm.Paired, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()'''

# split data

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.1, random_state=42)

lr = cv2.ml.LogisticRegression_create()
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)
lr.setIterations(100)

lr.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
print(lr.get_learnt_thetas())

ret, y_pred = lr.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
