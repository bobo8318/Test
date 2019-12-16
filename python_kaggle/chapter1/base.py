import pandas as pd
import numpy as np
import os

#os.getcwd()

# 创建特征列表
column_names = ['Sample_code_number','Clump Thickness','Uniformity of cell Size','Uniformity of Cell Shape','Marginal Adhension'
,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('./python_kaggle/chapter1/breast-cancer-wisconsin.data',names = column_names)
# ? 缺失值替换
data = data.replace(to_replace='?',value=np.nan)
# 丢弃带有缺省值的数据
data = data.dropna(how='any')

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 从sklearn.linear_model里导入LogisticRegression与SGDClassifier。
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

lr = LogisticRegression()
sgdc = SGDClassifier()


# 调用LogisticRegression中的fit函数/模块用来训练模型参数。
lr.fit(X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果储存在变量lr_y_predict中。
lr_y_predict = lr.predict(X_test)

# 从sklearn.metrics里导入classification_report模块。
from sklearn.metrics import classification_report

print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))