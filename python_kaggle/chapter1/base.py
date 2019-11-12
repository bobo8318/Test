import pandas as pd
import numpy as np
import os

#os.getcwd()

# 创建特征列表
column_names = ['Sample_code_number','Clump Thickness','Uniformity of cell Size','Uniformity of Cell Shape','Marginal Adhension'
,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('python_kaggle/chapter1/breast-cancer-wisconsin.data',column_names,engine='python')
# ? 缺失值替换
data = data.replace(to_replace='?',value=np.nan)
# 丢弃带有缺省值的数据
data = data.dropna(how='any')
data.shape