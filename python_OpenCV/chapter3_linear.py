import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

boston = datasets.load_boston()
print(dir(boston))

linreg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = modsel.train_test_split(boston.data, boston.target, test_size=0.1, random_state=42)