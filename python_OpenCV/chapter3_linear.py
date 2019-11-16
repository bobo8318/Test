import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

boston = datasets.load_boston()

# use model
linreg = linear_model.LinearRegression()

# split data
X_train, X_test, y_train, y_test = modsel.train_test_split(boston.data, boston.target, test_size=0.1, random_state=42)
linreg.fit(X_train,y_train)


squared_error = metrics.mean_squared_error(y_train, linreg.predict(X_train))
print(squared_error)

# R**2
score = linreg.score(X_train, y_train)
print(score)

# test model
y_pred = linreg.predict(X_test)
squared_error = metrics.mean_squared_error(y_test, y_pred)
print(squared_error)

