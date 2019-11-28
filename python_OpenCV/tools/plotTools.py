import numpy as np
def plot_decision_boundary(X_test):
    # create
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))# 生成网格点坐标矩阵
    x_hypo = np.column_stack((xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)))
    return x_hypo