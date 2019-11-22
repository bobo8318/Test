import numpy as np

x = np.array([
    [1., -2., 2.],
    [3., 0., 0.],
    [0., 1., -1.]
])

y = np.array([
    [1., -2., 2.],
    [3., 0., 0.],
    [0., 1., -1.]
])

x_y= np.vstack((x,y))

print(x_y)