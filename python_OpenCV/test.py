import numpy as np
from tools import tools
'''
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
'''

datadir =  "/opt/gitwb/opencv-machine-learning/notebooks/data/chapter6" #linux


#dataset = "pedestrians128x64"
#datafile = "%s/%s.tar.gz" % (datadir, dataset)
#extractdir = "%s/%s" % (datadir, dataset)

#fade_dataset = "pedestrians_neg"
#fade_dataFile = "%s/%s.tar.gz" % (datadir, fade_dataset)
#fade_extractdir = "%s/%s" % (datadir, fade_dataset)


tool = tools.OpenTools()
#tool.extract_tar(fade_dataFile, fade_extractdir)
tool.test()
