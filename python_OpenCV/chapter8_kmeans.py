import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from tools import tools

# make datasets 
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=10)

#plt.scatter(X[:, 0], X[:, 1], s=100)
#plt.show()

import cv2

#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#flags = cv2.KMEANS_RANDOM_CENTERS

#comp, labels, centers = cv2.kmeans(X.astype(np.float32), 4, None, criteria, 10, flags)

#plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#plt.show()
tools = tools.OpenTools()
centers, labels = tools.find_clusters(X, 4)
print(centers)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
