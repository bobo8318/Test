import numpy as np
from sklearn.metrics import pairwise_distances_argmin

class OpenTools():
    def __init__(self):
        pass
    def extract_tar(self, dataFile, extraDir):
        try:
            import tarfile
        except ImportError:
            raise ImportError("tarfile dont install")

        tar = tarfile.open(dataFile)
        tar.extractall(path=extraDir)
        tar.close()
        print("%s successfully extracted to %s" % (dataFile, extraDir))
        pass
    def test(self):
        print("hello world")
        pass
    def find_clusters(self, X, n_clusters, rseed=5):
        rng = np.random.RandomState(rseed)#产生一个2行3列的assarray，其中的每个元素都是[0,1]区间的均匀分布的随机数
        i = rng.permutation(X.shape[0])[:n_clusters] #permutation 随机排列一个序列，返回一个排列的序列
        centers = X[i]
        while True:
            labels = pairwise_distances_argmin(X, centers)
            print(labels)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

            #for i in range(n_clusters):
            if np.all(centers == new_centers):
                break

            centers = new_centers

        return centers, labels
