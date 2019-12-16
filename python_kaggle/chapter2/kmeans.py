import numpy as np
import pandas as pd

data_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
data_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)

print(data_test.describe)