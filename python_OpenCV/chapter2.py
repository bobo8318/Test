import cv2
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


digits_data = datasets.load_digits()

for image_index in range(10):
    subplot_index  = image_index+1
    plt.subplot(2,5,subplot_index)
    plt.imshow(digits_data.images[image_index, :, :], cmap="gray")
plt.show()