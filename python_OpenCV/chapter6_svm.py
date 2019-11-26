import numpy as np
from sklearn import datasets
from sklearn import model_selection as ms
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
'''
# 生成二分类测试数据 100行2列
x,y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, random_state=7816)

plt.scatter(x[:, 0], x[:, 1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

# 数据预处理
x = x.astype(np.float32)
y = y*2 - 1 # -1 或 1
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

# 创建支持向量机
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred = svm.predict(x_test)

#print(metrics.accuracy_score(y_test, y_pred))

# 可视化决策边界
def plot_decision_boundary(svm, x_test, y_test):
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() -1, x_test[:, 1].max() + 1
    h = 0.02 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_hypo = np.c_[xx.ravel().astype(np.float32),yy.ravel().astype(np.float32)]
    _, zz = svm.predict(x_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_test[:, 0], x_test[: ,1], c=y_test, s=200)

    return

#plot_decision_boundary(svm, x_test, y_test)

# 设置不同内核

kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF]

for idx, kernel in enumerate(kernels):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel)
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    _, y_pred = svm.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    plt.subplot(2, 2, idx+1)
    plot_decision_boundary(svm, x_test, y_test)
    plt.title('accuracy = %.2f' % accuracy)
    pass
'''
# 自然环境下检测
# 获取数据
from tools import OpenTools
datadir =  "/opt/gitwb/opencv-machine-learning/notebooks/data/chapter6" #linux
# datadir =  "G:/test/opencv-machine-learning/notebooks/data/chapter6" #win

dataset = "pedestrians128x64"
fade_dataset = "pedestrians_neg"

datafile = "%s/%s.tar.gz" % (datadir, dataset)
fade_dataFile = "%s/%s.tar.gz" % (datadir, fade_dataset)
extractdir = "%s/%s" % (datadir, dataset)
fade_extractdir = "%s/%s" % (datadir, fade_dataset)

img_test_file = "%s/%s" % (datadir, "pedestrian_test.jpg")
mytool = OpenTools()
'''

mytool.extract_tar(datafile, extractdir)

# 看看图片
for i in range(5):
    filename = "%s/per0010%d.ppm" % (extractdir, i)
    #print(filename)
    img = cv2.imread(filename)
    plt.subplot(1, 5, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()
'''
# 方向梯度直方图 HOG
win_size = (48, 96) # 检测窗口大小
block_size = (16,16) # 块大小
block_stride = (8, 8) # 单元格步长
cell_size = (8, 8) # 单元格尺寸
num_bins = 9 # 9个方向的直方图
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

# 随机挑选400张图片 生成正样本
import random

random.seed(42)
x_pose = []
for i in random.sample(range(900), 400):
    filename = "%s/per%05d.ppm" % (extractdir, i+1)
    img = cv2.imread(filename)
    if img is None:
        print("could not find image %s" % filename)
        continue
    x_pose.append(hog.compute(img, (64, 64)))

x_pos = np.array(x_pose, dtype=np.float32)
y_pos = np.ones(x_pos.shape[0], dtype=np.int32)

# 生成负样本
#mytool.extract_tar(fade_dataFile, fade_extractdir)
import os
hroi = 128 # 需要切出来图片的高
wroi = 64 # 需要切出来图片的宽
x_neg = []
for negfile  in os.listdir(fade_extractdir):
    filename = "%s/%s" % (fade_extractdir, negfile)
    img = cv2.imread(filename)
    img = cv2.resize(img, (512, 512))
    for j in range(5):
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y+hroi, rand_x:rand_x+wroi, :]

        x_neg.append(hog.compute(roi, (64, 64)))
    

x_neg = np.array(x_neg, dtype=np.float32)
y_neg = np.ones(x_neg.shape[0], dtype=np.int32)

# 合并正负样本
x = np.concatenate((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

# 实现向量机

svm = cv2.ml.SVM_create()
svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred = svm.predict(x_test)

# 在普通图片中检测行人
img_test = cv2.imread(img_test_file)
stride = 16

 
found = []
for ystart in np.arange(0, img_test.shape[0], stride):
    for xstart in np.arange(0, img_test.shape[1], stride):
        if ystart + hroi > img_test.shape[0]:
            continue
        if xstart + wroi > img_test.shape[1]:
            continue
        roi = img_test[ystart:ystart+hroi, xstart:xstart+wroi, :]
        feat = np.array([hog.compute(roi, (64, 64))])
        _, y_pred = svm.predict(feat)

        if np.allclose(y_pred, 1):
            found.append((ystart, xstart, hroi, wroi))




sv = svm.getSupportVectors()
rho, _, _ = svm.getDecisionFunction(0)# 多尺寸检测

hog.setSVMDetector(np.append(sv.ravel(), rho))
found = hog.detectMultiScale(img_test)

hogdef = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
pdetect = cv2.HOGDescriptor_getDaimlerPeopleDetector()
hogdef.setSVMDetector(pdetect) # opencv 训练好的自带的分类器
found, _ = hogdef.detectMultiScale(img_test)



from matplotlib import patches
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
for f in found:
    ax.add_patch(patches.Rectangle((f[0],f[1]),f[2],f[3],color='y',linewidth=3,fill=False))

plt.show()
