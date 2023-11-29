# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt
import cv2
import numpy as np
#%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img=cv2.imread(r"moon.jpg",0)

#拉普拉斯锐化：
fi1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
fi2=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
#使用opencv的卷积函数 滤波去噪声
img1 = cv2.filter2D(img,-1,fi1,borderType=cv2.BORDER_DEFAULT)
img2 = cv2.filter2D(img,-1,fi2,borderType=cv2.BORDER_DEFAULT)

#输出对应的图片
plt.figure(figsize=(12,6))
plt.subplot(131)
plt.imshow(img,vmin=0,vmax=255,cmap=plt.cm.gray)#设置内部的坐标，以及灰度的最大值和最小值
plt.title("原图像")
plt.axis('off')
plt.subplot(132)
plt.imshow(img1,vmin=0,vmax=255,cmap=plt.cm.gray)
plt.title("拉普拉斯滤波图像")
plt.axis('off')
plt.subplot(133)
plt.imshow(img2,vmin=0,vmax=255,cmap=plt.cm.gray)
plt.title("拉普拉斯锐化增强图像")
plt.axis('off')

plt.savefig("ch3-43.jpg")
plt.show()