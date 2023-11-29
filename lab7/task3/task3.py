import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img = cv2.imread(r'./paopao.jpg',0)
G1 = np.zeros(img.shape, np.uint8)  # 定义矩阵分别用来装被阈值T1分开的两部分
G2 = np.zeros(img.shape, np.uint8)
T1 = np.mean(img)  #用图像均值做初始阈值
diff=255
T0=0.01
while(diff>T0):
    #THRESH_TOZERO超过T1的像素不变, 其他设为0,THRESH_TOZERO_INV反过来
    _,G1=cv2.threshold(img,T1,255,cv2.THRESH_TOZERO_INV)
    _,G2=cv2.threshold(img,T1,255,cv2.THRESH_TOZERO)
#     plt.imshow(G2,cmap="gray")
    loc1 = np.where(G1>0.001)  #获得非0像素的坐标
    loc2 = np.where(G2 > 0.001)
    ave1=np.mean(G1[loc1])   #求非0像素的均值
    ave2=np.mean(G2[loc2])
    T2=(ave1+ave2)/2.0
    diff=abs(T2 - T1)
    T1=T2
_,img_result=cv2.threshold(img,T2,255,cv2.THRESH_BINARY)
plt.subplot(121)
plt.title("原灰度图像")
plt.axis("off")
plt.imshow(img,cmap="gray")
plt.subplot(122)
plt.title("迭代全阈值分割二值图像")
plt.axis("off")
plt.imshow(img_result,cmap="gray")
plt.savefig('result3.jpg')
plt.show()
