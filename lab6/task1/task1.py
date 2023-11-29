import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img = cv2.imread(r'alphabet.jpg',0)
# 平移
dx=dy=50
rows,cols= img.shape[:2]
M1 = np.float32([[1,0,dx],[0,1,dy]])  #平移变换矩阵
dst1 = cv2.warpAffine(img,M1,(cols,rows))
# 旋转
M2=cv2.getRotationMatrix2D((cols/2,rows/2),30 ,1)
dst2=cv2.warpAffine(img,M2,(cols,rows))
# 水平镜像
M3 = np.float32([[1,0,0],[0,-1,rows]])
dst3 = cv2.warpAffine(img,M3,(cols,rows))
plt.figure(figsize=(12,5))
plt.subplot(141)
plt.imshow(img,cmap="gray")
plt.title('原图像'),plt.axis("off")
plt.subplot(142)
plt.imshow(dst1,cmap="gray")
plt.title('平移变换'),plt.axis("off")
plt.subplot(143)
plt.imshow(dst2,cmap="gray")
plt.title('旋转变换'),plt.axis("off")
plt.subplot(144)
plt.imshow(dst3,cmap="gray")
plt.title('水平镜像'),plt.axis("off")
plt.savefig('result1.jpg')
plt.show()