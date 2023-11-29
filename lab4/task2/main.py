import cv2
import numpy as np
from math import *
import random
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

def addGaussianNoise(src,means=0,sigma=0.1):
    NoiseImg=src/src.max()
    rows=NoiseImg.shape[0]
    cols=NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            #python里使用random.gauss函数加高斯噪声
            NoiseImg[i,j]=NoiseImg[i,j]+random.gauss(means,sigma)
            if  NoiseImg[i,j]< 0:
                 NoiseImg[i,j]=0
            elif  NoiseImg[i,j]>1:
                 NoiseImg[i,j]=1
    return NoiseImg

img0 = cv2.imread(r'peppers.bmp',0)
img=addGaussianNoise(img0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum0 = 20*np.log(1+np.abs(fshift))
plt.figure(figsize=(10,5))
plt.subplot(141),plt.imshow(img, cmap = 'gray')  #显示加噪图像
plt.title('噪声图像'), plt.axis("off")
plt.subplot(142),plt.imshow(magnitude_spectrum0, cmap = 'gray')  #显示加噪图像
plt.title('噪声图像幅值谱')
plt.axis("off")
#进行理想低通滤波
r=50      #截止频率的设置
[m,n]=fshift.shape
H=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        d=sqrt((i-m/2)*(i-m/2)+(j-n/2)*(j-n/2))
        if d<r:
            H[i,j]= 1
G=H*fshift
magnitude_spectrum1 =20*np.log(1+np.abs(G))  #理想低通滤波后的幅值谱
f1 = np.fft.ifftshift(G)
img1 = abs(np.fft.ifft2(f1))   #重构图像
plt.subplot(143),plt.imshow(magnitude_spectrum1, cmap = 'gray')  #显示滤波后幅值谱
plt.title('ILPF滤波后幅值谱'), plt.axis("off")
plt.subplot(144),plt.imshow(img1, cmap = 'gray')  #显示重构图像
plt.title('ILPF滤波后重构图像'), plt.axis("off")
plt.savefig("task2_result.jpg")
plt.show()
