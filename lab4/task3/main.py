import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

def BLPF(fshift,D0,n):
    rows,cols=fshift.shape
    crow,ccol=rows//2,cols//2
    H=np.zeros((rows,cols))
    for u in range(rows):
        for v in range(cols):
            D=np.sqrt((u-crow)**2+(v-ccol)**2)
            H[u,v]=1/(1+(D/D0)**(2*n))
    fshift1=fshift*H
    return fshift1,H

img=cv.imread(r'alphabet.jpg',0)
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(1+np.abs(fshift))
angle_spectrum=np.abs(np.angle(fshift))
fshift1,H=BLPF(fshift,30,2)
f_ishift=np.fft.ifftshift(fshift1)
img_back=np.fft.ifft2(f_ishift)
img_back=np.abs(img_back)
plt.figure(figsize=(10,5))
plt.subplot(151),plt.imshow(img, cmap = 'gray')  #显示加噪图像
plt.title('原图像'), plt.axis("off")
plt.subplot(152),plt.imshow(magnitude_spectrum, cmap = 'gray')  #显示加噪图像
plt.title('原幅值谱')
plt.axis("off")
plt.subplot(153),plt.imshow(H, cmap = 'gray')
plt.title('巴特沃斯传递函数'), plt.axis("off")
plt.subplot(154),plt.imshow(20*np.log(1+np.abs(fshift1)), cmap = 'gray')  #显示加噪图像
plt.title('滤波后的幅值谱')
plt.axis("off")
plt.subplot(155),plt.imshow(img_back, cmap = 'gray')  #显示加噪图像
plt.title('重构图像')
plt.axis("off")
plt.savefig("task3_result.jpg")
plt.show()
