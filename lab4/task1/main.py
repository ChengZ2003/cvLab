import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

#OpenCV函数实现傅里叶变换
img = cv2.imread(r'peppers.bmp',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft) # 将低频分量移至中心

#dft[:,:,0]为傅里叶变换的实部,dft[:,:,1]为傅里叶变换的虚部
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) # 幅值谱
phase_angle = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]) # 相位谱

# 逆变换
img_back = cv2.idft(dft)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.figure(figsize=(10,6))
plt.subplot(151),plt.imshow(img, cmap = 'gray')
plt.title('原图像'), plt.axis('off')
plt.subplot(152),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('幅值谱'), plt.axis('off')
plt.subplot(153),plt.imshow(phase_angle, cmap = 'gray')
plt.title('相位谱'), plt.axis('off')
plt.subplot(154),plt.imshow(img_back, cmap = 'gray')
plt.title('重构图像'),plt.axis('off')
plt.savefig("task1_result.jpg")
plt.show()
