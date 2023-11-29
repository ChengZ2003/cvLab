# coding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np

from task1.task1 import make_PSF

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

def extension_PSF(image0,PSF0):
    [ img_h, img_w]=image0.shape
    [ h,w ] = PSF0.shape
    PSF=np.zeros((img_h,img_w))
    PSF[0:h,0:w]= PSF0[0:h,0:w]
    return PSF

def make_blurred(input,PSF,eps):
    input_fft = np.fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = np.fft.fft2(PSF)+eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(blurred)
    return blurred

def inverse(input, PSF,eps):  # 逆滤波
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps  # 为了避免分母为零，将PSF的傅里叶变换加一个极小值eps
    result = np.fft.ifft2(input_fft / PSF_fft)  #
    result = np.abs(result)
    return result

if __name__=="__main__":
    img = cv2.imread(r'../kennysmall.jpg',0)
    eps=1e-3
    plt.figure(figsize=(8,6))
    plt.subplot(131)
    plt.axis("off")
    plt.title("Original image")
    plt.gray()
    plt.imshow(img)

    PSF=make_PSF(15,60)
    PSF=extension_PSF(img,PSF)
    blurred=make_blurred(img,PSF,eps)
    plt.subplot(132)
    plt.axis("off")
    plt.title("Motion blurred")
    plt.imshow(blurred)
    result=inverse(img,PSF,eps)
    plt.subplot(133)
    plt.axis("off")
    plt.title("inverse deblurred")
    plt.imshow(result)
    plt.savefig('result2.png')
    plt.show()

