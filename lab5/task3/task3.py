import matplotlib.pyplot as plt
import  numpy as np

from task1.task1 import make_PSF
from task2 import *
import cv2

from task2.task2 import make_blurred, inverse


def extension_PSF(image0,PSF0):
    [img_h, img_w] = image0.shape
    [h, w] = PSF0.shape
    PSF = np.zeros((img_h, img_w))
    PSF[0:h, 0:w] = PSF0[0:h, 0:w]
    return PSF

def wiener(input, PSF, K=0.01):  # 维纳滤波
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF)
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(result)
    # result = np.abs(fft.fftshift(result))
    return result

if __name__=="__main__":
    eps=1e-3
    image=cv2.imread(r'../kennysmall.jpg',0)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.figure(1)
    PSF=make_PSF()
    PSF=extension_PSF(image,PSF)
    blurred=make_blurred(image,PSF,eps)
    blurred_noisy=blurred+0.1*blurred.std() * np.random.standard_normal(blurred.shape)
    plt.figure(figsize=(8,6))
    plt.subplot(131)
    plt.axis("off")
    plt.gray()
    plt.title("motion & noisy blurred")
    plt.imshow(blurred_noisy)
    result=inverse(image,PSF,eps)
    plt.subplot(132)
    plt.axis("off")
    plt.title("inverse deblurred")
    plt.imshow(result)
    result=wiener(result,PSF)
    plt.subplot(133)
    plt.title("wienre deblurred(K=0.01)")
    plt.imshow(result)
    plt.savefig('result3.png')
    plt.show()