import cv2
import numpy as np

from task1.task1 import make_PSF
from task2.task2 import make_blurred
from task3.task3 import extension_PSF, wiener


def meanSquare(image0,image1):
    [m,n]=image0.shape
    MSE=0
    for i in range(m):
        for j in range(n):
            MSE = MSE + (image0[i, j] - image1[i, j]) ** 2
    MSE=MSE/(m*n)
    return MSE
if __name__=="__main__":
    image=cv2.imread('../kennysmall.jpg')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    PSF=make_PSF()
    PSF=extension_PSF(image,PSF)
    blurred=make_blurred(image,PSF,1e-3)
    blurred_noisy=blurred+0.5*blurred.std()*np.random.standard_normal(blurred.shape)
    K=0.005
    MSE_min=np.inf
    while K<0.5:
        restruct=wiener(blurred_noisy,PSF,1e-3)
        MSE=meanSquare(image, restruct)
        if MSE<MSE_min:
            K_best=K
            MSE_min=MSE
        K += 0.005
    print('The best K is',K_best)