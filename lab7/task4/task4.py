import numpy as np
import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

def histogram(image):
    (row, col) = image.shape
    #创建长度为256的list
    hist = [0]*256
    for i in range(row):
        for j in range(col):
            hist[image[i,j]] += 1
    return hist
img = cv2.imread(r'./polygon.jpg',0)
img_noisy = np.uint8(img + 0.8 * img.std() * np.random.standard_normal(img.shape))
img_noisy_hist=histogram(img_noisy)
_,img_seg=cv2.threshold(img_noisy,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# img_smooth=cv2.GaussianBlur(img_noisy,(5,5),0.1)
img_smooth=cv2.blur(img_noisy,(7,7))
img_smooth_hist=histogram(img_smooth)
_,img_seg_smooth=cv2.threshold(img_smooth,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize=(8,5))
plt.axes([0.1, 0.55, 0.2, 0.4])
plt.title("带噪声的图像")
plt.axis("off")
plt.imshow(img_noisy,cmap="gray")
plt.axes([0.4, 0.6, 0.2, 0.28])
plt.title("噪声图像直方图")
# plt.axis("off")
plt.xlabel("灰度值")
plt.ylabel("像素个数")
plt.plot(img_noisy_hist)
plt.axes([0.7, 0.55, 0.2, 0.4])
plt.title("带噪声图像的OTSU分割")
plt.axis("off")
plt.imshow(img_seg,cmap="gray")
plt.axes([0.1, 0.1, 0.2, 0.4])
plt.title("高斯平滑的图像")
plt.axis("off")
plt.imshow(img_smooth,cmap="gray")
plt.axes([0.4, 0.15, 0.2, 0.28])
plt.title("平滑图像直方图")
plt.xlabel("灰度值")
plt.ylabel("像素个数")
plt.plot(img_smooth_hist)
plt.axes([0.7, 0.1, 0.2, 0.4])
plt.title("平滑图像的OTSU分割")
plt.axis("off")
plt.imshow(img_seg_smooth,cmap="gray")
plt.savefig('result4.jpg')
plt.show()