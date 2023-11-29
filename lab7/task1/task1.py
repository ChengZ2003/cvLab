import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img = cv2.imread(r'fingerprint.png', 0)
_, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernelSize=5
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
img_erode = cv2.erode(img_binary,kernel,iterations = 1)
img_open = cv2.dilate(img_erode, kernel,iterations=1)
img_dilate = cv2.dilate(img_open.copy(), kernel, iterations=1)
img_close = cv2.erode(img_dilate,kernel,iterations = 1)

plt.figure(figsize=(12,6.2))
plt.subplots_adjust(wspace=0,hspace=0)
plt.subplot(231)
plt.axis("off")
plt.imshow(img_binary,cmap="gray")
plt.subplot(232)
plt.axis("off")
plt.imshow(img_erode,cmap="gray")
plt.subplot(233)
plt.axis("off")
plt.imshow(img_open,cmap="gray")
plt.subplot(234)
plt.axis("off")
plt.imshow(img_open,cmap="gray")
plt.subplot(235)
plt.axis("off")
plt.imshow(img_dilate,cmap="gray")
plt.subplot(236)
plt.axis("off")
plt.imshow(img_close,cmap="gray")
plt.savefig('result1.jpg')
plt.show()
