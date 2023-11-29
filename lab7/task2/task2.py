import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负

img = cv2.imread(r'./mapleleaf.tif',0)
_, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(figsize=(12,6))
plt.subplot(141)
plt.axis("off")
plt.imshow(img_binary,cmap="gray")
i=1
for kernelSize in [3,9,15]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
    img_result = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
    i+=1
    plt.subplot(1,4,i)
    plt.axis("off")
    plt.imshow(img_result,cmap="gray")
plt.savefig('result2.jpg')
plt.show()