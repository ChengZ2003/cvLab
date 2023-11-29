import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负
img0 = cv2.imread(r'./light_circle.jpg',0)
_,img_seg=cv2.threshold(img0,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernalSize=19
img_seg_adapt = cv2.adaptiveThreshold(img0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernalSize, 6)
plt.subplot(121)
plt.title('待处理灰度图像')
plt.imshow(img0,cmap="gray")
plt.axis("off")
plt.subplot(122)
plt.title('自适应阈值分割结果')
plt.imshow(img_seg_adapt,cmap="gray")
plt.axis("off")
plt.savefig('result5.jpg')
plt.show()