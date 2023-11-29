import cv2
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负

img0 = cv2.imread(r'alphabet.jpg')
imgT = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
cv2.rectangle(img0, (160, 140), (190, 170), (255, 0, 0), 3)  # openCV是图像坐标

img = imgT[140:170, 160:190]
img1 = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
img2 = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
img3 = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
img4 = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_AREA)
plt.figure(figsize=(10, 6))
plt.subplot(141), plt.imshow(img0, cmap="viridis")
plt.title('原图像'), plt.axis('off')
plt.subplot(142), plt.imshow(img1, cmap='gray')
plt.title('最近邻插值'), plt.axis('off')
plt.subplot(143), plt.imshow(img2, cmap='gray')
plt.title('双线性插值'), plt.axis('off')
plt.subplot(144), plt.imshow(img3, cmap='gray')
plt.title('双三次插值'), plt.axis('off')
plt.savefig('result2.jpg')
plt.show()
