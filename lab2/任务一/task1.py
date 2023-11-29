import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# 读取图像
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

# 对比度增强
alpha = 1.5  # 可以调整的参数，控制对比度增强程度
new_img1 = np.clip(alpha * img1, 0, 255).astype(np.uint8)
new_img2 = np.clip(alpha * img2, 0, 255).astype(np.uint8)

# 保存图像
cv2.imwrite('output1.png', new_img1)
cv2.imwrite('output2.png', new_img2)

# 显示原始图像和增强后的图像
cv2_imshow(img1)
cv2_imshow(new_img1)

cv2_imshow(img2)
cv2_imshow(new_img2)