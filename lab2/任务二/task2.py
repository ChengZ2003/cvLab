import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# 读取图像
img = cv2.imread('kennysmall.jpg', 0)

# 定义不同的滤波算子
kernel_smooth = np.ones((3, 3), np.float32) / 9
kernel_sharpen = np.array([[1, 1, 1, 1, 2, 1, 1, 1, 1]], np.float32) / 10
kernel_edge_enhance = np.array([[-1, -1, -1, -1, 9, -1, -1, -1, -1]], np.float32)

# 应用滤波算子
smoothed_img = cv2.filter2D(img, -1, kernel_smooth)
sharpened_img = cv2.filter2D(img, -1, kernel_sharpen)
edge_enhanced_img = cv2.filter2D(img, -1, kernel_edge_enhance)

# 保存处理后的图像
cv2.imwrite('平滑后的kennysmall.jpg', smoothed_img)
cv2.imwrite('锐化后的kennysmall.jpg', sharpened_img)
cv2.imwrite('边缘增强后的kennysmall.jpg', edge_enhanced_img)

# 显示原始图像和处理后的图像
cv2_imshow(img)
cv2_imshow(smoothed_img)
cv2_imshow(sharpened_img)
cv2_imshow(edge_enhanced_img)