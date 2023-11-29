import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取多张彩色图像
input_images = []
input_images.append(cv2.imread('./img/1.png', cv2.IMREAD_COLOR))
input_images.append(cv2.imread('./img/3.png', cv2.IMREAD_COLOR))
input_images.append(cv2.imread('./img/5.png', cv2.IMREAD_COLOR))
# 继续添加更多图像

# 定义增强后的拉普拉斯卷积核
laplacian_kernel = np.array([[0.1, -2, 0.1], [-2, 9, -2], [0.1, -2, 0.1]], dtype=np.float32)

# 创建一个新的Matplotlib Figure
fig, axes = plt.subplots(2, len(input_images), figsize=(12, 6))

for i, image in enumerate(input_images):
    sharpened_image = cv2.filter2D(image, -1, laplacian_kernel)

    # 显示原图像
    axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Original Image {i + 1}')
    axes[0, i].axis('off')

    # 显示增强后的图像
    axes[1, i].imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'Sharpened Image {i + 1}')
    axes[1, i].axis('off')

# 调整布局
plt.tight_layout()

# 显示Figure
plt.show()