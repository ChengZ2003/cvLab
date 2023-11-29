import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread('./kennysmall.jpg', 0)

# 添加椒盐噪声
def add_salt_pepper_noise(image, amount=0.001):
    row, col = image.shape
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 255
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0
    return noisy_image

noisy_image_salt_pepper = add_salt_pepper_noise(img, 0.01)

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, variance=0.01):
    row, col = image.shape
    sigma = variance ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gaussian
    return noisy_image

noisy_image_gaussian = add_gaussian_noise(img)

# 应用均值滤波、中值滤波和高斯滤波
mean_filtered_salt_pepper = cv2.blur(noisy_image_salt_pepper, (3, 3))
median_filtered_salt_pepper = cv2.medianBlur(noisy_image_salt_pepper.astype(np.uint8), 3)
gaussian_filtered_salt_pepper = cv2.GaussianBlur(noisy_image_salt_pepper, (3, 3), 0)

mean_filtered_gaussian = cv2.blur(noisy_image_gaussian, (3, 3))
median_filtered_gaussian = cv2.medianBlur(noisy_image_gaussian.astype(np.uint8), 3)
gaussian_filtered_gaussian = cv2.GaussianBlur(noisy_image_gaussian, (3, 3), 0)

# 保存图像
cv2.imwrite('./task3/椒盐噪声图像.jpg', noisy_image_salt_pepper)
cv2.imwrite('./task3/高斯噪声图像.jpg', noisy_image_gaussian)
cv2.imwrite('./task3/均值滤波_椒盐噪声.jpg', mean_filtered_salt_pepper)
cv2.imwrite('./task3/中值滤波_椒盐噪声.jpg', median_filtered_salt_pepper)
cv2.imwrite('./task3/高斯滤波_椒盐噪声.jpg', gaussian_filtered_salt_pepper)
cv2.imwrite('./task3/均值滤波_高斯噪声.jpg', mean_filtered_gaussian)
cv2.imwrite('./task3/中值滤波_高斯噪声.jpg', median_filtered_gaussian)
cv2.imwrite('./task3/高斯滤波_高斯噪声.jpg', gaussian_filtered_gaussian)

# 显示图像及滤波后的图像
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(232), plt.imshow(noisy_image_salt_pepper, 'gray'), plt.title('Salt & Pepper Noise')
plt.subplot(233), plt.imshow(noisy_image_gaussian, 'gray'), plt.title('Gaussian Noise')
plt.subplot(234), plt.imshow(mean_filtered_salt_pepper, 'gray'), plt.title('Mean Filtered (Salt & Pepper)')
plt.subplot(235), plt.imshow(median_filtered_salt_pepper, 'gray'), plt.title('Median Filtered (Salt & Pepper)')
plt.subplot(236), plt.imshow(gaussian_filtered_salt_pepper, 'gray'), plt.title('Gaussian Filtered (Salt & Pepper)')

plt.show()

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(232), plt.imshow(noisy_image_salt_pepper, 'gray'), plt.title('Salt & Pepper Noise')
plt.subplot(233), plt.imshow(noisy_image_gaussian, 'gray'), plt.title('Gaussian Noise')
plt.subplot(234), plt.imshow(mean_filtered_gaussian, 'gray'), plt.title('Mean Filtered (Gaussian)')
plt.subplot(235), plt.imshow(median_filtered_gaussian, 'gray'), plt.title('Median Filtered (Gaussian)')
plt.subplot(236), plt.imshow(gaussian_filtered_gaussian, 'gray'), plt.title('Gaussian Filtered (Gaussian)')

plt.show()