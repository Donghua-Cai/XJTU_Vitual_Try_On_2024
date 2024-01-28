import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from data.canny import cannyEdgeDetector
import cv2
import numpy as np
from tqdm import tqdm  # 引入 tqdm

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def load_data(dir_name):
    imgs = []
    for filename in tqdm(os.listdir(dir_name), desc="加载图像"):
        if os.path.isfile(os.path.join(dir_name, filename)):
            img = mpimg.imread(os.path.join(dir_name, filename))
            img = rgb2gray(img)
            imgs.append(img)
    return imgs

# 显示图像
def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(10, 20))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        plt_idx = i + 1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()



imgs = load_data(dir_name='VITON_test/VITON_test/test_clothes')
# visualize(imgs, 'gray')
detector = cannyEdgeDetector(imgs, sigma=1, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=10)
imgs_final = detector.detect()
# visualize(imgs_final, 'gray')

# 创建保存图像的新文件夹
output_dir = 'canny_output_images'
os.makedirs(output_dir, exist_ok=True)

# 保存图像
for i, img in enumerate(imgs_final):
    # Extracting the filename from the original image path
    filename = os.path.basename(f"VITON_test/VITON_test/test_clothes/{i}.jpg")

    # Removing the file extension
    filename = os.path.splitext(filename)[0]

    # Creating the output filename
    output_filename = f"{output_dir}/result_{filename}.png"

    cv2.imwrite(output_filename, img)
