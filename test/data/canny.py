from scipy import ndimage
from scipy.ndimage.filters import convolve
import numpy as np
import cv2
from tqdm import tqdm  # 引入 tqdm

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05,
                 highthreshold=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    def sobel_filters(self, img):
        # 定义Sobel滤波器的卷积核
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # 对图像应用Sobel滤波器，分别计算水平和垂直方向的梯度
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        # 计算梯度幅值
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255

        # 计算梯度方向
        theta = np.arctan2(Iy, Ix)

        # 返回梯度幅值和梯度方向
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):
        # 计算高阈值和低阈值
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        # 初始化一个与输入图像相同大小的数组，用于存储阈值化后的结果
        res = np.zeros((M, N), dtype=np.int32)

        # 定义弱像素和强像素的整数值
        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        # 找到高于高阈值的像素的坐标
        strong_i, strong_j = np.where(img >= highThreshold)

        # 找到低于低阈值的像素的坐标
        zeros_i, zeros_j = np.where(img < lowThreshold)

        # 找到介于高阈值和低阈值之间的像素的坐标
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        # 将强像素和弱像素赋值到结果数组中
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        # 返回阈值化后的结果数组
        return res

    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel  # 弱像素阈值
        strong = self.strong_pixel  # 强像素阈值

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == weak):
                    try:
                        # 检查周围8个像素中是否存在强像素，如果存在，则将当前像素标记为强像素，否则置为0
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img

    def fill_contours(self, img):
        # 确保图像的数据类型是 uint8
        img = img.astype(np.uint8)

        # 对图像进行膨胀以连接边缘
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=2)

        # 寻找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白图像作为填充结果
        fill_img = np.zeros_like(img)

        # 绘制轮廓并进行填充
        cv2.drawContours(fill_img, contours, -1, 255, thickness=cv2.FILLED)

        return fill_img

    def detect(self):
        imgs_final = []
        for i, img in tqdm(enumerate(self.imgs), total=len(self.imgs), desc="Canny 边缘检测进度"):
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            img_final_filled = self.fill_contours(img_final)
            imgs_final.append(img_final_filled)

        return imgs_final


