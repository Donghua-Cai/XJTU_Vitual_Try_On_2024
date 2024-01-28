import cv2
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from numpy import unicode

if __name__ == '__main__':
    img1 = cv2.imread(r"koutuoutput\111.png")  # 读取彩色图像(BGR)
    img2 = cv2.imread("backWhite.png")  # 读取 CV Logo

    # 我想把logo放在左上角，所以我创建了ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # 现在创建logo的掩码，并同时创建其相反掩码
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # 现在将ROI中logo的区域涂黑
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # 仅从logo图像中提取logo区域
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # 将logo放入ROI并修改主图像
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow("imgAdd", img1)  # 显示叠加图像 imgAdd

    cv2.waitKey(0)
    cv2.destroyAllWindows()