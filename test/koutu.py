import os
import paddlehub as hub
import cv2
import numpy as np

# 加载人像分割模型
human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")

# 定义输入文件夹和输出文件夹
input_folder = "koutubase"
output_folder = "koutuoutput"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 构造输入文件路径和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".", "_output."))

        # 读取图像
        image = cv2.imread(input_path)

        # 检查图像是否成功加载
        if image is None:
            print(f"无法加载图像: {input_path}")
            continue

        # 进行人像分割
        result = human_seg.segmentation(images=[image])
        segmentation_result = result[0]['data']

        # 将背景颜色更改为(74, 141, 206)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if segmentation_result[i, j] == 0:
                    image[i, j] = [206, 141, 74]

        # 保存处理后的结果
        cv2.imwrite(output_path, image)

print("处理完成！")
