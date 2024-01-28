import os
from PIL import Image

# 设定待转换图片存放的路径
path = r'E:\Virtual_Try_On\code\Flow-Style-VTON-main\test\cloudtry\test_img'

# 遍历指定文件夹内所有的 PNG 图片
for file_name in os.listdir(path):
    if file_name.endswith('.png'):
        # 读取 PNG 图片
        png_image = Image.open(os.path.join(path, file_name))

        # 将 PNG 图片转换为 JPG 格式
        jpg_image = png_image.convert('RGB')

        # 修改文件后缀名为 JPG
        new_file_name = os.path.splitext(file_name)[0] + '.jpg'

        # 保存转换后的 JPG 图片
        jpg_image.save(os.path.join(path, new_file_name), 'JPEG')

        # 删除原始的 PNG 图片
        os.remove(os.path.join(path, file_name))