from PIL import Image
import numpy as np

# 读取标签文件（假设路径正确）
mask_path = "D:/ltf/DRIVE/train/1st_manual/21_manual1.gif"
mask = Image.open(mask_path)
mask_array = np.array(mask)

# 打印唯一像素值
unique_values = np.unique(mask_array)
print("标签像素值:", unique_values)
print("类别数:", len(unique_values))  # 输出应为2（0和1）