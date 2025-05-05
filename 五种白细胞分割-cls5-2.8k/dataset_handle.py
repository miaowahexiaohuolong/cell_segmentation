import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc
from tqdm import tqdm

#train: ../train/images
#val: ../valid/images
#test: ../test/images

#nc: 6
#names: ['apoptosis', 'lumen', 'mitosis', 'nolumen', 'notumor', 'tumor']

def read_cell_segmentation_cls6_64k(root_path, mode='train'):
    images = []
    labels = []  # 更清晰的变量名（原 masks 实际是标签路径）

    # 定义图像和标签的根目录（根据 mode 调整，示例为 'train'）
    image_root = os.path.join(root_path, 'train/images')
    label_root = os.path.join(root_path, 'train/labels')
    #print(f"图像目录: {image_root}")
    #print(f"标签目录: {label_root}")

    # 遍历图像目录下的所有文件
    for image_name in os.listdir(image_root):
        # 分离图像文件名的主名和扩展名（处理 .jpg 格式）
        base_name, ext = os.path.splitext(image_name)  # 例如: base_name='100_png_jpg...', ext='.jpg'
        
        # 只处理 .jpg 图像（根据实际调整扩展名）
        if ext.lower() != '.jpg':
            continue

        # 图像路径：直接使用原文件名（含 .jpg）
        image_path = os.path.join(image_root, image_name)
        
        # 标签路径：主名 + .txt 扩展名（关键修复！）
        label_name = f"{base_name}.txt"  # 正确标签名：100_png_jpg... .txt
        label_path = os.path.join(label_root, label_name)

        # 检查标签文件是否存在（避免无效路径）
        if not os.path.exists(label_path):
            print(f"警告：标签文件不存在，跳过: {label_path}")
            continue

        images.append(image_path)
        labels.append(label_path)


    return images, labels

def parse_label_to_mask(label_path, img_size):
    """将标注文件中的多边形坐标转换为单通道二进制掩码（目标区域为 1，背景为 0）"""
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.float32)  # 初始化全0掩码
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]  # 忽略类别ID（假设二分类，所有目标统一为 1）
            # 归一化坐标转像素坐标（假设坐标是 0~1 范围）
            coords = [int(coord * w) if i % 2 == 0 else int(coord * h) for i, coord in enumerate(coords)]
            points = np.array(coords).reshape(-1, 2)  # 转换为 (n, 2) 的顶点坐标
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], color=1)  # 填充目标区域为 1
    return mask

def cell_segmentation_loader(img_path, label_path, img_size=(448, 448)):
    # 1. 加载图像并Resize
    img = cv2.imread(img_path)                          # BGR格式
    img = cv2.resize(img, img_size)                     # 调整大小为 448x448
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # 转为 RGB（避免通道顺序问题）
    
    # 2. 解析标注生成掩码（假设标注为多边形坐标，归一化到 0~1）
    mask = parse_label_to_mask(label_path, img_size)     # 生成单通道掩码（0/1）
    
    # 4. 转换为 PyTorch 所需格式（C, H, W）并归一化
    img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # [H, W, C] → [C, H, W], 归一化到 [0, 1]
    mask = mask.astype(np.float32).reshape(1, img_size[0], img_size[1])  # [H, W] → [1, H, W]
    
    return img, mask

class ImageFolder(data.Dataset):
    def __init__(self, root_path, datasets='Cell_segmentation_cls6_64k', mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        # 支持你的数据集
        assert self.dataset in ['Cell_segmentation_cls6_64k'], "仅支持 Cell_segmentation_cls6_64k 数据集"
        self.images, self.labels = read_cell_segmentation_cls6_64k(root_path, mode)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img, mask = cell_segmentation_loader(img_path, label_path)  # 调用专属加载器
        # 转换为 Tensor（确保数据类型为 float32）
        img = torch.from_numpy(img).type(torch.float32)
        mask = torch.from_numpy(mask).type(torch.float32)
        return img, mask
    
    def __len__(self):
        return len(self.images)
    

if __name__ == "__main__":
    # 初始化数据集（假设你的数据集路径为 '/path/to/your/dataset'）
    dataset = ImageFolder(
        root_path='',
        mode='train'
    )
    
    # 获取第一个样本
    img, mask = dataset[0]
    
#    # 打印信息（符合你的预期输出）
    print("img 形状:", img.shape)       # torch.Size([3, 448, 448])
    print("mask 形状:", mask.shape)     # torch.Size([1, 448, 448])
    print("img 数据类型:", img.dtype)   # torch.float32
    print("mask 数据类型:", mask.dtype) # torch.float32
# img 形状: torch.Size([3, 448, 448])
# mask 形状: torch.Size([1, 448, 448])
# img 数据类型: torch.float32
#mask 数据类型: torch.float32
