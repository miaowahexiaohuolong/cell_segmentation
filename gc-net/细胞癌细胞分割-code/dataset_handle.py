import torch
import torch.utils.data as data
import cv2
import numpy as np
import os

def read_dataset(root_path, mode='train'):
    images = []
    labels = []
    image_dir = os.path.join(root_path, mode, 'images')
    label_dir = os.path.join(root_path, mode, 'labels')
    
    for img_name in os.listdir(image_dir):
        base_name, ext = os.path.splitext(img_name)
        if ext.lower() not in ['.jpg', '.png']:
            continue
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        if not os.path.exists(label_path):
            print(f"警告：标签缺失，跳过 {img_path}")
            continue
        images.append(img_path)
        labels.append(label_path)
    return images, labels

def parse_label_to_onehot(label_path, img_size, num_classes=6):
    h, w = img_size
    mask = np.zeros((num_classes, h, w), dtype=np.float32)  # 6通道独热掩码
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue
            class_id = int(parts[0])  # 类别ID（0-5）
            coords = parts[1:]
            coords_px = [int(coord * w) if i % 2 == 0 else int(coord * h) for i, coord in enumerate(coords)]
            points = np.array(coords_px).reshape(-1, 2)
            if 0 <= class_id < num_classes:
                cv2.fillPoly(mask[class_id], [points.reshape(-1, 1, 2)], color=1)  # 填充对应通道为1
    return mask

def cell_segmentation_loader(img_path, label_path, img_size=(512, 512)):
    # 读取图像（支持中文路径）
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    
    # 预处理图像
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # 转为 (C, H, W)
    
    # 解析标签为6通道独热掩码
    mask = parse_label_to_onehot(label_path, img_size)
    
    return img, mask

class ImageFolder(data.Dataset):
    def __init__(self, root_path, mode='train'):
        self.root = root_path
        self.mode = mode
        self.images, self.labels = read_dataset(root_path, mode)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img, mask = cell_segmentation_loader(img_path, label_path)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    
    def __len__(self):
        return len(self.images)

# 测试加载（可选）
if __name__ == "__main__":
    dataset = ImageFolder(root_path="D:/ltf/细胞癌细胞分割-cls6-6.4k", mode='train')
    img, mask = dataset[0]
    print(f"图像形状: {img.shape}")       # torch.Size([3, 512, 512])
    print(f"标签形状: {mask.shape}")     # torch.Size([6, 512, 512])