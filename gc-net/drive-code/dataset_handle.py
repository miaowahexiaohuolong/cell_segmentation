import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
from PIL import Image  # 依赖Pillow库处理PGM格式

# -------------------- 数据集加载部分（与之前相同） --------------------
def read_dataset(root_path, mode='train'):
    """读取图像和对应PGM标签路径"""
    images = []
    labels = []
    mode_dir = 'train' if mode.lower() in ['train', 'training'] else 'test'
    image_dir = os.path.join(root_path, mode_dir, 'images')
    label_dir = os.path.join(root_path, mode_dir, '1st_manual')
    
    for img_name in os.listdir(image_dir):
        base_name, ext = os.path.splitext(img_name)
        if ext.lower() not in ['.tif', '.tiff']:
            continue
        label_name = f"{base_name.split('_')[0]}_manual1.gif"  # DRIVE标签扩展名特殊
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        if not os.path.exists(label_path):
            print(f"警告：标签缺失，跳过 {img_path}")
            continue
        images.append(img_path)
        labels.append(label_path)
    return images, labels

def pgm_mask_to_onehot(mask_path, img_size, num_classes=2):
    """将PGM掩码转换为独热编码"""
    with Image.open(mask_path) as img:
        mask = np.array(img.convert("L"), dtype=np.uint8)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    onehot = np.zeros((num_classes, *img_size), dtype=np.float32)
    onehot[0] = (mask == 0).astype(np.float32)    # 背景
    onehot[1] = (mask == 255).astype(np.float32)  # 血管
    return onehot

def data_loader(img_path, label_path, img_size=(512, 512)):
    """加载并预处理图像和掩码"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = img.astype(np.float32)  # 新增：显式转换为float32
    img = img.transpose(2, 0, 1)  # (C, H, W)
    
    mask = pgm_mask_to_onehot(label_path, img_size)
    return img, mask

class SegmentationDataset(data.Dataset):
    def __init__(self, root_path, mode='train', img_size=(512, 512)):
        self.images, self.labels = read_dataset(root_path, mode)
        self.img_size = img_size
        if not self.images:
            raise RuntimeError(f"未找到{mode}数据")
    
    def __getitem__(self, idx):
        img, mask = data_loader(self.images[idx], self.labels[idx], self.img_size)
        return torch.tensor(img), torch.tensor(mask)
    
    def __len__(self):
        return len(self.images)

# -------------------- 新增Batch处理部分 --------------------
if __name__ == "__main__":
    root_path = "D:/ltf/DRIVE"
    
    # 1. 创建数据集实例
    train_dataset = SegmentationDataset(root_path, mode='train')
    test_dataset = SegmentationDataset(root_path, mode='test')
    
    # 2. 配置DataLoader（关键步骤）
    batch_size = 4  # 可根据显卡内存调整（建议8/16/32）
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 训练时打乱数据顺序
        num_workers=4,      # 数据加载线程数（Windows建议设为0或2）
        pin_memory=True     # 加速GPU数据传输（仅当使用GPU时有效）
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,      # 测试时不打乱
        num_workers=4
    )
    
    # 3. 验证Batch加载效果
    print(f"训练集Batch数量：{len(train_loader)}")
    print(f"测试集Batch数量：{len(test_loader)}")
    
    # 4. 遍历Batch数据（示例）
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx+1} 图像形状：{imgs.shape}")    # (Batch, C, H, W)
        print(f"Batch {batch_idx+1} 掩码形状：{masks.shape}")  # (Batch, Classes, H, W)
        if batch_idx == 0:  # 仅打印第一个Batch
            break