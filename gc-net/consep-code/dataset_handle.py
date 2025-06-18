import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio

def read_consep_pairs(root_path, mode='Train'):
    """
    遍历 root_path/mode/Images 下的图像文件，配对同名的.mat标注文件（位于mode/Labels）
    """
    image_list = []
    mat_list = []
    
    # 图像和标注的目录路径（适配你的目录结构）
    image_dir = os.path.join(root_path, mode, 'Images')  # 例如：root_path/Train/Images
    label_dir = os.path.join(root_path, mode, 'Labels')  # 例如：root_path/Train/Labels
    
    # 遍历图像文件
    for fname in os.listdir(image_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in ('.tif', '.png', '.jpg'):  # 支持常见图像格式
            continue
        
        img_path = os.path.join(image_dir, fname)
        mat_path = os.path.join(label_dir, f"{base}.mat")  # 标注文件与图像同名（.mat）
        
        if not os.path.exists(mat_path):
            print(f'Warning: 缺少标注文件 {mat_path}')
            continue
        
        image_list.append(img_path)
        mat_list.append(mat_path)
    
    return image_list, mat_list

def load_consep_mat(mat_path, img_size=(512, 512)):
    """
    加载.mat文件并生成实例/类别掩码（调整尺寸后）
    """
    try:
        mat_data = sio.loadmat(mat_path)
        h, w = img_size
        
        # 读取原始掩码（1000x1000）
        inst_map = mat_data['inst_map'].astype(np.int32)  # 实例掩码（不同数值代表不同细胞核）
        type_map = mat_data['type_map'].astype(np.int32)  # 类别掩码（数值代表细胞类型）
        
        # 调整掩码尺寸到目标大小（使用最近邻插值保持离散值准确性）
        inst_mask = cv2.resize(inst_map, img_size, interpolation=cv2.INTER_NEAREST)
        type_mask = cv2.resize(type_map, img_size, interpolation=cv2.INTER_NEAREST)
        
        return inst_mask, type_mask
    
    except Exception as e:
        raise ValueError(f"加载.mat文件失败: {str(e)}")

def consep_loader(img_path, mat_path, img_size=(512, 512)):
    """
    加载单个样本：图像 + 实例掩码 + 类别掩码
    """
    # 1. 读取并预处理图像
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, img_size)
    if img.ndim == 2:  # 单通道转RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:  # BGR转RGB（OpenCV默认读取为BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32).transpose(2, 0, 1) / 255.0  # [H,W,C]→[C,H,W]
    
    # 2. 加载并预处理掩码
    inst_mask, type_mask = load_consep_mat(mat_path, img_size)
    inst_mask = inst_mask.astype(np.float32)[None, ...]  # [H,W]→[1,H,W]
    type_mask = type_mask.astype(np.float32)[None, ...]  # [H,W]→[1,H,W]
    
    return img, inst_mask, type_mask

class CoNSePDataset(data.Dataset):
    """
    PyTorch Dataset，适配目录结构：
      root_path/
        Train/
          Images/   → 图像文件（.tif/.png等）
          Labels/   → 同名的.mat标注文件
        Test/
          Images/
          Labels/
    """
    def __init__(self, root_path, mode='Train', img_size=(448, 448)):
        super().__init__()
        self.img_size = img_size
        # 注意：mode参数应与目录名一致（如'Train'或'Test'）
        self.images, self.labels = read_consep_pairs(root_path, mode)

    def __getitem__(self, idx):
        img, inst_mask, type_mask = consep_loader(
            self.images[idx],
            self.labels[idx],
            self.img_size
        )
        return (
            torch.from_numpy(img),          # 图像Tensor [3,H,W]
            torch.from_numpy(inst_mask),    # 实例掩码Tensor [1,H,W]
            torch.from_numpy(type_mask)     # 类别掩码Tensor [1,H,W]
        )

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    # 示例用法（根据你的实际路径修改）
    DATA_ROOT = ""  # 例如："/data/CoNSeP"
    
    # 加载训练集（对应Train目录）
    train_ds = CoNSePDataset(DATA_ROOT, mode='Train', img_size=(448, 448))
    print(f"训练集样本数: {len(train_ds)}")  # 输出实际样本数量
    
    # 加载测试集（对应Test目录）
    test_ds = CoNSePDataset(DATA_ROOT, mode='Test', img_size=(448, 448))
    print(f"测试集样本数: {len(test_ds)}")
    
    # 验证数据加载（以训练集第一个样本为例）
    sample_img, sample_inst, sample_type = train_ds[0]
    print(f"图像形状: {sample_img.shape}  类型: {sample_img.dtype}")      # torch.Size([3, 448, 448])  float32
    print(f"实例掩码形状: {sample_inst.shape}  类型: {sample_inst.dtype}")  # torch.Size([1, 448, 448])  float32
    print(f"类别掩码形状: {sample_type.shape}  类型: {sample_type.dtype}")  # torch.Size([1, 448, 448])  float32

    