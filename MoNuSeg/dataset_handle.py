import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

def read_cell_segmentation_tif_xml(root_path, mode='train'):
    """
    遍历 root_path/mode/images 下的 .tif 文件，
    并且配对同名的 .xml 标注文件
    """
    images = []
    labels = []
    image_root = os.path.join(root_path, mode, 'images')
    xml_root   = os.path.join(root_path, mode, 'annotations')
    for fname in os.listdir(image_root):
        base, ext = os.path.splitext(fname)
        if ext.lower() != '.tif':
            continue
        img_path = os.path.join(image_root, fname)
        xml_path = os.path.join(xml_root, base + '.xml')
        if not os.path.exists(xml_path):
            print(f'Warning: missing annotation {xml_path}')
            continue
        images.append(img_path)
        labels.append(xml_path)
    return images, labels

def parse_xml_to_mask(xml_path, img_size):
    """
    把 XML 标注中的所有 Region 多边形顶点解析出来，
    并在 h×w 大小的 mask 上填充成二值图
    """
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.uint8)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 假设每个 <Region> 下有多个 <Vertex X=".." Y=".."/>
    for region in root.findall('.//Region'):
        pts = []
        for v in region.findall('.//Vertex'):
            x = float(v.get('X'))
            y = float(v.get('Y'))
            # 如果 XML 中坐标与原始大图尺寸对应，需要自行缩放到 img_size
            # 这里假设坐标已与 img_size 对应
            pts.append([int(x), int(y)])
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], color=1)
    return mask

def cell_segmentation_loader_tif(img_path, xml_path, img_size=(448, 448)):
    """
    加载单个样本：
      1. 读取 .tif 图像，resize，并转为 RGB float32
      2. 解析 .xml 标注，生成二值 mask
      3. 返回 (img_tensor, mask_tensor)
    """
    # 1. 读图
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, img_size)
    # 单通道或多通道都要转成 RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 2. 生成 mask
    mask = parse_xml_to_mask(xml_path, img_size)
    # 3. 转 numpy → torch
    img = img.astype(np.float32).transpose(2, 0, 1) / 255.0   # [H,W,C]→[C,H,W]
    mask = mask.astype(np.float32)[None, ...]               # [H,W]→[1,H,W]
    return img, mask

class ImageFolder(data.Dataset):
    """
    PyTorch Dataset，用于加载 TIFF+XML 格式的分割数据集
    目录结构：
      root_path/
        train/
          images/       （.tif 文件）
          annotations/  （.xml 文件）
        val/
        test/
    """
    def __init__(self, root_path, mode='train'):
        super().__init__()
        self.images, self.labels = read_cell_segmentation_tif_xml(root_path, mode)

    def __getitem__(self, idx):
        img, mask = cell_segmentation_loader_tif(
            self.images[idx],
            self.labels[idx]
        )
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.images)
    
class FastDataset(Dataset):
    """
    从两个文件夹分别加载 images/ 和 masks/ 里的 .npy 文件
    """
    def __init__(self, root_preprocessed):
        """
        root_preprocessed: e.g. '/path/to/your/dataset/preprocessed_train'
        该路径下应有 'images/' 和 'masks/' 两个子文件夹
        """
        img_dir = os.path.join(root_preprocessed, 'images')
        msk_dir = os.path.join(root_preprocessed, 'masks')

        # 按文件名排序
        self.img_paths = sorted([
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir) if fname.endswith('.npy')
        ])
        self.msk_paths = sorted([
            os.path.join(msk_dir, fname)
            for fname in os.listdir(msk_dir) if fname.endswith('.npy')
        ])

        assert len(self.img_paths) == len(self.msk_paths), \
            "images 与 masks 数量不匹配"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])    # [3,H,W]
        mask = np.load(self.msk_paths[idx])   # [1,H,W]
        return torch.from_numpy(img), torch.from_numpy(mask)

if __name__ == '__main__':
    # 示例用法
    root = ''  
    ds = ImageFolder(root, mode='train')
    print(f'训练集样本数: {len(ds)}') #训练集样本数: 37
    img, mask = ds[0]
    print('img 形状:', img.shape, 'dtype:', img.dtype)   # torch.Size([3,448,448])
    print('mask 形状:', mask.shape, 'dtype:', mask.dtype) # torch.Size([1,448,448])
