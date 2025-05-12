import os
import numpy as np
from tqdm import tqdm

# 复用你已有的 reader/loader
from dataset_handle import read_cell_segmentation_tif_xml, cell_segmentation_loader_tif

# 原始数据根目录
raw_root = ''
img_size = (448, 448)
modes = ['train', 'test']  # 只有 train 和 test

for mode in modes:
    image_paths, xml_paths = read_cell_segmentation_tif_xml(raw_root, mode)
    base_pre = os.path.join(raw_root, f'preprocessed_{mode}')
    img_dir  = os.path.join(base_pre, 'images')
    msk_dir  = os.path.join(base_pre, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    print(f"[{mode}] 共 {len(image_paths)} 张，开始预处理 → {base_pre}")
    for idx, (img_p, xml_p) in enumerate(tqdm(zip(image_paths, xml_paths), total=len(image_paths))):
        img, mask = cell_segmentation_loader_tif(img_p, xml_p, img_size=img_size)
        np.save(os.path.join(img_dir,  f'image_{idx:05d}.npy'), img)
        np.save(os.path.join(msk_dir,  f'mask_{idx:05d}.npy'), mask)
    print(f"[{mode}] 预处理完成\n")
