from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from loss import dice_bce_loss
from cenet import CE_Net_
from framework import MyFrame
import sklearn.metrics as metrics

# Import the dataset handling function
from dataset_handle import ImageFolder

def read_cell_segmentation_cls6_64k(root_path, mode='train'):
    images = []
    labels = []  # Clearer variable name (masks are actually label paths)

    image_root = os.path.join(root_path, 'test/images')
    label_root = os.path.join(root_path, 'test/labels')

    for image_name in os.listdir(image_root):
        base_name, ext = os.path.splitext(image_name)
        
        if ext.lower() != '.jpg':
            continue

        image_path = os.path.join(image_root, image_name)
        label_name = f"{base_name}.txt"
        label_path = os.path.join(label_root, label_name)

        if not os.path.exists(label_path):
            print(f"Warning: Label file does not exist, skipping: {label_path}")
            continue

        images.append(image_path)
        labels.append(label_path)

    return images, labels

def parse_label_to_mask(label_path, img_size):
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.float32)
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]  
            coords = [int(coord * w) if i % 2 == 0 else int(coord * h) for i, coord in enumerate(coords)]
            points = np.array(coords).reshape(-1, 2)
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], color=1)
    return mask

def cell_segmentation_loader(img_path, label_path, img_size=(448, 448)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = parse_label_to_mask(label_path, img_size)
    
    img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    mask = mask.astype(np.float32).reshape(1, img_size[0], img_size[1])
    
    return img, mask

class ImageFolder(data.Dataset):
    def __init__(self, root_path, datasets='Cell_segmentation_cls6_64k', mode='test'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['Cell_segmentation_cls6_64k'], "Only 'Cell_segmentation_cls6_64k' dataset is supported"
        self.images, self.labels = read_cell_segmentation_cls6_64k(root_path, mode)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img, mask = cell_segmentation_loader(img_path, label_path)
        img = torch.from_numpy(img).type(torch.float32)
        mask = torch.from_numpy(mask).type(torch.float32)
        return img, mask
    
    def __len__(self):
        return len(self.images)

from sklearn import metrics

def calculate_auc_test(prediction, label):
    # 将预测和标签转化为1D数组
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    # 将标签转换为二分类：假设标签是0-255的范围，将其转为0和1
    label_1D = (label_1D > 0).astype(int)  # 将非零标签视为1，零视为0

    # 确保 result_1D 是概率值 (如果预测是0/1，则应该使用预测的概率)
    # 如果 prediction 是0/1，获取概率值
    if result_1D.max() > 1:
        # 如果预测值是概率值
        result_1D = result_1D / result_1D.max()  # 归一化为 0 到 1 之间

    # 计算 AUC
    auc = metrics.roc_auc_score(label_1D, result_1D)

    if np.isnan(auc):
        auc = 0

    return auc


def accuracy(pred_mask, label):
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    return acc, sen

def CE_Net_Test():
    solver = MyFrame(CE_Net_, dice_bce_loss, 1e-6)
    solver.load('D:/阿里云盘/新建文件夹/数据集/数据集/细胞癌细胞分割-cls6-6.4k/weights/200.th')
    solver.net.eval()

    dataset = ImageFolder('', mode='test')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=12)

    mylog = open('logs/' + 'test' + '.log', 'w')

    total_accuracy = []
    total_dice = []
    total_sen = []
    total_auc = []

    for img, mask in tqdm(data_loader):
        with torch.no_grad():
            solver.set_input(img, mask)
            pred, _ = solver.test_batch()
            pred = pred.squeeze()

        mask = mask.squeeze().cpu().numpy()

        acc, sen = accuracy(pred, mask)
        total_accuracy.append(acc)
        total_sen.append(sen)

        intersection = (pred * mask).sum()
        dice = (2. * intersection) / (pred.sum() + mask.sum()) if (pred.sum() + mask.sum()) != 0 else 0
        total_dice.append(dice)

        auc = calculate_auc_test(pred, mask)
        total_auc.append(auc)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy)
    avg_sen = sum(total_sen) / len(total_sen)
    avg_dice = sum(total_dice) / len(total_dice)
    avg_auc = sum(total_auc) / len(total_auc)

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Sensitivity: {avg_sen}")
    print(f"Average Dice: {avg_dice}")
    print(f"Average AUC: {avg_auc}")

    mylog.write(f"Average Accuracy: {avg_accuracy}\n")
    mylog.write(f"Average Sensitivity: {avg_sen}\n")
    mylog.write(f"Average Dice: {avg_dice}\n")
    mylog.write(f"Average AUC: {avg_auc}\n")
    mylog.close()

if __name__ == '__main__':
    CE_Net_Test()
