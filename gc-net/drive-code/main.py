import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset_handle import SegmentationDataset
from model.final_model import CombinedModel
from loss import MultiLabelBCEDiceLoss
import Constants
import os

# 配置日志
# 完全移除日志格式中的占位符，仅使用固定格式
def setup_logger():
    log_dir = Path("./log")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "train.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # 标准格式，不含自定义字段
        datefmt="%Y-%m-%d %H:%M:%S"
    )



# 计算四个评价指标（多标签版本）
def calculate_metrics(pred, mask, threshold=0.5):
    """
    pred: [B, 6, H, W] 概率值（0-1）
    mask: [B, 6, H, W] 独热编码标签（0或1）
    """
    pred = pred.detach().cpu().numpy()  # 转为numpy
    mask = mask.cpu().numpy()
    num_classes = pred.shape[1]
    metrics = {
        'accuracy': np.zeros(num_classes),
        'sensitivity': np.zeros(num_classes),
        'dice': np.zeros(num_classes),
        'auc': np.zeros(num_classes),
        'count': np.zeros(num_classes, dtype=int)
    }
    
    for b in range(pred.shape[0]):  # 遍历批次
        for c in range(num_classes):  # 遍历每个类别
            p = pred[b, c].ravel()  # 预测概率
            m = mask[b, c].ravel()  # 真实标签
            
            # 跳过全0或全1的无效样本（避免AUC计算错误）
            if np.unique(m).size < 2:
                continue
            
            p_bin = (p > threshold).astype(np.float32)  # 二值化预测
            tp = np.sum(p_bin * m)
            fp = np.sum(p_bin * (1 - m))
            fn = np.sum((1 - p_bin) * m)
            tn = np.sum((1 - p_bin) * (1 - m))
            
            # 计算各指标
            metrics['dice'][c] += (2 * tp) / (2 * tp + fp + fn + 1e-8)
            metrics['accuracy'][c] += (tp + tn) / (tp + tn + fp + fn + 1e-8)
            metrics['sensitivity'][c] += tp / (tp + fn + 1e-8) if (tp + fn) != 0 else 0.0
            metrics['auc'][c] += roc_auc_score(m, p)
            metrics['count'][c] += 1
    
    # 计算类别平均（忽略无效类别）
    avg_metrics = {}
    for key in ['accuracy', 'sensitivity', 'dice', 'auc']:
        valid_mask = metrics['count'] > 0
        if np.any(valid_mask):
            # 按类别平均（每个类别在所有样本上的平均指标）
            class_avg = np.mean(metrics[key][valid_mask] / metrics['count'][valid_mask])
            avg_metrics[key] = np.clip(class_avg, 0, 1)  # 确保值在0~1之间
        else:
            avg_metrics[key] = 0.0
    return avg_metrics

def train():
    setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    model = CombinedModel().to(device)
    loss_fn = MultiLabelBCEDiceLoss(alpha=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    train_dataset = SegmentationDataset(root_path=Constants.DATA_ROOT, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=Constants.BATCHSIZE_PER_CARD,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    
    os.makedirs(Constants.WEIGHTS_DIR, exist_ok=True)
    best_dice = -1.0
    
    for epoch in range(1, Constants.TOTAL_EPOCH + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'dice': 0.0,
            'auc': 0.0
        }
        batch_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}/{Constants.TOTAL_EPOCH}") as pbar:
            for img, mask in pbar:
                img = img.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                pred = model(img)  # 输出: [B, 6, 512, 512]
                loss = loss_fn(pred, mask)
                loss.backward()
                optimizer.step()
                
                # 计算批次指标
                batch_metrics = calculate_metrics(pred, mask)
                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
                batch_count += 1
                
                epoch_loss += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{batch_metrics['dice']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # 计算 epoch 平均指标
        if batch_count == 0:
            continue
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        # 记录日志
        log_msg = (
            f"Loss: {epoch_loss / batch_count:.4f} | "
            f"Average Accuracy: {epoch_metrics['accuracy']:.4f} | "
            f"Average Sensitivity: {epoch_metrics['sensitivity']:.4f} | "
            f"Average Dice: {epoch_metrics['dice']:.4f} | "
            f"Average AUC: {epoch_metrics['auc']:.4f}"
        )
        logging.info(log_msg)  # 直接记录完整消息，无需占位符
        print(log_msg)
        
        # 保存最佳模型（基于Dice）
        if epoch_metrics['dice'] > best_dice:
            best_dice = epoch_metrics['dice']
            torch.save(model.state_dict(), os.path.join(Constants.WEIGHTS_DIR, "best_model.pth"))
            logging.info(f"保存最佳模型，Dice: {best_dice:.4f}")
        
        scheduler.step()

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    train()