import torch
import numpy as np
import os
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset_handle import ImageFolder  # 使用当前项目的 ImageFolder 数据集
from model.final_model import CombinedModel
import Constants

# 配置测试日志（解决中文乱码）
def setup_test_logger():
    log_dir = Path("./log")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "test.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8"
    )

# 复用训练中的指标计算函数（保持逻辑一致）
def calculate_metrics(pred, mask, threshold=0.3):
    pred = pred.detach().cpu().numpy()
    mask = mask.cpu().numpy()
    num_classes = pred.shape[1]
    metrics = {
        'accuracy': np.zeros(num_classes),
        'sensitivity': np.zeros(num_classes),
        'dice': np.zeros(num_classes),
        'auc': np.zeros(num_classes),
        'count': np.zeros(num_classes, dtype=int)
    }
    
    for b in range(pred.shape[0]):
        for c in range(num_classes):
            p = pred[b, c].ravel()
            m = mask[b, c].ravel()
            
            if np.unique(m).size < 2:
                continue
            
            p_bin = (p > threshold).astype(np.float32)
            tp = np.sum(p_bin * m)
            fp = np.sum(p_bin * (1 - m))
            fn = np.sum((1 - p_bin) * m)
            tn = np.sum((1 - p_bin) * (1 - m))
            
            metrics['dice'][c] += (2 * tp) / (2 * tp + fp + fn + 1e-8)
            metrics['accuracy'][c] += (tp + tn) / (tp + tn + fp + fn + 1e-8)
            metrics['sensitivity'][c] += tp / (tp + fn + 1e-8) if (tp + fn) != 0 else 0.0
            metrics['auc'][c] += roc_auc_score(m, p)
            metrics['count'][c] += 1
    
    avg_metrics = {}
    for key in ['accuracy', 'sensitivity', 'dice', 'auc']:
        valid_mask = metrics['count'] > 0
        if np.any(valid_mask):
            class_avg = np.mean(metrics[key][valid_mask] / metrics['count'][valid_mask])
            avg_metrics[key] = np.clip(class_avg, 0, 1)
        else:
            avg_metrics[key] = 0.0
    return avg_metrics

def test():
    setup_test_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"测试设备: {device}")
    
    # 加载模型（与训练一致）
    model = CombinedModel().to(device)
    model_path = os.path.join(Constants.WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先完成训练！")
    model.load_state_dict(torch.load(model_path, map_location=device))

    first_conv_weights = next(model.parameters()).data
    if torch.allclose(first_conv_weights, torch.tensor(0.0)):
        raise ValueError("模型参数全为0，未成功加载训练权重！")
    logging.info(f"成功加载模型: {model_path}")
    
    # 加载测试数据集（使用 ImageFolder，返回 (img, mask)）
    test_dataset = ImageFolder(root_path=Constants.DATA_ROOT, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=Constants.BATCHSIZE_PER_CARD,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    
    model.eval()
    total_metrics = {'accuracy': 0.0, 'sensitivity': 0.0, 'dice': 0.0, 'auc': 0.0}
    total_batches = 0
    
    # 关键修改：数据加载器返回 2 个值，解包为 (img, mask)
    with torch.no_grad(), tqdm(test_loader, desc="测试进度") as pbar:
        for img, mask in pbar:  # 改为 2 个变量解包
            img = img.to(device)
            mask = mask.to(device)  # 直接使用 mask（无需 inst_mask/type_mask）
            #print(f"测试标签维度: {mask.shape}, 取值范围: {mask.min().item()}-{mask.max().item()}")
            mask = torch.clamp(mask, 0, 1)  # 确保标签在 [0,1]
            
            pred = model(img)  # 模型输出 [B, 6, H, W]

            if total_batches == 0:
                print(f"\n预测概率范围: min={pred.min().item():.4f}, max={pred.max().item():.4f}")
            
            # 计算指标（输入为 pred 和 mask）
            batch_metrics = calculate_metrics(pred, mask)
            
            # 累计指标
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            total_batches += 1
            
            pbar.set_postfix({"当前批次Dice": f"{batch_metrics['dice']:.4f}"})
    
    # 计算全局平均指标
    if total_batches == 0:
        logging.warning("测试集无有效样本，无法计算指标！")
        return
    
    avg_metrics = {key: total_metrics[key] / total_batches for key in total_metrics}
    log_msg = (
        f"测试集评估结果 | "
        f"平均准确率: {avg_metrics['accuracy']:.4f} | "
        f"平均灵敏度: {avg_metrics['sensitivity']:.4f} | "
        f"平均Dice: {avg_metrics['dice']:.4f} | "
        f"平均AUC: {avg_metrics['auc']:.4f}"
    )
    logging.info(log_msg)
    print(log_msg)

if __name__ == "__main__":
    test()