import torch
import numpy as np
import os
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset_handle import CoNSePDataset
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
        encoding="utf-8"  # 关键：确保中文日志正常编码
    )

# 修复后的指标计算函数（增加维度检查和扩展）
def calculate_metrics(pred, mask, threshold=0.5):
    """
    pred: [B, 6, H, W] 概率值（0-1）
    mask: [B, 1, H, W] 或 [B, 6, H, W] 标签（需处理单通道情况）
    """
    pred = pred.detach().cpu().numpy()
    mask = mask.cpu().numpy()
    
    # 检查mask是否为单通道，如果是则扩展到与pred相同的通道数
    if mask.shape[1] == 1 and pred.shape[1] > 1:
        mask = np.repeat(mask, pred.shape[1], axis=1)
    
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
            p = pred[b, c].ravel()  # 预测概率（展平为1D）
            m = mask[b, c].ravel()  # 真实标签（展平为1D）
            
            # 跳过全0或全1的无效样本（避免AUC计算错误）
            if np.unique(m).size < 2:
                continue
            
            p_bin = (p > threshold).astype(np.float32)  # 二值化预测
            tp = np.sum(p_bin * m)          # 真阳性
            fp = np.sum(p_bin * (1 - m))    # 假阳性
            fn = np.sum((1 - p_bin) * m)    # 假阴性
            tn = np.sum((1 - p_bin) * (1 - m))  # 真阴性
            
            # 计算各指标（按类别累加）
            metrics['dice'][c] += (2 * tp) / (2 * tp + fp + fn + 1e-8)
            metrics['accuracy'][c] += (tp + tn) / (tp + tn + fp + fn + 1e-8)
            metrics['sensitivity'][c] += tp / (tp + fn + 1e-8) if (tp + fn) != 0 else 0.0
            metrics['auc'][c] += roc_auc_score(m, p)
            metrics['count'][c] += 1  # 有效样本计数+1
    
    # 计算类别平均指标（忽略无有效样本的类别）
    avg_metrics = {}
    for key in ['accuracy', 'sensitivity', 'dice', 'auc']:
        valid_mask = metrics['count'] > 0  # 有效类别掩码
        if np.any(valid_mask):
            class_avg = np.mean(metrics[key][valid_mask] / metrics['count'][valid_mask])
            avg_metrics[key] = np.clip(class_avg, 0, 1)  # 确保值在0~1之间
        else:
            avg_metrics[key] = 0.0  # 无有效样本时指标为0
    return avg_metrics

def test():
    setup_test_logger()  # 初始化测试日志
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备配置
    logging.info(f"测试设备: {device}")
    
    # 加载训练好的模型
    model = CombinedModel().to(device)
    model_path = os.path.join(Constants.WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先完成训练！")
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"成功加载模型: {model_path}")
    
    # 加载测试数据集（与训练集同结构，mode='test'）
    test_dataset = CoNSePDataset(root_path=Constants.DATA_ROOT, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=Constants.BATCHSIZE_PER_CARD,  # 与训练批次一致
        shuffle=False,  # 测试无需打乱顺序
        num_workers=16,
        pin_memory=True
    )
    
    model.eval()  # 切换为评估模式（关闭Dropout等）
    total_metrics = {  # 累计全局指标
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'dice': 0.0,
        'auc': 0.0
    }
    total_batches = 0  # 总批次数
    
    # 测试主循环（禁用梯度计算）
    with torch.no_grad(), tqdm(test_loader, desc="测试进度") as pbar:
        for img, inst_mask, type_mask in pbar:
            # 数据加载到设备（与训练一致）
            img = img.to(device)
            type_mask = type_mask.to(device)  # 使用类别掩码计算指标（与训练逻辑一致）
            type_mask = torch.clamp(type_mask, 0, 1)  # 确保标签在[0,1]
            
            # 模型前向传播
            pred = model(img)  # 输出: [B, 6, H, W] 概率值
            
            # 打印维度信息用于调试
            logging.debug(f"预测维度: {pred.shape}, 标签维度: {type_mask.shape}")
            
            # 计算当前批次指标
            batch_metrics = calculate_metrics(pred, type_mask)
            
            # 累计指标和批次数
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            total_batches += 1
            
            # 更新进度条（显示当前批次Dice）
            pbar.set_postfix({"当前批次Dice": f"{batch_metrics['dice']:.4f}"})
    
    # 计算全局平均指标（避免除零错误）
    if total_batches == 0:
        logging.warning("测试集无有效样本，无法计算指标！")
        return
    
    avg_metrics = {key: total_metrics[key] / total_batches for key in total_metrics}
    
    # 记录测试结果到日志（含中文描述）
    log_msg = (
        f"测试集评估结果 | "
        f"平均准确率: {avg_metrics['accuracy']:.4f} | "
        f"平均灵敏度: {avg_metrics['sensitivity']:.4f} | "
        f"平均Dice: {avg_metrics['dice']:.4f} | "
        f"平均AUC: {avg_metrics['auc']:.4f}"
    )
    logging.info(log_msg)  # 写入日志文件（中文正常显示）
    print(log_msg)  # 打印到控制台

if __name__ == "__main__":
    test()