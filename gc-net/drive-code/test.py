import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
from dataset_handle import SegmentationDataset
from model.final_model import CombinedModel
from loss import MultiLabelBCEDiceLoss
import Constants
import os

# 配置日志（与训练日志格式一致）
def setup_test_logger():
    log_dir = Path("./log")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "test.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# 复用训练中的指标计算函数
from main import calculate_metrics

def test():
    setup_test_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"测试设备: {device}")

    # 初始化模型并加载最佳权重
    model = CombinedModel().to(device)
    best_model_path = os.path.join(Constants.WEIGHTS_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到最佳模型文件 {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()  # 切换为评估模式

    # 定义损失函数（可选，用于测试损失计算）
    loss_fn = MultiLabelBCEDiceLoss(alpha=0.5).to(device)

    # 加载测试数据集
    test_dataset = SegmentationDataset(root_path=Constants.DATA_ROOT, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=Constants.BATCHSIZE_PER_CARD,
        shuffle=False,  # 测试时不打乱数据
        num_workers=16,
        pin_memory=True
    )

    # 初始化指标统计
    total_loss = 0.0
    total_metrics = {
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'dice': 0.0,
        'auc': 0.0
    }
    batch_count = 0

    # 测试循环
    with torch.no_grad():  # 禁用梯度计算
        with tqdm(test_loader, desc="测试进度") as pbar:
            for img, mask in pbar:
                img = img.to(device)
                mask = mask.to(device)

                # 模型推理
                pred = model(img)
                loss = loss_fn(pred, mask)  # 可选：计算测试损失

                # 计算批次指标
                batch_metrics = calculate_metrics(pred, mask)
                
                # 累加统计
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]
                batch_count += 1

                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{batch_metrics['dice']:.4f}"
                })

    # 计算平均指标
    if batch_count == 0:
        logging.error("测试数据为空！")
        return

    avg_loss = total_loss / batch_count
    avg_metrics = {key: total_metrics[key] / batch_count for key in total_metrics}

    # 输出测试结果
    log_msg = (
        f"测试结果 | Loss: {avg_loss:.4f} | "
        f"Accuracy: {avg_metrics['accuracy']:.4f} | "
        f"Sensitivity: {avg_metrics['sensitivity']:.4f} | "
        f"Dice: {avg_metrics['dice']:.4f} | "
        f"AUC: {avg_metrics['auc']:.4f}"
    )
    logging.info(log_msg)
    print(log_msg)

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    test()