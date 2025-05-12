import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime

# 导入自定义模块（确保路径正确）
from cenet import CE_Net_  # 模型定义
from dataset_handle import CoNSePDataset  # 数据集类
import Constants  # 常量（如ROOT, IMG_SIZE等）


# ------------------- 指标计算函数 -------------------
def calculate_metrics(y_pred, y_true, num_classes=8):
    """计算多类别分割的四项指标（Accuracy, Sensitivity, Dice, AUC）"""
    y_pred_flat = y_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [N, C]
    y_true_flat = y_true.squeeze(1).reshape(-1).long()                 # [N]

    TP = torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)
    FP = torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)
    FN = torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)
    AUC = torch.zeros(num_classes, dtype=torch.float32, device=y_pred.device)

    for c in range(num_classes):
        pred_c = torch.argmax(y_pred_flat, dim=1) == c
        true_c = y_true_flat == c

        TP[c] = torch.sum(pred_c & true_c).float()
        FP[c] = torch.sum(pred_c & ~true_c).float()
        FN[c] = torch.sum(~pred_c & true_c).float()

        if torch.any(true_c):
            prob_c = y_pred_flat[:, c].cpu().numpy()
            true_binary = true_c.cpu().numpy().astype(int)
            AUC[c] = roc_auc_score(true_binary, prob_c)
        else:
            AUC[c] = 0.0

    global_accuracy = torch.sum(torch.argmax(y_pred_flat, dim=1) == y_true_flat) / y_true_flat.numel()
    Sensitivity = TP / (TP + FN + 1e-8)
    Dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    valid_classes = (TP + FN) > 0
    avg_sensitivity = Sensitivity[valid_classes].mean() if torch.any(valid_classes) else 0.0
    avg_dice = Dice[valid_classes].mean() if torch.any(valid_classes) else 0.0
    avg_auc = AUC[valid_classes].mean() if torch.any(valid_classes) else 0.0

    return {
        "global_accuracy": global_accuracy.item(),
        "avg_sensitivity": avg_sensitivity.item(),
        "avg_dice": avg_dice.item(),
        "avg_auc": avg_auc.item()
    }


# ------------------- 测试主函数（带日志记录） -------------------
def test(weight_path, num_classes=8, log_dir="logs"):
    # 创建日志目录（若不存在）
    os.makedirs(log_dir, exist_ok=True)
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_log_{timestamp}.txt")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 初始化模型并加载权重
        model = CE_Net_(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        # 加载测试数据集
        test_dataset = CoNSePDataset(
            root_path=Constants.ROOT,
            mode='Test',
            img_size=Constants.IMG_SIZE
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0
        )

        # 累计指标
        total_metrics = {
            "global_accuracy": 0.0,
            "avg_sensitivity": 0.0,
            "avg_dice": 0.0,
            "avg_auc": 0.0
        }
        num_samples = len(test_loader)

        # 遍历测试数据
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for img, inst_mask, type_mask in pbar:
                img = img.to(device)
                type_mask = type_mask.to(device)

                pred_logits = model(img)
                pred_probs = F.softmax(pred_logits, dim=1)

                metrics = calculate_metrics(pred_probs, type_mask, num_classes)

                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                pbar.set_postfix({
                    "Acc": f"{metrics['global_accuracy']:.4f}",
                    "Dice": f"{metrics['avg_dice']:.4f}"
                })

        # 计算平均值
        avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}

        # 构造输出内容
        output = [
            f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Weight Path: {weight_path}",
            f"Number of Classes: {num_classes}",
            "="*50,
            f"Average Accuracy:    {avg_metrics['global_accuracy']:.4f}",
            f"Average Sensitivity: {avg_metrics['avg_sensitivity']:.4f}",
            f"Average Dice:        {avg_metrics['avg_dice']:.4f}",
            f"Average AUC:         {avg_metrics['avg_auc']:.4f}",
            "="*50
        ]

        # 打印并写入日志
        with open(log_path, "w") as f:
            for line in output:
                print(line)
                f.write(line + "\n")

        print(f"\n测试日志已保存至: {log_path}")

    except Exception as e:
        # 捕获异常并记录到日志
        error_msg = f"Test Failed: {str(e)}"
        with open(log_path, "w") as f:
            f.write(error_msg)
        print(error_msg)


# ------------------- 运行测试 -------------------
if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    WEIGHT_PATH = "weights/consep_nuclei_segmentation_best.th"  # 训练保存的权重路径
    NUM_CLASSES = 8  # 类别数
    LOG_DIR = "logs"  # 日志保存目录

    # 执行测试
    test(WEIGHT_PATH, NUM_CLASSES, LOG_DIR)