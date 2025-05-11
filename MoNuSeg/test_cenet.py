import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from framework import MyFrame
from loss import dice_bce_loss
from cenet import CE_Net_
from dataset_handle import FastDataset  # 或者 ImageFolder，看你用哪个

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def compute_metrics(pred, mask, threshold=0.65):
    prob = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()

    p = pred_bin.cpu().numpy().flatten()
    g = mask.cpu().numpy().flatten()
    pr = prob.cpu().numpy().flatten()

    TP = np.sum((p == 1) & (g == 1))
    TN = np.sum((p == 0) & (g == 0))
    FP = np.sum((p == 1) & (g == 0))
    FN = np.sum((p == 0) & (g == 1))

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    sen = TP / (TP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    try:
        auc = roc_auc_score(g.astype(np.uint8), pr)
    except ValueError:
        auc = 0.0

    return acc, sen, dice, auc


def evaluate_model(model, dataloader, device):
    model.eval()
    accs, sens, dices, aucs = [], [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="[Test]"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)

            acc, sen, dice, auc = compute_metrics(preds, masks)
            accs.append(acc)
            sens.append(sen)
            dices.append(dice)
            aucs.append(auc)

    return {
        "Average Accuracy": np.mean(accs),
        "Average Sensitivity": np.mean(sens),
        "Average Dice": np.mean(dices),
        "Average AUC": np.mean(aucs)
    }


def main():
    # 加载模型
    solver = MyFrame(CE_Net_, dice_bce_loss, lr=1e-4)
    solver.load('weights/200.th')  # 换成你的模型路径
    solver.net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver.net.to(device)

    # 加载测试数据
    test_dataset = FastDataset('preprocessed_test')  # 或 ImageFolder(root_path='', mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 计算四个指标
    results = evaluate_model(solver.net, test_loader, device)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()
