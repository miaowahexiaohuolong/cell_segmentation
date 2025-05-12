from __future__ import division, print_function, absolute_import

import os
import torch
import torch.utils.data as data
from tqdm import tqdm

from cenet import CE_Net_  # 确保模型输出通道数=8（num_classes）
from framework import MyFrame  # 确保MyFrame支持多类别损失和设备同步
from loss import MultiClassCombinedLoss, pixel_accuracy  # 导入多类别组合损失
from dataset_handle import CoNSePDataset  # 确保数据集返回正确格式的标签
import Constants

# ------------------- 关键配置 -------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 使用第0块GPU（无GPU时自动切换CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动判断设备



def CE_Net_Train():
    NAME = 'consep_nuclei_segmentation'

    # ------------------- 初始化模型与损失函数 -------------------
    model = CE_Net_(num_classes=8).to(device)
    loss_fn = MultiClassCombinedLoss(num_classes=8).to(device)
    solver = MyFrame(model, loss_fn, lr=2e-5)

    # ------------------- 数据加载 -------------------
    batchsize = 2 if torch.cuda.is_available() else 1
    dataset = CoNSePDataset(
        root_path=Constants.ROOT,
        mode='Train',
        img_size=Constants.IMG_SIZE
    )
    data_loader = data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0
    )

    # ------------------- 训练配置 -------------------
    os.makedirs('logs', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    log_file = open(f'logs/{NAME}.log', 'w')

    total_epoch = Constants.TOTAL_EPOCH
    best_loss = float('inf')

    # ------------------- 训练循环（添加准确率） -------------------
    for epoch in range(1, total_epoch + 1):
        train_loss_total = 0.0
        train_acc_total = 0.0  # 累计准确率
        solver.net.train()

        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{total_epoch}")
        for img, inst_mask, type_mask in pbar:
            img = img.to(device)
            type_mask = type_mask.to(device)

            solver.set_input(img, type_mask)
            train_loss, pred = solver.optimize()  # pred是模型输出logits（[B,8,H,W]）
            
            # 计算准确率
            acc = pixel_accuracy(pred, type_mask)  # type_mask是真实标签（[B,1,H,W]）
            train_acc_total += acc.item()

            train_loss_total += train_loss.item()
            pbar.set_postfix({
                "Loss": f"{train_loss.item():.4f}",
                "Acc": f"{acc.item():.4f}"  # 实时显示当前批次的准确率
            })

        # 计算平均损失和平均准确率
        avg_loss = train_loss_total / len(data_loader)
        avg_acc = train_acc_total / len(data_loader)
        log_file.write(f"Epoch {epoch} | train_loss: {avg_loss:.6f} | train_acc: {avg_acc:.4f}\n")
        log_file.flush()

        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Accuracy: {avg_acc:.4f}")

        # 保存最佳模型（根据损失或准确率，可自定义）
        if avg_loss < best_loss:
            best_loss = avg_loss
            solver.save(f'weights/{NAME}_best.th')
            print(f"Best model saved at epoch {epoch} (Loss: {avg_loss:.6f})")

        # 保存每轮模型（可选）
        # solver.save(f'weights/{NAME}_epoch_{epoch}.th')

    log_file.close()
    print("Training finished.")


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    CE_Net_Train()