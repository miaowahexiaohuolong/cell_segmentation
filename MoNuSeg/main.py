from __future__ import division, print_function, absolute_import

import os
from tqdm import tqdm
import torch
import torch.utils.data as data

from cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from dataset_handle import ImageFolder
import Constants

# 设置使用的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def CE_Net_Train():
    NAME = 'cell_segmentation'

    # 初始化模型 + 损失 + 学习率
    solver = MyFrame(CE_Net_, dice_bce_loss, lr=2e-5)

    # 批大小：每个 GPU 上的 batch × GPU 数
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    # 学习率调度器：每个 epoch lr *= 0.9
    scheduler = torch.optim.lr_scheduler.ExponentialLR(solver.optimizer, gamma=0.9)

    # 加载训练集（确保 root_path 设置正确）
    dataset = ImageFolder(root_path='', mode='train')
    data_loader = data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=12
    )

    # 日志与权重目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    log_file = open('logs/' + NAME + '.log', 'w')

    total_epoch = Constants.TOTAL_EPOCH
    best_loss = Constants.INITAL_EPOCH_LOSS

    for epoch in range(1, total_epoch + 1):
        train_loss_total = 0.0
        solver.net.train()

        for img, mask in tqdm(data_loader, desc=f"Epoch {epoch}/{total_epoch}"):
            solver.set_input(img, mask)
            train_loss, _ = solver.optimize()
            train_loss_total += train_loss.item()

        avg_loss = train_loss_total / len(data_loader)
        log_file.write(f"Epoch {epoch} train_loss: {avg_loss:.6f}\n")
        log_file.flush()

        print(f"Epoch {epoch} | train_loss: {avg_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        # 保存当前模型权重
        solver.save(f'weights/{epoch}.th')

        # 学习率衰减
        scheduler.step()

    log_file.close()
    print("Training finished.")

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    CE_Net_Train()
