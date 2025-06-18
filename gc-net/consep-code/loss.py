
# -*- coding: utf-8 -*-   
import torch
import torch.nn as nn

class MultiLabelBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5, reduction='sum'):
        super(MultiLabelBCEDiceLoss, self).__init__()
        self.alpha = alpha  # BCE权重
        self.smooth = smooth
        self.bce_loss = nn.BCELoss(reduction='none')  # 多标签二分类损失，不进行降维
        self.reduction = reduction  # 损失的合并方式，'sum' 或 'mean'

    def soft_dice_coeff(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        return (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

    def soft_dice_loss(self, y_pred, y_true):
        return 1 - self.soft_dice_coeff(y_pred, y_true)

    def forward(self, y_pred, y_true):
        # y_pred: [B, C, H, W]，sigmoid输出
        # y_true: [B, C, H, W]，独热编码（0/1）
        bce_per_channel = self.bce_loss(y_pred, y_true)  # [B, C, H, W]
        dice_per_channel = torch.zeros(y_pred.size(1), device=y_pred.device)
        for c in range(y_pred.size(1)):
            dice_per_channel[c] = self.soft_dice_loss(y_pred[:, c, :, :], y_true[:, c, :, :])

        bce_loss = bce_per_channel.mean(dim=(0, 2, 3))  # [C]
        total_loss_per_channel = self.alpha * bce_loss + (1 - self.alpha) * dice_per_channel

        if self.reduction == 'sum':
            return total_loss_per_channel.sum()
        elif self.reduction == 'mean':
            return total_loss_per_channel.mean()
        else:
            raise ValueError("reduction must be either 'sum' or 'mean'")